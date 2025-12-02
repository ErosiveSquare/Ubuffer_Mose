# /your_project_root/agent/ablation_mose_v3.py
# 2025.11.8 12:00 LJY

import os
from copy import deepcopy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from losses.loss import sup_con_loss
from models.uncertainty_buffer import UncertaintyBuffer
from utils import get_transform
from utils.rotation_transform import RandomFlip

class AblationMOSE_v3(object):
    """
    Ablation-ready version of the MOSE v3 agent.
    This class integrates configurable switches to enable/disable key modules:
    1. Knowledge Distillation (KD)
    2. Internal Distillation Mechanisms (IDM)
    3. Uncertainty-based Dual-Buffer Strategy (U-DB)
    """

    def __init__(self, model: nn.Module, buffer, optimizer, input_size, args):
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer

        # 从 args 中提取核心参数
        self.ins_t = args.ins_t
        self.epoch = args.epoch
        self.expert = int(args.expert)
        self.n_classes_num = args.n_classes
        self.use_ncm = (args.classifier == 'ncm')
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_cur_task = (self.buffer_batch_size // 2) - args.batch_size

        # 数据集相关配置
        if args.dataset == "cifar10":
            self.total_samples, self.print_num = 10000, 2000
        elif "cifar100" in args.dataset:
            self.total_samples, self.print_num = 5000, 500
        elif args.dataset == "tiny_imagenet":
            self.total_samples, self.print_num = 1000, 100
        elif args.dataset == "mnist":
            self.total_samples, self.print_num = 6000, 1200
        else:
            self.total_samples, self.print_num = 1000, 200  # Default

        self.transform = get_transform(args.augmentation, input_size)
        self.total_step = 0
        self.class_holder = []
        self.scaler = GradScaler()
        self.main_buffer_uncertainties = {}

        # ==================== 消融实验控制开关 (来自args) ====================
        # 这些开关将通过命令行参数（如 --no_kd）在主脚本中被设置
        self.use_kd = getattr(args, 'use_kd', True)
        self.use_idm = getattr(args, 'use_idm', True)
        self.use_udb = getattr(args, 'use_udb', True)

        print("\n" + "=" * 60)
        print("          Ablation Study Configuration for MOSE v3")
        print("-" * 60)
        print(f"  - Knowledge Distillation (KD)        : {'Enabled' if self.use_kd else 'Disabled'}")
        print(f"  - Internal Distillation (IDM)        : {'Enabled' if self.use_idm else 'Disabled'}")
        print(f"  - Uncertainty Dual-Buffer (U-DB)     : {'Enabled' if self.use_udb else 'Disabled'}")
        print("=" * 60 + "\n")
        # ====================================================================

        # 模块1: 知识蒸馏 (KD)
        self.old_model = None
        self.distill_temperature = getattr(args, 'distill_temperature', 2.0)
        self.distill_weight = getattr(args, 'distill_weight', 1.0) if self.use_kd else 0.0

        # 模块2: 内部蒸馏 (IDM)
        self.deep_to_shallow_temperature = getattr(args, 'deep_to_shallow_temperature', 2.0)
        self.deep_to_shallow_weight = getattr(args, 'deep_to_shallow_weight', 0.5) if self.use_idm else 0.0

        self.teacher_student_temperature = getattr(args, 'teacher_student_temperature', 4.0)
        self.teacher_student_kl_weight = getattr(args, 'teacher_student_kl_weight', 0.5) if self.use_idm else 0.0
        self.teacher_student_l2_weight = getattr(args, 'teacher_student_l2_weight', 0.1) if self.use_idm else 0.0

        # 模块3: 不确定性双缓冲区 (U-DB)
        if self.use_udb:
            self.u_buffer_capacity = int(getattr(args, 'u_buffer_capacity', 64))
            self.u_buffer_class_quota = int(getattr(args, 'u_buffer_class_quota', 7))
            buffer_device = self.buffer.bx.device if hasattr(self.buffer, 'bx') else torch.device('cuda')
            self.u_buffer = UncertaintyBuffer(capacity=self.u_buffer_capacity, device=buffer_device)

    def train(self, task_id, train_loader):
        """主训练函数，在每个任务开始时调用"""
        self.model.train()
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log = self.train_any_task(task_id, train_loader, epoch)
            train_log_holder.extend(epoch_log)

        # 任务结束后，如果启用KD，则保存模型作为教师
        if self.use_kd:
            self.old_model = deepcopy(self.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False

        return train_log_holder

    def train_any_task(self, task_id, train_loader, epoch):
        """每个epoch的训练循环"""
        num_d = 0
        epoch_log_holder = []
        if epoch == 0:
            self.new_class_holder = []

        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            # 更新已见类别列表
            y_list = y.tolist()
            newly_seen = [cls for cls in y_list if cls not in self.class_holder]
            if newly_seen:
                self.class_holder.extend(newly_seen)
                self.new_class_holder.extend(newly_seen)

            x_orig, y_orig = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            batch_uncertainty = None

            # U-DB模块控制更新次数，否则只更新一次
            num_updates = 2 if self.use_udb and task_id > 0 and len(self.u_buffer) > 0 else 1

            for update_round in range(num_updates):
                self.optimizer.zero_grad()
                loss = torch.tensor(0.0, device=x_orig.device)
                loss_log = self._initialize_loss_log()

                # 仅当缓冲区有数据或在第一个任务时进行训练
                if len(self.buffer) > 0 or task_id == 0:
                    with autocast():
                        # --- 数据准备 ---
                        mem_x, mem_y, main_buffer_indices = None, None, None
                        if task_id > 0:
                            if self.use_udb and update_round == 1:
                                # U-DB 第二轮：从U-Buffer采样
                                mem_x, mem_y, u_buffer_indices = self._sample_u_buffer_with_quota(
                                    self.buffer_batch_size)
                            else:
                                # 第一轮或U-DB禁用：从主缓冲区采样
                                mem_x, mem_y, _, main_buffer_indices = self.buffer.sample(self.buffer_batch_size,
                                                                                          exclude_task=task_id,
                                                                                          ret_ind=True)

                        if mem_x is not None and mem_x.numel() > 0:
                            all_x, all_y = torch.cat((x_orig, mem_x)), torch.cat((y_orig, mem_y))
                        else:
                            all_x, all_y = x_orig, y_orig

                        # --- 数据增强 ---
                        all_x_aug, all_y_aug = self._augment_batch(all_x, all_y)

                        # --- 前向传播 ---
                        feat_list = self.model.features(all_x_aug)
                        proj_list = self.model.head(feat_list, use_proj=True)
                        pred_list = self.model.head(feat_list, use_proj=False)
                        stu_feat = self.model.final_addaption_layer(feat_list[self.expert])

                        # --- 损失计算 ---
                        # 模块2: 内部蒸馏 (IDM)
                        if self.use_idm:
                            idm_loss, d2s_loss, ts_kl_loss, ts_l2_loss = self._calculate_idm_loss(pred_list, feat_list)
                            loss += idm_loss
                            loss_log['train/d2s'] += d2s_loss.item()
                            loss_log['train/ts_kl'] += ts_kl_loss.item()
                            loss_log['train/ts_l2'] += ts_l2_loss.item()

                        # 基础损失循环 (CE, SupCon, Align, KD)
                        for i in range(len(feat_list)):
                            # 基础损失
                            ins_loss = sup_con_loss(proj_list[i], self.ins_t, all_y_aug)
                            ce_loss = F.cross_entropy(pred_list[i], all_y_aug)
                            align_loss = torch.dist(F.normalize(stu_feat, dim=1),
                                                    F.normalize(feat_list[i].detach(), dim=1),
                                                    p=2) if i != self.expert else torch.tensor(0.0)

                            # 模块1: 知识蒸馏 (KD)
                            kd_loss = torch.tensor(0.0, device=x_orig.device)
                            if self.use_kd and self.old_model is not None and task_id > 0:
                                with torch.no_grad():
                                    old_pred = self.old_model.head(self.old_model.features(all_x_aug), use_proj=False)[
                                        i]
                                mask = torch.ones(pred_list[i].size(1), dtype=torch.bool, device=pred_list[i].device)
                                if self.new_class_holder: mask[self.new_class_holder] = False
                                if mask.sum() > 0:
                                    kd_loss = F.kl_div(
                                        F.log_softmax(pred_list[i][:, mask] / self.distill_temperature, 1),
                                        F.softmax(old_pred[:, mask] / self.distill_temperature, 1),
                                        reduction='batchmean') * (self.distill_temperature ** 2)

                            loss += ins_loss + ce_loss + (kd_loss * self.distill_weight) + align_loss
                            loss_log['train/ins'] += ins_loss.item();
                            loss_log['train/ce'] += ce_loss.item()
                            loss_log['train/distill'] += (kd_loss * self.distill_weight).item();
                            loss_log['train/align'] += align_loss.item()

                if torch.is_grad_enabled():  # 确保有损失才反向传播
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # --- 模块3: U-DB 不确定性更新 ---
                if self.use_udb and task_id > 0:
                    with torch.no_grad():
                        if update_round == 0:  # 仅在第一轮计算新样本不确定性
                            batch_uncertainty = self._feature_inconsistency_scores(x_orig)
                            self.u_buffer.add_batch(x_orig, y_orig, batch_uncertainty)

                            if main_buffer_indices is not None and main_buffer_indices.numel() > 0:
                                main_scores = self._feature_inconsistency_scores(mem_x)
                                if hasattr(self.buffer, 'update_uncertainty'):
                                    self.buffer.update_uncertainty(main_buffer_indices, main_scores)

                        elif update_round == 1 and u_buffer_indices is not None and u_buffer_indices.numel() > 0:
                            u_scores = self._feature_inconsistency_scores(mem_x)
                            self.u_buffer.update_uncertainty(u_buffer_indices, u_scores)

                loss_log['train/loss'] = loss.item()
                epoch_log_holder.append(loss_log)

            # --- 主缓冲区填充 ---
            if epoch == 0:
                add_kwargs = {}
                if self.use_udb and batch_uncertainty is not None:
                    add_kwargs['uncertainty'] = batch_uncertainty.detach()
                self.buffer.add_reservoir(x=x_orig.detach(), y=y_orig.detach(), logits=None, t=task_id, **add_kwargs)

            self.total_step += 1

        return epoch_log_holder

    # ==================== 辅助函数 (清晰化逻辑) ====================

    def _initialize_loss_log(self):
        """初始化用于记录的损失字典"""
        return {
            'step': self.total_step, 'train/loss': 0., 'train/ins': 0.,
            'train/ce': 0., 'train/distill': 0., 'train/d2s': 0.,
            'train/ts_kl': 0., 'train/ts_l2': 0., 'train/align': 0.,
        }

    def _augment_batch(self, x, y):
        """对批次数据进行数据增强"""
        x_aug = RandomFlip(x, 2)
        y_aug = y.repeat(2)
        x_aug = torch.cat((x_aug, self.transform(x_aug)))
        y_aug = torch.cat((y_aug, y_aug))
        return x_aug.detach(), y_aug.detach()

    def _calculate_idm_loss(self, pred_list, feat_list):
        """计算内部蒸馏损失"""
        device = pred_list[0].device
        d2s_loss, ts_kl_loss, ts_l2_loss = torch.tensor(0.0, device=device), torch.tensor(0.0,
                                                                                          device=device), torch.tensor(
            0.0, device=device)

        if len(pred_list) < 2:
            total_idm_loss = torch.tensor(0.0, device=device)
            return total_idm_loss, d2s_loss, ts_kl_loss, ts_l2_loss

        # Deep-to-Shallow (D2S)
        for i in range(1, len(pred_list)):
            s_logits, d_logits = pred_list[i - 1].detach(), pred_list[i]
            d2s_loss += F.kl_div(F.log_softmax(d_logits / self.deep_to_shallow_temperature, 1),
                                 F.softmax(s_logits / self.deep_to_shallow_temperature, 1), reduction='batchmean') * (
                                    self.deep_to_shallow_temperature ** 2)

        # Teacher-Student (TSD)
        t_logits, t_feat = pred_list[-1].detach(), feat_list[-1].detach()
        s_logits, s_feat = pred_list[0], feat_list[0]
        ts_kl_loss = F.kl_div(F.log_softmax(s_logits / self.teacher_student_temperature, 1),
                              F.softmax(t_logits / self.teacher_student_temperature, 1), reduction='batchmean') * (
                                 self.teacher_student_temperature ** 2)
        ts_l2_loss = F.mse_loss(s_feat, t_feat)

        total_idm_loss = d2s_loss * self.deep_to_shallow_weight + \
                         ts_kl_loss * self.teacher_student_kl_weight + \
                         ts_l2_loss * self.teacher_student_l2_weight

        return total_idm_loss, d2s_loss, ts_kl_loss, ts_l2_loss

    # ==================== 测试与评估函数 (之前代码复制的) ====================

    def test_buffer(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_buffer_task(j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_buffer_task_mean(j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def test_buffer_task(self, i, feat_id):
        # test specific layer's output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        _ret = self.buffer.onlysample(self.buffer.current_index, task=i)
        x_i, y_i = _ret[0], _ret[1]

        if self.use_ncm:
            class_means = self.class_means_ls[feat_id]
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                features = self.model.features(x)[feat_id]
                features = F.normalize(features, dim=1)
                features = features.unsqueeze(2)
                means = torch.stack([class_means[cls] for cls in self.class_holder])
                means = torch.stack([means] * x.size(0))
                means = means.transpose(1, 2)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1).squeeze(1)
                pred = dists.min(1)[1]
                class_index_to_label = torch.tensor(self.class_holder, device=pred.device, dtype=pred.dtype)
                pred = class_index_to_label[pred]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        else:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                model_output = self.model(x)
                pred = model_output[0][feat_id]
                pred = pred.max(1, keepdim=True)[1]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_buffer_task_mean(self, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        _ret = self.buffer.onlysample(self.buffer.current_index, task=i)
        x_i, y_i = _ret[0], _ret[1]

        if self.use_ncm:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                features_ls = self.model.features(x)
                dists_ls = []

                for feat_id in range(4):
                    class_means = self.class_means_ls[feat_id]
                    features = features_ls[feat_id]
                    features = F.normalize(features, dim=1)
                    features = features.unsqueeze(2)
                    means = torch.stack([class_means[cls] for cls in self.class_holder])
                    means = torch.stack([means] * x.size(0))
                    means = means.transpose(1, 2)
                    features = features.expand_as(means)
                    dists = (features - means).pow(2).sum(1).squeeze(1)
                    dists_ls.append(dists)

                dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
                dists = dists_ls.mean(dim=1).squeeze(1)
                pred = dists.min(1)[1]
                class_index_to_label = torch.tensor(self.class_holder, device=pred.device, dtype=pred.dtype)
                pred = class_index_to_label[pred]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        else:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                model_output = self.model(x)
                expert_logits = model_output[0]
                pred = torch.stack(expert_logits, dim=1)
                pred = pred.mean(dim=1).squeeze(1)
                pred = pred.max(1, keepdim=True)[1]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_train(self, i, task_loader, feat_ids=[0,1,2,3]):
        # train accuracy of current task i
        self.model.eval()
        all_acc_list = {'step': self.total_step}

        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                acc = self.test_model(task_loader[i]['train'], i, feat_id=feat_id)
                acc_list[i] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            acc = self.test_model_mean(task_loader[i]['train'], i)
            acc_list[i] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def test_model(self, loader, i, feat_id):
        # test specific layer's output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            class_means = self.class_means_ls[feat_id]
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                features = self.model.features(data)[feat_id]
                features = F.normalize(features, dim=1)
                features = features.unsqueeze(2)
                means = torch.stack([class_means[cls] for cls in self.class_holder])
                means = torch.stack([means] * data.size(0))
                means = means.transpose(1, 2)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1)
                pred = dists.min(1)[1]
                class_index_to_label = torch.tensor(self.class_holder, device=data.device, dtype=target.dtype)
                pred = class_index_to_label[pred]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        else:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                model_output = self.model(data)
                pred = model_output[0][feat_id]
                pred = pred.max(1, keepdim=True)[1]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test(self, i, task_loader, feat_ids=[0, 1, 2, 3]):
        self.model.eval()
        if self.use_ncm:
            print("\nCalculating class means for NCM classifier...\n")
            self.class_means_ls = [{} for _ in range(4)]
            class_inputs = {cls: [] for cls in self.class_holder}
            for x_buf, y_buf in zip(self.buffer.x, self.buffer.y_int):
                class_inputs[y_buf.item()].append(x_buf)

            for cls, inputs in class_inputs.items():
                if not inputs: continue
                features = [[] for _ in range(4)]
                for ex in inputs:
                    return_features_ls = self.model.features(ex.unsqueeze(0).cuda())
                    for feat_id in range(4):
                        feature = F.normalize(return_features_ls[feat_id].detach(), dim=1)
                        features[feat_id].append(feature.squeeze())

                for feat_id in range(4):
                    if features[feat_id]:
                        mu_y = torch.stack(features[feat_id]).mean(0)
                    else:  # Fallback if class has no samples in buffer
                        mu_y = torch.randn(self.model.features(inputs[0].unsqueeze(0).cuda())[feat_id].shape[1],
                                           device='cuda')
                    self.class_means_ls[feat_id][cls] = F.normalize(mu_y.reshape(1, -1), dim=1).squeeze()

        all_acc_list = {'step': self.total_step}
        for feat_id in feat_ids:
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_model(task_loader[j]['test'], j, feat_id=feat_id)
                    acc_list[j] = acc.item()
                all_acc_list[str(feat_id)] = acc_list

        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_mean(task_loader[j]['test'], j)
                acc_list[j] = acc.item()
            all_acc_list['mean'] = acc_list

        return acc_list, all_acc_list

    def test_model_mean(self, loader, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()
                features_ls = self.model.features(data)
                dists_ls = []

                for feat_id in range(4):
                    class_means = self.class_means_ls[feat_id]
                    features = features_ls[feat_id]
                    features = F.normalize(features, dim=1)
                    features = features.unsqueeze(2)
                    means = torch.stack([class_means[cls] for cls in self.class_holder])
                    means = torch.stack([means] * data.size(0))
                    means = means.transpose(1, 2)
                    features = features.expand_as(means)
                    dists = (features - means).pow(2).sum(1)
                    dists_ls.append(dists)

                dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
                dists = dists_ls.mean(dim=1).squeeze(1)
                pred = dists.min(1)[1]
                class_index_to_label = torch.tensor(self.class_holder, device=data.device, dtype=target.dtype)
                pred = class_index_to_label[pred]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        else:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                model_output = self.model(data)
                expert_logits = model_output[0]
                pred = torch.stack(expert_logits, dim=1)
                pred = pred.mean(dim=1)
                pred = pred.max(1, keepdim=True)[1]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def save_checkpoint(self, save_path = './outputs/final.pt'):
        print(f"Save checkpoint to: {save_path}")
        ckpt_dict = {
            'model': self.model.state_dict(),
            'buffer': self.buffer.state_dict(),
        }
        folder, file_name = os.path.split(save_path)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(ckpt_dict, save_path)

    def load_checkpoint(self, load_path = './outputs/final.pt'):
        print(f"Load checkpoint from: {load_path}")
        ckpt_dict = torch.load(load_path)
        self.model.load_state_dict(ckpt_dict['model'])
        self.buffer.load_state_dict(ckpt_dict['buffer'])

    def _sample_u_buffer_with_quota(self, request_size: int):
        if request_size <= 0 or len(self.u_buffer) == 0:
            empty_x = torch.empty(0, device=self.u_buffer.device)
            empty_y = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            empty_idx = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            return empty_x, empty_y, empty_idx

        total = len(self.u_buffer)
        all_x, all_y, all_idx = self.u_buffer.sample(total)
        if all_x.numel() == 0:
            empty_x = torch.empty(0, device=self.u_buffer.device)
            empty_y = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            empty_idx = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            return empty_x, empty_y, empty_idx

        u_values = torch.tensor([self.u_buffer.u_list[int(i.item())] for i in all_idx], device=self.u_buffer.device)
        order = torch.argsort(u_values, descending=True)

        selected_positions = []
        class_counts = {}
        class_lowest = {}

        for pos in order:
            cls = int(all_y[pos].item())
            score = float(u_values[pos].item())
            current = class_counts.get(cls, 0)
            if current < self.u_buffer_class_quota:
                selected_positions.append(int(pos.item()))
                class_counts[cls] = current + 1
                class_lowest.setdefault(cls, (score, len(selected_positions) - 1))
                if score < class_lowest[cls][0]:
                    class_lowest[cls] = (score, len(selected_positions) - 1)
            else:
                lowest_score, lowest_idx = class_lowest[cls]
                if score > lowest_score:
                    selected_positions[lowest_idx] = int(pos.item())
                    class_lowest[cls] = (score, lowest_idx)
            if len(selected_positions) >= request_size:
                break

        if len(selected_positions) == 0:
            empty_x = torch.empty(0, device=self.u_buffer.device)
            empty_y = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            empty_idx = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            return empty_x, empty_y, empty_idx

        selected_positions_tensor = torch.tensor(selected_positions, device=all_idx.device, dtype=torch.long)
        selected_x = all_x[selected_positions_tensor]
        selected_y = all_y[selected_positions_tensor]
        selected_idx = all_idx[selected_positions_tensor]
        return selected_x, selected_y, selected_idx

    def _feature_inconsistency_scores(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs is None or inputs.numel() == 0:
            ref_device = inputs.device if inputs is not None else next(self.model.parameters()).device
            return torch.empty(0, device=ref_device)
        device = inputs.device
        with torch.no_grad():
            views = RandomFlip(inputs, 2)
            views = torch.cat((views, self.transform(views)))
            B = inputs.size(0)
            K = views.size(0) // B
            feat_list = self.model.features(views)
            expert_feat = feat_list[self.expert]
            if hasattr(self.model, 'final_addaption_layer'):
                expert_feat = self.model.final_addaption_layer(expert_feat)
            expert_feat = expert_feat.reshape(K, B, -1)
            expert_feat = F.normalize(expert_feat, dim=-1)
            expert_feat = expert_feat.transpose(0, 1)  # B x K x D
            pairwise = torch.matmul(expert_feat, expert_feat.transpose(-1, -2))
            if K < 2:
                return torch.zeros(B, device=device)
            mask = torch.triu(torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1)
            distances = (1.0 - pairwise)[:, mask]
            scores = distances.sum(dim=1) * (2.0 / (K * (K - 1)))
            return scores

