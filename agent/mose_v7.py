import os
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from losses.loss import sup_con_loss
from models.uncertainty_buffer import UncertaintyBuffer
from utils import get_transform
from utils.rotation_transform import RandomFlip


class MOSE(object):
    def __init__(self, model:nn.Module, buffer, optimizer, input_size, args):
        self.model = model
        self.optimizer = optimizer

        self.ins_t = args.ins_t
        self.epoch = args.epoch
        self.expert = int(args.expert)
        self.n_classes_num = args.n_classes
        self.use_ncm = (args.classifier == 'ncm')

        self.buffer = buffer
        self.buffer_per_class = 7
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_cur_task = (self.buffer_batch_size // 2) - args.batch_size

        if args.dataset == "cifar10":
            self.total_samples = 10000
            self.print_num = self.total_samples // 5
        elif "cifar100" in args.dataset:
            self.total_samples = 5000
            self.print_num = self.total_samples // 10
        elif args.dataset == "tiny_imagenet":
            self.total_samples = 1000
            self.print_num = self.total_samples // 10
        elif args.dataset == "mnist":
            self.total_samples = 6000
            self.print_num = self.total_samples // 5

        self.transform = get_transform(args.augmentation, input_size)

        self.total_step = 0
        self.class_holder = []
        self.scaler = GradScaler()
        # 保存旧模型用于知识蒸馏
        self.old_model = None
        # 蒸馏损失的温度和权重
        self.distill_temperature = getattr(args, 'distill_temperature', 2.0)
        self.distill_weight = getattr(args, 'distill_weight', 1.0)
        
        
        # 特征对齐损失权重
        self.align_weight = getattr(args, 'align_weight', 1.0)
        
        # 教师-学生特征蒸馏权重
        self.teacher_student_l2_weight = getattr(args, 'teacher_student_l2_weight', 1.0)
        
        # 原视图与增强视图一致性约束权重
        self.lambda_uncert = getattr(args, 'lambda_uncert', 0.0)
        
        # 不确定性缓冲区（U-Buffer）配置
        self.u_buffer_capacity = int(getattr(args, 'u_buffer_capacity', 64))
        self.u_buffer_batch_size = int(getattr(args, 'u_buffer_batch_size', min(self.buffer_batch_size, self.u_buffer_capacity)))
        buffer_device = self.buffer.bx.device if hasattr(self.buffer, 'bx') else torch.device('cuda')
        self.u_buffer = UncertaintyBuffer(capacity=self.u_buffer_capacity, device=buffer_device)
        
        # 主缓冲区不确定性跟踪
        self.main_buffer_uncertainties = {}  # 存储主缓冲区样本的不确定性 {index: uncertainty_score}

    def train_any_task(self, task_id, train_loader, epoch):
        num_d = 0
        epoch_log_holder = []
        if epoch == 0:
            self.new_class_holder = []
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach().item())
                    self.new_class_holder.append(Y[j].detach().item())

            # 保存原始数据
            x_orig = x.cuda(non_blocking=True)
            y_orig = y.cuda(non_blocking=True)

            # 用于保存缓冲区采样
            shared_buffer_x = None
            shared_buffer_y = None
            buffer_indices = None  # U-Buffer样本索引
            main_buffer_indices = None  # 主缓冲区样本索引
            batch_uncertainty = None  # 当前批次样本不确定性
            cat_view_from_u = False
            cat_u_base_size = 0
            buffer_batch_size = 0
            u_buffer_size = 0
            new_batch_size = 0
            cat_base_size = 0
            cat_input_size = 0
            new_input_size = 0
            new_view_scores = None
            cat_view_scores = None

            # 对同一批次进行两次更新
            # 第一次更新：使用当前批次的新任务样本 + 主缓冲区样本
            # 第二次更新：使用当前批次样本 + U-Buffer高不确定性样本
            for update_round in range(2):
                loss = 0.
                cat_base_size = 0
                cat_input_size = 0
                cat_view_from_u = False
                cat_u_base_size = 0
                loss_log = {
                    'step':     self.total_step,
                    'train/loss':     0.,
                    'train/ins':      0.,
                    'train/ce':       0.,
                    'train/distill':  0.,
                    'train/align':    0.,  # feature alignment loss
                    'train/ts_feat':  0.,  # teacher-student feature distillation loss
                    'train/view_cons':0.,  # canonical-vs-aug views consistency
                }

                if len(self.buffer) > 0:

                    with autocast():
                        x, y = x_orig, y_orig

                        # sample enough new class samples
                        new_x = x.detach()
                        new_y = y.detach()
                        if batch_idx != 0:
                            buffer_cur_task = self.buffer_batch_size if task_id==0 else self.buffer_cur_task
                            _ret = self.buffer.onlysample(buffer_cur_task, task=task_id)
                            cur_x, cur_y = _ret[0], _ret[1]
                            if len(cur_x.shape) > 3:
                                new_x = torch.cat((x.detach(), cur_x))
                                new_y = torch.cat((y.detach(), cur_y))
                        new_base_size = new_x.size(0)

                        if task_id > 0:
                            cat_base_size = 0
                            # 第一次更新：使用主缓冲区样本
                            if update_round == 0:
                                new_over_all = len(self.new_class_holder) / len(self.class_holder)
                                new_batch_size = min(
                                    int(self.buffer_batch_size * new_over_all), x.size(0)
                                )
                                buffer_batch_size = self.buffer_batch_size - new_batch_size
                                
                                # 从主缓冲区采样（排除当前任务）并记录索引
                                mem_x, mem_y, bt, main_indices = self.buffer.sample(
                                    buffer_batch_size, exclude_task=task_id, ret_ind=True)
                                main_buffer_indices = main_indices
                                
                                shared_buffer_x = mem_x
                                shared_buffer_y = mem_y
                                buffer_indices = None  # 第一轮不涉及U-Buffer
                                
                                # 使用当前批次的样本
                                cat_x = torch.cat((x[:new_batch_size].detach(), shared_buffer_x))
                                cat_y = torch.cat((y[:new_batch_size].detach(), shared_buffer_y))
                                cat_base_size = cat_x.size(0)
                                cat_u_base_size = 0
                                cat_view_from_u = False
                            
                            # 第二次更新：根据两类缓冲区的不确定性动态选择采样来源
                            elif update_round == 1:
                                new_over_all = len(self.new_class_holder) / len(self.class_holder)
                                new_batch_size = min(
                                    int(self.buffer_batch_size * new_over_all), x.size(0)
                                )

                                buffer_batch_size = self.buffer_batch_size - new_batch_size

                                if hasattr(self.buffer, 'uncertainty_stats'):
                                    main_stats = self.buffer.uncertainty_stats()
                                else:
                                    main_stats = {'mean': 0.0}
                                if len(self.u_buffer) > 0:
                                    u_stats = self.u_buffer.stats()
                                else:
                                    u_stats = {'mean': -float('inf')}

                                use_u_buffer = (len(self.u_buffer) > 0) and (u_stats['mean'] > main_stats['mean'])

                                if use_u_buffer:
                                    u_buffer_size = buffer_batch_size
                                    u_x, u_y, u_idx = self._sample_u_buffer_with_quota(u_buffer_size)

                                    shared_buffer_x = u_x
                                    shared_buffer_y = u_y
                                    buffer_indices = u_idx
                                    main_buffer_indices = torch.empty(0, dtype=torch.long, device=self.buffer.bx.device)

                                    if u_x.numel() > 0:
                                        cat_x = torch.cat((x[:new_batch_size].detach(), u_x))
                                        cat_y = torch.cat((y[:new_batch_size].detach(), u_y))
                                        cat_u_base_size = u_x.size(0)
                                        cat_view_from_u = True
                                    else:
                                        cat_x = x[:new_batch_size].detach()
                                        cat_y = y[:new_batch_size].detach()
                                        cat_u_base_size = 0
                                        cat_view_from_u = False
                                else:
                                    # 使用主缓冲区样本
                                    mem_x, mem_y, bt, main_indices = self.buffer.sample(
                                        buffer_batch_size, exclude_task=task_id, ret_ind=True)
                                    main_buffer_indices = main_indices
                                    shared_buffer_x = mem_x
                                    shared_buffer_y = mem_y
                                    buffer_indices = None

                                    if mem_x.numel() > 0:
                                        cat_x = torch.cat((x[:new_batch_size].detach(), mem_x))
                                        cat_y = torch.cat((y[:new_batch_size].detach(), mem_y))
                                    else:
                                        cat_x = x[:new_batch_size].detach()
                                        cat_y = y[:new_batch_size].detach()
                                    cat_u_base_size = 0
                                    cat_view_from_u = False
                                cat_base_size = cat_x.size(0)
                            else:
                                continue

                            # 样本增强：两次更新都进行相同的增强
                            new_x = RandomFlip(new_x, 2)
                            new_y = new_y.repeat(2)
                            cat_x = RandomFlip(cat_x, 2)
                            cat_y = cat_y.repeat(2)

                            new_x = torch.cat((new_x, self.transform(new_x)))
                            new_y = torch.cat((new_y, new_y))
                            cat_x = torch.cat((cat_x, self.transform(cat_x)))
                            cat_y = torch.cat((cat_y, cat_y))

                            new_input_size = new_x.size(0)
                            cat_input_size = cat_x.size(0)

                            all_x = torch.cat((new_x, cat_x))
                            all_y = torch.cat((new_y, cat_y))
                            all_x = all_x.detach()
                            all_y = all_y.detach()

                            feat_list = self.model.features(all_x)
                            proj_list = self.model.head(feat_list, use_proj=True)
                            pred_list = self.model.head(feat_list, use_proj=False)
                            
                            # 专家特征处理（用于对齐损失）
                            stu_feat = self._expert_aligned_features(feat_list)
                            consistency_feat = stu_feat

                            with torch.no_grad():
                                if new_input_size > 0 and new_base_size > 0:
                                    new_view_scores = self._feature_inconsistency_scores(
                                        consistency_feat[:new_input_size].detach(),
                                        new_base_size,
                                    )
                                if cat_input_size > 0 and cat_base_size > 0:
                                    cat_view_scores = self._feature_inconsistency_scores(
                                        consistency_feat[new_input_size:].detach(),
                                        cat_base_size,
                                    )

                            ts_feat_loss = None
                            if self.teacher_student_l2_weight > 0 and len(feat_list) >= 2:
                                teacher_feat = feat_list[-1].detach()
                                student_feat = feat_list[0]
                                if len(self.new_class_holder) > 0:
                                    new_class_tensor = torch.as_tensor(self.new_class_holder, device=all_y.device, dtype=all_y.dtype)
                                    is_old = ~torch.isin(all_y, new_class_tensor)
                                else:
                                    is_old = torch.ones_like(all_y, dtype=torch.bool, device=all_y.device)
                                if is_old.any():
                                    student_old = student_feat[is_old]
                                    teacher_old = teacher_feat[is_old]
                                    if student_old.numel() > 0:
                                        ts_feat_loss = F.mse_loss(student_old, teacher_old) * self.teacher_student_l2_weight

                            if (
                                self.lambda_uncert > 0
                                and cat_view_from_u
                                and cat_u_base_size > 0
                                and cat_base_size > 0
                                and cat_input_size > 0
                            ):
                                select_start = max(0, cat_base_size - cat_u_base_size)
                                view_loss = self._view_consistency_loss(
                                    consistency_feat[new_input_size:],
                                    cat_base_size,
                                    select_start=select_start,
                                    select_count=cat_u_base_size,
                                )
                                weighted_view_loss = self.lambda_uncert * view_loss
                                loss += weighted_view_loss
                                loss_log['train/view_cons'] += weighted_view_loss.item()

                            for i in range(len(feat_list)):
                                feat = feat_list[i]
                                proj = proj_list[i]
                                pred = pred_list[i]

                                new_pred = pred[:new_input_size]
                                cat_pred = pred[new_input_size:]

                                # instance-wise contarstive loss
                                ins_loss = sup_con_loss(proj, self.ins_t, all_y)

                                # balanced cross entropy loss
                                ce_loss  = 2 * F.cross_entropy(cat_pred, cat_y)

                                new_pred = new_pred[:, self.new_class_holder]
                                mapping = torch.full((self.n_classes_num,), -1, dtype=torch.long, device=new_y.device)
                                for idx, cls in enumerate(self.new_class_holder):
                                    mapping[cls] = idx
                                new_y_mapped = mapping[new_y]
                                valid_mask = new_y_mapped.ge(0)
                                if valid_mask.all():
                                    ce_loss += F.cross_entropy(new_pred, new_y_mapped)
                                elif valid_mask.any():
                                    ce_loss += F.cross_entropy(new_pred[valid_mask], new_y_mapped[valid_mask])

                                # 知识蒸馏损失：对齐旧模型的logit，屏蔽新任务类别
                                distill_loss = 0.
                                if self.old_model is not None:
                                    with torch.no_grad():
                                        old_feat_list = self.old_model.features(all_x)
                                        old_pred_list = self.old_model.head(old_feat_list, use_proj=False)
                                        old_pred = old_pred_list[i]
                                        old_feat = old_feat_list[i]
                                    
                                    # 创建mask，屏蔽新任务类别
                                    mask = torch.ones(pred.size(1), dtype=torch.bool, device=pred.device)
                                    for new_cls in self.new_class_holder:
                                        mask[new_cls] = False
                                    
                                    # 只对旧类别计算KL散度蒸馏损失
                                    if mask.sum() > 0:
                                        pred_old_classes = pred[:, mask]
                                        old_pred_old_classes = old_pred[:, mask]
                                        
                                        distill_loss = F.kl_div(
                                            F.log_softmax(pred_old_classes / self.distill_temperature, dim=1),
                                            F.softmax(old_pred_old_classes / self.distill_temperature, dim=1),
                                            reduction='batchmean'
                                        ) * (self.distill_temperature ** 2) * self.distill_weight
                                    
                                
                                # 特征对齐损失（来自mose - 副本.py）
                                align_loss = 0.
                                if i != self.expert:
                                    align_loss = torch.dist(
                                        F.normalize(stu_feat, dim=1), 
                                        F.normalize(feat.detach(), dim=1), p=2
                                    )
                                align_loss = align_loss * self.align_weight

                                loss += ins_loss + ce_loss + distill_loss + align_loss
                                loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                                loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.
                                loss_log['train/distill'] += distill_loss.item() if distill_loss != 0. else 0.
                                loss_log['train/align'] += align_loss.item() if align_loss != 0. else 0.
                            
                            if ts_feat_loss is not None and ts_feat_loss.item() != 0.:
                                loss += ts_feat_loss
                                loss_log['train/ts_feat'] += ts_feat_loss.item()
                        else:
                            # 样本增强：两次更新都进行相同的增强
                            new_x = RandomFlip(new_x, 2)
                            new_y = new_y.repeat(2)

                            new_x = torch.cat((new_x, self.transform(new_x)))
                            new_y = torch.cat((new_y, new_y))
                            
                            new_x = new_x.detach()
                            new_y = new_y.detach()

                            feat_list = self.model.features(new_x)
                            proj_list = self.model.head(feat_list, use_proj=True)
                            pred_list = self.model.head(feat_list, use_proj=False)
                            consistency_feat = self._expert_aligned_features(feat_list)
                            new_input_size = new_x.size(0)
                            with torch.no_grad():
                                if new_input_size > 0 and new_base_size > 0:
                                    new_view_scores = self._feature_inconsistency_scores(
                                        consistency_feat.detach(),
                                        new_base_size,
                                    )

                            ts_feat_loss = None
                            if self.teacher_student_l2_weight > 0 and len(feat_list) >= 2:
                                teacher_feat = feat_list[-1].detach()
                                student_feat = feat_list[0]
                                if len(self.new_class_holder) > 0:
                                    new_class_tensor = torch.as_tensor(self.new_class_holder, device=new_y.device, dtype=new_y.dtype)
                                    is_old = ~torch.isin(new_y, new_class_tensor)
                                else:
                                    is_old = torch.ones_like(new_y, dtype=torch.bool, device=new_y.device)
                                if is_old.any():
                                    student_old = student_feat[is_old]
                                    teacher_old = teacher_feat[is_old]
                                    if student_old.numel() > 0:
                                        ts_feat_loss = F.mse_loss(student_old, teacher_old) * self.teacher_student_l2_weight

                            for i in range(len(feat_list)):
                                feat = feat_list[i]
                                proj = proj_list[i]
                                pred = pred_list[i]

                                # instance-wise contarstive loss
                                ins_loss = sup_con_loss(proj, self.ins_t, new_y)

                                # balanced cross entropy loss
                                ce_loss  = F.cross_entropy(pred, new_y)

                                loss += ins_loss + ce_loss
                                loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                                loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.
                            if ts_feat_loss is not None and ts_feat_loss.item() != 0.:
                                loss += ts_feat_loss
                                loss_log['train/ts_feat'] += ts_feat_loss.item()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # 在每次更新完成后：计算和更新不确定性（复用当前视图特征）
                with torch.no_grad():
                    if update_round == 0:
                        B = x_orig.size(0)
                        if new_view_scores is not None and new_view_scores.numel() >= B:
                            batch_uncertainty = new_view_scores[:B].clone()
                        else:
                            batch_uncertainty = torch.zeros(B, device=x_orig.device)

                        if len(self.u_buffer) > 0:
                            u_buffer_stats = self.u_buffer.stats()
                            threshold = u_buffer_stats['mean'] + u_buffer_stats['std']
                            high_mask = batch_uncertainty > threshold
                            if high_mask.any():
                                self.u_buffer.add_batch(
                                    x_orig[high_mask],
                                    y_orig[high_mask],
                                    batch_uncertainty[high_mask],
                                )
                        else:
                            self.u_buffer.add_batch(x_orig, y_orig, batch_uncertainty)

                        if (
                            main_buffer_indices is not None
                            and main_buffer_indices.numel() > 0
                            and cat_view_scores is not None
                            and cat_view_scores.numel() >= main_buffer_indices.numel()
                        ):
                            main_sample_count = main_buffer_indices.numel()
                            if main_sample_count > 0:
                                main_scores = cat_view_scores[-main_sample_count:].clone()
                                self.buffer.update_uncertainty(main_buffer_indices, main_scores)
                                for idx, uncertainty in zip(main_buffer_indices.tolist(), main_scores.tolist()):
                                    self.main_buffer_uncertainties[int(idx)] = float(uncertainty)

                    elif update_round == 1:
                        if (
                            buffer_indices is not None
                            and buffer_indices.numel() > 0
                            and cat_view_scores is not None
                            and cat_view_scores.numel() >= buffer_indices.numel()
                        ):
                            u_sample_count = buffer_indices.numel()
                            if u_sample_count > 0:
                                u_scores = cat_view_scores[-u_sample_count:].clone()
                                self.u_buffer.update_uncertainty(buffer_indices, u_scores)
                        elif (
                            main_buffer_indices is not None
                            and main_buffer_indices.numel() > 0
                            and cat_view_scores is not None
                            and cat_view_scores.numel() >= main_buffer_indices.numel()
                        ):
                            main_sample_count = main_buffer_indices.numel()
                            if main_sample_count > 0:
                                main_scores = cat_view_scores[-main_sample_count:].clone()
                                self.buffer.update_uncertainty(main_buffer_indices, main_scores)
                                for idx, uncertainty in zip(main_buffer_indices.tolist(), main_scores.tolist()):
                                    self.main_buffer_uncertainties[int(idx)] = float(uncertainty)

                if hasattr(self.buffer, 'uncertainty_stats'):
                    buffer_stats = self.buffer.uncertainty_stats()
                else:
                    buffer_stats = {'mean': 0.0, 'max': 0.0, 'min': 0.0}
                loss_log['buffer/u_mean'] = buffer_stats['mean']
                loss_log['buffer/u_max'] = buffer_stats['max']
                loss_log['buffer/u_min'] = buffer_stats['min']
                if hasattr(self.u_buffer, 'stats'):
                    u_buffer_stats = self.u_buffer.stats()
                else:
                    if hasattr(self.u_buffer, 'u_list') and len(self.u_buffer.u_list) > 0:
                        u_tensor = torch.tensor(self.u_buffer.u_list, device=self.u_buffer.device)
                        u_buffer_stats = {
                            'mean': float(u_tensor.mean().item()),
                            'max': float(u_tensor.max().item()),
                            'min': float(u_tensor.min().item()),
                        }
                    else:
                        u_buffer_stats = {'mean': 0.0, 'max': 0.0, 'min': 0.0}
                loss_log['u_buffer/u_mean'] = u_buffer_stats['mean']
                loss_log['u_buffer/u_max'] = u_buffer_stats['max']
                loss_log['u_buffer/u_min'] = u_buffer_stats['min']

                loss_log['train/loss'] = loss.item() if loss != 0. else 0.
                epoch_log_holder.append(loss_log)
                self.total_step += 1

                if num_d % self.print_num == 0 or batch_idx == 1:
                    buf_mean = loss_log.get('buffer/u_mean', 0.0)
                    u_mean = loss_log.get('u_buffer/u_mean', 0.0)
                    if task_id > 0:
                        print(
                            f"==>>> it: {batch_idx}, round: {update_round+1}, loss: ins {loss_log['train/ins']:.2f} "
                            f"+ ce {loss_log['train/ce']:.3f} + distill {loss_log['train/distill']:.3f} "
                            f"+ align {loss_log['train/align']:.3f} + ts_feat {loss_log['train/ts_feat']:.3f} "
                            f"+ view_cons {loss_log['train/view_cons']:.3f} "
                            f"= {loss_log['train/loss']:.6f}, main_u {buf_mean:.3f}, u_u {u_mean:.3f}, "
                            f"{100 * (num_d / self.total_samples)}%"
                        )
                    else:
                        print(
                            f"==>>> it: {batch_idx}, round: {update_round+1}, loss: ins {loss_log['train/ins']:.2f} "
                            f"+ ce {loss_log['train/ce']:.3f} + align {loss_log['train/align']:.3f} "
                            f"+ ts_feat {loss_log['train/ts_feat']:.3f} + view_cons {loss_log['train/view_cons']:.3f} "
                            f"= {loss_log['train/loss']:.6f}, "
                            f"main_u {buf_mean:.3f}, u_u {u_mean:.3f}, {100 * (num_d / self.total_samples)}%"
                        )
            
            # 只在第一轮epoch时将样本加入缓冲区
            if epoch == 0:
                if batch_uncertainty is None:
                    batch_uncertainty = torch.zeros(x_orig.size(0), device=x_orig.device)
                self.buffer.add_reservoir(
                    x=x_orig.detach(),
                    y=y_orig.detach(),
                    logits=None,
                    t=task_id,
                    uncertainty=batch_uncertainty.detach(),
                    u_buffer=self.u_buffer,
                )

        return epoch_log_holder

    def train(self, task_id, train_loader):
        self.model.train()
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log_holder = self.train_any_task(task_id, train_loader, epoch)
            train_log_holder.extend(epoch_log_holder)
            # self.buffer.print_per_task_num()
        
        # 任务完成后，保存当前模型作为旧模型用于下一个任务的知识蒸馏
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        return train_log_holder

    def test(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        if self.use_ncm:
            # calculate the class means for each feature layer
            print("\nCalculate class means for each layer...\n")
            self.class_means_ls = [{} for _ in range(4)]
            class_inputs = {cls: [] for cls in self.class_holder}
            for x, y in zip(self.buffer.x, self.buffer.y_int):
                class_inputs[y.item()].append(x)

            for cls, inputs in class_inputs.items():
                features = [[] for _ in range(4)]
                for ex in inputs:
                    return_features_ls = self.model.features(ex.unsqueeze(0))
                    for feat_id in range(4):
                        feature = return_features_ls[feat_id].detach().clone()
                        feature = F.normalize(feature, dim=1)
                        features[feat_id].append(feature.squeeze())

                for feat_id in range(4):
                    if len(features[feat_id]) == 0:
                        mu_y = torch.normal(
                            0, 1, size=tuple(self.model.features(x.unsqueeze(0))[feat_id].detach().size())
                        )
                        mu_y = mu_y.to(x.device)
                    else:
                        features[feat_id] = torch.stack(features[feat_id])
                        mu_y = features[feat_id].mean(0)

                    mu_y = F.normalize(mu_y.reshape(1, -1), dim=1)
                    self.class_means_ls[feat_id][cls] = mu_y.squeeze()

        all_acc_list = {'step': self.total_step}
        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_model(task_loader[j]['test'], j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_mean(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # # clear the calculated class_means
        # self.class_means_ls = None

        return acc_list, all_acc_list

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

    def _expert_aligned_features(self, feat_list):
        expert_feat = feat_list[self.expert]
        if hasattr(self.model, 'final_addaption_layer'):
            expert_feat = self.model.final_addaption_layer(expert_feat)
        return expert_feat

    def _view_consistency_loss(
        self,
        features: torch.Tensor,
        base_batch: int,
        *,
        select_start: int = 0,
        select_count: Optional[int] = None,
    ) -> torch.Tensor:
        if base_batch <= 0 or features.numel() == 0:
            return features.new_zeros(())
        total = features.size(0)
        if total % base_batch != 0:
            return features.new_zeros(())
        view_count = total // base_batch
        if view_count <= 1:
            return features.new_zeros(())
        norm_feat = F.normalize(features, dim=1)
        reshaped = norm_feat.reshape(view_count, base_batch, -1).permute(1, 0, 2)  # B x K x D
        start = max(0, min(base_batch, select_start))
        if select_count is None:
            subset = reshaped[start:]
        else:
            end = max(start, min(base_batch, start + select_count))
            subset = reshaped[start:end]
        if subset.numel() == 0:
            return features.new_zeros(())
        anchor = subset[:, :1, :]
        others = subset[:, 1:, :]
        if others.numel() == 0:
            return features.new_zeros(())
        distances = (anchor - others).pow(2).sum(dim=-1)
        return distances.mean()

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
        top_k = order[: min(request_size, order.numel())]
        if top_k.numel() == 0:
            empty_x = torch.empty(0, device=self.u_buffer.device)
            empty_y = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            empty_idx = torch.empty(0, dtype=torch.long, device=self.u_buffer.device)
            return empty_x, empty_y, empty_idx

        selected_x = all_x[top_k]
        selected_y = all_y[top_k]
        selected_idx = all_idx[top_k]
        return selected_x, selected_y, selected_idx

    def _feature_inconsistency_scores(self, features: torch.Tensor, base_batch: int) -> torch.Tensor:
        if base_batch <= 0:
            ref_device = features.device if features is not None else next(self.model.parameters()).device
            return torch.zeros(0, device=ref_device)
        if features is None or features.numel() == 0:
            ref_device = features.device if features is not None else next(self.model.parameters()).device
            return torch.zeros(base_batch, device=ref_device)

        total = features.size(0)
        device = features.device
        if total % base_batch != 0:
            return torch.zeros(base_batch, device=device)

        view_count = total // base_batch
        if view_count <= 1:
            return torch.zeros(base_batch, device=device)

        reshaped = features.reshape(view_count, base_batch, -1).permute(1, 0, 2)  # B x K x D
        norm_feat = F.normalize(reshaped, dim=-1)
        anchor = norm_feat[:, :1, :]
        others = norm_feat[:, 1:, :]
        if others.numel() == 0:
            return torch.zeros(base_batch, device=device)

        cos_sim = (others * anchor).sum(dim=-1)
        distances = 1.0 - cos_sim
        return distances.mean(dim=1)
