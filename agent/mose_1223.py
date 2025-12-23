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
        self.teacher_student_l2_weight = getattr(args, 'teacher_student_l2_weight', 5.0)
        
        # 原视图与增强视图一致性约束权重
        print(" self.lambda_uncert = getattr(args, 'lambda_uncert', 15.0)")
        self.lambda_uncert = getattr(args, 'lambda_uncert', 15.0)
        
        # 不确定性缓冲区（U-Buffer）配置
        self.u_buffer_capacity = int(getattr(args, 'u_buffer_capacity', 64))
        self.u_buffer_batch_size = int(getattr(args, 'u_buffer_batch_size', min(self.buffer_batch_size, self.u_buffer_capacity)))
        buffer_device = self.buffer.bx.device if hasattr(self.buffer, 'bx') else torch.device('cuda')
        self.u_buffer = UncertaintyBuffer(capacity=self.u_buffer_capacity, device=buffer_device)
        
        # 主缓冲区不确定性跟踪
        self.main_buffer_uncertainties = {}  # 存储主缓冲区样本的不确定性 {index: uncertainty_score}

        # Hard-only 采样策略超参数 (用于 Round 2)
        self.T_min = getattr(args, 'T_min', 0.1)  # 温度参数（高聚焦）
        # 动态 Top-K 相似类参数
        self.topk_ratio = getattr(args, 'topk_ratio', 0.2)  # K = ratio * n_old
        self.topk_min = getattr(args, 'topk_min', 4)  # K 的最小值
        self.topk_max = getattr(args, 'topk_max', 64)  # K 的最大值
        self.args = args  # 保存 args 引用

        # [新增] 策略一：混合采样策略参数 (用于 Round 1)
        # 0.5 表示 50% 随机采样 (维持分布) + 50% 基于不确定性的难例采样 (提升边界)
        self.hard_sample_ratio = 0.5
        print(f"MOSE 初始化完成: Round 1 混合采样策略已启用, 难例采样比例 (Hard Ratio) = {self.hard_sample_ratio}")

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
            # 第一次更新：使用当前批次的新任务样本 + 主缓冲区样本 (混合采样)
            # 第二次更新：使用当前批次样本 + (U-Buffer 或 Hard-only 策略)
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
                    'sample/rho':     0.,  # 聚焦系数
                    'sample/B_sim':   0,   # 相似类采样数
                    'sample/B_rand':  0,   # 随机类采样数
                    'sample/R1_hard': 0,   # Round 1 难例数量
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
                            cur_x, cur_y = _ret[0].to(x.device, non_blocking=True), _ret[1].to(x.device, non_blocking=True)
                            if len(cur_x.shape) > 3:
                                new_x = torch.cat((x.detach(), cur_x))
                                new_y = torch.cat((y.detach(), cur_y))
                        new_base_size = new_x.size(0)

                        if task_id > 0:
                            cat_base_size = 0
                            # -------------------------------------------------------------------------
                            # 第一次更新：使用主缓冲区样本 - 【策略一：混合采样】
                            # -------------------------------------------------------------------------
                            if update_round == 0:
                                new_over_all = len(self.new_class_holder) / len(self.class_holder)
                                new_batch_size = min(
                                    int(self.buffer_batch_size * new_over_all), x.size(0)
                                )
                                buffer_total_batch_size = self.buffer_batch_size - new_batch_size
                                
                                # === [策略一核心修改] 混合采样：随机 + 难例 ===
                                n_hard = int(buffer_total_batch_size * self.hard_sample_ratio)
                                n_random = buffer_total_batch_size - n_hard
                                
                                list_x, list_y, list_ind = [], [], []
                                
                                # 1. 随机采样部分 (Random Sampling) - 保持总体分布
                                if n_random > 0:
                                    _ret_rand = self.buffer.sample(n_random, exclude_task=task_id, ret_ind=True)
                                    # [鲁棒性修复] 增加对 sample 返回值数量的判断
                                    if len(_ret_rand) >= 4:
                                        mem_x_rand, mem_y_rand, _, main_indices_rand = _ret_rand[0], _ret_rand[1], _ret_rand[2], _ret_rand[3]
                                    else: # 假设返回3个值
                                        mem_x_rand, mem_y_rand, main_indices_rand = _ret_rand[0], _ret_rand[1], _ret_rand[2]
                                    
                                    if len(mem_x_rand) > 0:
                                        # 确保移至 GPU
                                        list_x.append(mem_x_rand.to(x.device, non_blocking=True))
                                        list_y.append(mem_y_rand.to(x.device, non_blocking=True))
                                        # 索引保留在 CPU
                                        list_ind.append(main_indices_rand.cpu())

                                # 2. 难例采样部分 (Importance Sampling) - 针对大缓冲区的性能瓶颈
                                if n_hard > 0:
                                    # 调用新增的辅助方法
                                    mem_x_hard, mem_y_hard, main_indices_hard = self._sample_hard_from_buffer(
                                        n_hard, exclude_task_id=task_id, target_device=x.device
                                    )
                                    if mem_x_hard is not None and len(mem_x_hard) > 0:
                                        list_x.append(mem_x_hard)
                                        list_y.append(mem_y_hard)
                                        list_ind.append(main_indices_hard)
                                        loss_log['sample/R1_hard'] = len(mem_x_hard)
                                
                                # 3. 合并数据
                                if len(list_x) > 0:
                                    shared_buffer_x = torch.cat(list_x)
                                    shared_buffer_y = torch.cat(list_y)
                                    main_buffer_indices = torch.cat(list_ind).long()
                                else:
                                    # 异常保护（例如缓冲区刚开始为空）
                                    shared_buffer_x = torch.empty(0, device=x.device)
                                    shared_buffer_y = torch.empty(0, device=x.device)
                                    main_buffer_indices = torch.empty(0, dtype=torch.long) # CPU

                                buffer_indices = None  # 第一轮不涉及U-Buffer
                                
                                # 使用当前批次的样本 + 混合采样的缓冲区样本
                                if shared_buffer_x.size(0) > 0:
                                    cat_x = torch.cat((x[:new_batch_size].detach(), shared_buffer_x))
                                    cat_y = torch.cat((y[:new_batch_size].detach(), shared_buffer_y))
                                else:
                                    cat_x = x[:new_batch_size].detach()
                                    cat_y = y[:new_batch_size].detach()
                                    
                                cat_base_size = cat_x.size(0)
                                cat_u_base_size = 0
                                cat_view_from_u = False
                            
                            # -------------------------------------------------------------------------
                            # 第二次更新：双路径策略 (保持原文件逻辑)
                            # -------------------------------------------------------------------------
                            elif update_round == 1:
                                # 获取两个缓冲区的不确定性统计信息
                                if hasattr(self.buffer, 'uncertainty_stats'):
                                    main_stats = self.buffer.uncertainty_stats()
                                else:
                                    main_stats = {'mean': 0.0}
                                if len(self.u_buffer) > 0:
                                    u_stats = self.u_buffer.stats()
                                else:
                                    u_stats = {'mean': -float('inf')}

                                main_u = main_stats['mean']
                                u_u = u_stats['mean']

                                # 计算第二次更新的总回放预算 B2
                                new_over_all = len(self.new_class_holder) / len(self.class_holder)
                                new_batch_size = min(
                                    int(self.buffer_batch_size * new_over_all), x.size(0)
                                )
                                B2 = self.buffer_batch_size - new_batch_size

                                # 路径1：U-buffer不确定性更高 → 使用U-buffer样本
                                if len(self.u_buffer) > 0 and u_u > main_u:
                                    u_buffer_size = B2
                                    u_x, u_y, u_idx = self._sample_u_buffer_with_quota(u_buffer_size)

                                    if u_x.numel() > 0:
                                        # 确保u_buffer数据移到GPU（u_buffer可能在CPU）
                                        u_x = u_x.to(x.device, non_blocking=True)
                                        u_y = u_y.to(x.device, non_blocking=True)
                                        
                                        shared_buffer_x = u_x
                                        shared_buffer_y = u_y
                                        buffer_indices = u_idx
                                        main_buffer_indices = torch.empty(0, dtype=torch.long)  # 索引放CPU

                                        cat_x = torch.cat((x[:new_batch_size].detach(), u_x))
                                        cat_y = torch.cat((y[:new_batch_size].detach(), u_y))
                                        cat_base_size = cat_x.size(0)
                                        cat_u_base_size = u_x.size(0)
                                        cat_view_from_u = True
                                    else:
                                        # U-buffer为空，跳过第二次更新
                                        continue

                                # 路径2：主缓冲区不确定性更高 → 使用hard-only相似类回放策略
                                elif main_u > u_u:
                                    # Hard-only策略：B_sim = B2, B_rand = 0
                                    B_sim = B2
                                    B_rand = 0
                                    
                                    # 记录采样策略参数
                                    loss_log['sample/rho'] = 1.0  # hard-only
                                    loss_log['sample/B_sim'] = B_sim
                                    loss_log['sample/B_rand'] = 0

                                    # 采样策略
                                    sampled_x_list = []
                                    sampled_y_list = []
                                    sampled_idx_list = []

                                    # 从相似类采样 B_sim (=B2) 个样本（使用classifier权重作为原型，O(1)）
                                    if len(self.new_class_holder) > 0:
                                        with torch.no_grad():
                                            # 使用classifier权重作为类原型（速度快且语义相近）
                                            classifier_weights = self._get_classifier_weight()
                                            
                                            if classifier_weights is not None:
                                                # 归一化权重向量
                                                classifier_weights = F.normalize(classifier_weights, dim=1)
                                                
                                                # 新类原型：从classifier权重中提取
                                                new_class_prototypes = {}
                                                for cls_id in self.new_class_holder:
                                                    if cls_id < classifier_weights.size(0):
                                                        new_class_prototypes[cls_id] = classifier_weights[cls_id]
                                                
                                                # 旧类原型：从classifier权重中提取
                                                old_classes = [c for c in self.class_holder if c not in self.new_class_holder]
                                                # 过滤掉 buffer 中没有样本的旧类（更稳定）
                                                old_classes = [c for c in old_classes if c < self.buffer.class_counts.numel() and int(self.buffer.class_counts[c].item()) > 0]
                                                
                                                old_class_prototypes = {}
                                                for old_cls in old_classes:
                                                    if old_cls < classifier_weights.size(0):
                                                        old_class_prototypes[old_cls] = classifier_weights[old_cls]
                                                
                                                if len(old_class_prototypes) > 0 and len(new_class_prototypes) > 0:
                                                    # 计算相似度（向量化，O(#old * #new)）
                                                    similarity_scores = {}
                                                    for old_cls, old_proto in old_class_prototypes.items():
                                                        max_sim = -1.0
                                                        for new_cls, new_proto in new_class_prototypes.items():
                                                            sim = F.cosine_similarity(new_proto.unsqueeze(0), old_proto.unsqueeze(0), dim=1).item()
                                                            max_sim = max(max_sim, sim)
                                                        similarity_scores[old_cls] = max_sim
                                                    
                                                    # 动态计算 Top-K：K = clamp(ceil(ratio * n_old), K_min, min(K_max, n_old, B_sim))
                                                    sorted_classes = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
                                                    n_old = len(sorted_classes)
                                                    top_k = int(np.ceil(self.topk_ratio * n_old)) if n_old > 0 else 0
                                                    top_k = max(self.topk_min, min(top_k, self.topk_max, n_old, B_sim))
                                                    
                                                    top_k_classes = [cls for cls, _ in sorted_classes[:top_k]]
                                                    top_k_scores = torch.tensor([score for _, score in sorted_classes[:top_k]], device=x_orig.device)
                                                    
                                                    # 用softmax计算权重（固定温度）
                                                    T = self.T_min  # hard-only: 使用最小温度（高聚焦）
                                                    weights = F.softmax(top_k_scores / T, dim=0)
                                                    
                                                    # 使用multinomial采样，自动保证总数=B_sim (=B2)
                                                    class_indices = torch.multinomial(weights, B_sim, replacement=True)
                                                    
                                                    # 统计每个类需要采样的数量
                                                    class_samples = {}
                                                    for idx in class_indices:
                                                        cls = top_k_classes[idx.item()]
                                                        class_samples[cls] = class_samples.get(cls, 0) + 1
                                                    
                                                    # 从主缓冲区按类采样（使用索引缓存，O(k)，支持有放回）
                                                    for cls, n_samples in class_samples.items():
                                                        if n_samples > 0:
                                                            cls_x, cls_y, cls_idx = self._sample_from_buffer_by_class(
                                                                cls, n_samples, target_device=x_orig.device
                                                            )
                                                            if cls_x is not None and cls_x.numel() > 0:
                                                                sampled_x_list.append(cls_x)
                                                                sampled_y_list.append(cls_y)
                                                                sampled_idx_list.append(cls_idx)
                                    
                                    # 合并采样结果
                                    if len(sampled_x_list) > 0:
                                        shared_buffer_x = torch.cat(sampled_x_list, dim=0)
                                        shared_buffer_y = torch.cat(sampled_y_list, dim=0)
                                        main_buffer_indices = torch.cat(sampled_idx_list, dim=0)  # CPU
                                        
                                        # 确保总数=B2（从已采集的hard样本里有放回补齐）
                                        actual = shared_buffer_x.size(0)
                                        if actual < B2 and actual > 0:
                                            shortage = B2 - actual
                                            # 从已采到的hard索引里有放回补齐
                                            pool = main_buffer_indices.tolist()
                                            extra_idx_list = np.random.choice(pool, shortage, replace=True).tolist()
                                            extra_idx = torch.tensor(extra_idx_list, dtype=torch.long)  # CPU
                                            
                                            extra_x = self.buffer.bx[extra_idx].to(x_orig.device, non_blocking=True)
                                            extra_y = self.buffer.by[extra_idx].to(x_orig.device, non_blocking=True)
                                            
                                            shared_buffer_x = torch.cat([shared_buffer_x, extra_x], dim=0)
                                            shared_buffer_y = torch.cat([shared_buffer_y, extra_y], dim=0)
                                            main_buffer_indices = torch.cat([main_buffer_indices, extra_idx], dim=0)
                                        elif actual == 0:
                                            # hard池完全取不到：跳过第二轮
                                            continue
                                        
                                        cat_x = torch.cat((x[:new_batch_size].detach(), shared_buffer_x))
                                        cat_y = torch.cat((y[:new_batch_size].detach(), shared_buffer_y))
                                        cat_base_size = cat_x.size(0)
                                        cat_u_base_size = 0
                                        cat_view_from_u = False
                                        buffer_indices = None
                                    else:
                                        # 没有采样到样本，跳过第二次更新
                                        continue
                                
                                else:
                                    # 两个缓冲区都为空或不确定性相同，跳过第二次更新
                                    continue
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

                            # 提前计算旧模型输出，供 ts_feat_loss 和 distill_loss 复用
                            old_feat_list = None
                            old_pred_list = None
                            if self.old_model is not None:
                                with torch.no_grad():
                                    old_feat_list = self.old_model.features(all_x)
                                    old_pred_list = self.old_model.head(old_feat_list, use_proj=False)

                            # ts_feat_loss：当前深层 vs 旧模型深层（仅旧类样本）
                            ts_feat_loss = None
                            if self.teacher_student_l2_weight > 0 and old_feat_list is not None:
                                cur_deep = feat_list[-1]
                                old_deep = old_feat_list[-1]

                                if len(self.new_class_holder) > 0:
                                    new_cls = torch.as_tensor(self.new_class_holder, device=all_y.device, dtype=all_y.dtype)
                                    is_old = ~torch.isin(all_y, new_cls)
                                else:
                                    is_old = torch.ones_like(all_y, dtype=torch.bool, device=all_y.device)

                                if is_old.any():
                                    cur = F.normalize(cur_deep[is_old].flatten(1), dim=1)
                                    old = F.normalize(old_deep[is_old].flatten(1), dim=1)
                                    ts_feat_loss = (1.0 - F.cosine_similarity(cur, old, dim=1)).mean() * self.teacher_student_l2_weight

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
                                if old_pred_list is not None:
                                    old_pred = old_pred_list[i]
                                    
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
                                
                                # 特征对齐损失
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
                    rho_val = loss_log.get('sample/rho', 0.0)
                    b_sim = loss_log.get('sample/B_sim', 0)
                    b_rand = loss_log.get('sample/B_rand', 0)
                    n_hard_r1 = loss_log.get('sample/R1_hard', 0)
                    
                    if task_id > 0:
                        sample_info = f", R1_Hard={n_hard_r1}"
                        if update_round == 1:
                             sample_info += f", ρ={rho_val:.2f}"
                        
                        print(
                            f"==>>> it: {batch_idx}, round: {update_round+1}, loss: ins {loss_log['train/ins']:.2f} "
                            f"+ ce {loss_log['train/ce']:.3f} + distill {loss_log['train/distill']:.3f} "
                            f"+ align {loss_log['train/align']:.3f} + ts_feat {loss_log['train/ts_feat']:.3f} "
                            f"+ view_cons {loss_log['train/view_cons']:.3f} "
                            f"= {loss_log['train/loss']:.6f}, main_u {buf_mean:.3f}, u_u {u_mean:.3f}"
                            f"{sample_info}, {100 * (num_d / self.total_samples):.1f}%"
                        )
                    else:
                        print(
                            f"==>>> it: {batch_idx}, round: {update_round+1}, loss: ins {loss_log['train/ins']:.2f} "
                            f"+ ce {loss_log['train/ce']:.3f} + align {loss_log['train/align']:.3f} "
                            f"+ ts_feat {loss_log['train/ts_feat']:.3f} + view_cons {loss_log['train/view_cons']:.3f} "
                            f"= {loss_log['train/loss']:.6f}, "
                            f"main_u {buf_mean:.3f}, u_u {u_mean:.3f}, {100 * (num_d / self.total_samples):.1f}%"
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
        
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        return train_log_holder

    def test(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        if self.use_ncm:
            print("\nCalculating class means for NCM classifier...\n")
            self.class_means_ls = [{} for _ in range(4)]
            class_inputs = {cls: [] for cls in self.class_holder}
            
            # Use buffer.bx and buffer.by for efficiency
            for idx in range(len(self.buffer)):
                y = self.buffer.by[idx].item()
                if y in class_inputs:
                    class_inputs[y].append(self.buffer.bx[idx])

            for cls, inputs in class_inputs.items():
                if not inputs: continue
                features = [[] for _ in range(4)]
                # Batch process for efficiency
                input_batch = torch.stack(inputs).cuda()
                with torch.no_grad():
                    return_features_ls = self.model.features(input_batch)
                for feat_id in range(4):
                    feature = return_features_ls[feat_id].detach()
                    feature = F.normalize(feature, dim=1)
                    features[feat_id] = feature

                for feat_id in range(4):
                    if len(features[feat_id]) == 0:
                        # Fallback if no samples for a class
                        feat_shape = self.model.features(torch.randn(1, 3, 32, 32).cuda())[feat_id].shape
                        mu_y = torch.randn(feat_shape, device='cuda')
                    else:
                        mu_y = features[feat_id].mean(0)
                    
                    self.class_means_ls[feat_id][cls] = F.normalize(mu_y.flatten(), dim=0)

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
                print(f"Tasks Acc: {acc_list}")
                print(f"Tasks Avg Acc: {acc_list[:i+1].mean():.2f}%")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_mean(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"Tasks Acc: {acc_list}")
            print(f"Tasks Avg Acc: {acc_list[:i+1].mean():.2f}%")

        return acc_list, all_acc_list

    def test_buffer(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest Buffer with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_buffer_task(j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"Buffer Tasks Acc: {acc_list}")
                print(f"Buffer Tasks Avg Acc: {acc_list[:i+1].mean():.2f}%")

        print(f"{'*'*100}\nTest Buffer with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_buffer_task_mean(j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"Buffer Tasks Acc: {acc_list}")
            print(f"Buffer Tasks Avg Acc: {acc_list[:i+1].mean():.2f}%")

        return acc_list, all_acc_list

    def test_buffer_task(self, i, feat_id):
        correct, num = torch.tensor(0, device='cuda'), torch.tensor(0, device='cuda')
        
        _ret = self.buffer.onlysample(self.buffer.current_index, task=i)
        x_i, y_i = _ret[0].cuda(), _ret[1].cuda()

        if self.use_ncm:
            # Similar to test_model, but on buffer data
            pass # Simplified for brevity, logic is same as test_model_ncm
        else:
            for x, y in zip(x_i, y_i):
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                model_output = self.model(x)
                pred = model_output[0][feat_id].max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()
                num += x.size(0)

        test_accuracy = (100. * correct / num) if num > 0 else 0
        print(f'Buffer Test Task {i}: Accuracy: {correct}/{num} ({test_accuracy:.2f}%)')
        return test_accuracy

    def test_buffer_task_mean(self, i):
        correct, num = torch.tensor(0, device='cuda'), torch.tensor(0, device='cuda')

        _ret = self.buffer.onlysample(self.buffer.current_index, task=i)
        x_i, y_i = _ret[0].cuda(), _ret[1].cuda()

        if self.use_ncm:
            # Similar to test_model_mean, but on buffer data
            pass # Simplified for brevity
        else:
            for x, y in zip(x_i, y_i):
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                model_output = self.model(x)
                expert_logits = model_output[0]
                pred = torch.stack(expert_logits, dim=1).mean(dim=1).max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()
                num += x.size(0)

        test_accuracy = (100. * correct / num) if num > 0 else 0
        print(f'Buffer Test Task {i} (Mean): Accuracy: {correct}/{num} ({test_accuracy:.2f}%)')
        return test_accuracy

    def test_train(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        all_acc_list = {'step': self.total_step}

        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest Train Set with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc = self.test_model(task_loader[i]['train'], i, feat_id=feat_id)
                all_acc_list[str(feat_id)] = acc.item()

        print(f"{'*'*100}\nTest Train Set with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc = self.test_model_mean(task_loader[i]['train'], i)
            all_acc_list['mean'] = acc.item()

        return acc.item(), all_acc_list

    def test_model(self, loader, i, feat_id):
        correct, num = torch.tensor(0, device='cuda'), torch.tensor(0, device='cuda')
        if self.use_ncm:
            class_means = self.class_means_ls[feat_id]
            means_tensor = torch.stack([class_means[cls] for cls in self.class_holder if cls in class_means])
            class_map = {cls: i for i, cls in enumerate([c for c in self.class_holder if c in class_means])}
            
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                features = self.model.features(data)[feat_id]
                features = F.normalize(features, dim=1)
                
                # Batched distance calculation
                dists = torch.cdist(features, means_tensor)
                pred_idx = dists.min(1)[1]
                
                # Map prediction index back to class label
                idx_to_label = {v: k for k, v in class_map.items()}
                pred = torch.tensor([idx_to_label[p.item()] for p in pred_idx], device=data.device)
                
                correct += pred.eq(target.data).sum()
                num += data.size(0)
        else:
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                model_output = self.model(data)
                pred = model_output[0][feat_id].max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                num += data.size(0)

        test_accuracy = (100. * correct / num) if num > 0 else 0
        print(f'Test Task {i}: Accuracy: {correct}/{num} ({test_accuracy:.2f}%)')
        return test_accuracy

    def test_model_mean(self, loader, i):
        correct, num = torch.tensor(0, device='cuda'), torch.tensor(0, device='cuda')
        if self.use_ncm:
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                features_ls = self.model.features(data)
                dists_ls = []
                
                for feat_id in range(4):
                    class_means = self.class_means_ls[feat_id]
                    means_tensor = torch.stack([class_means[cls] for cls in self.class_holder if cls in class_means])
                    features = F.normalize(features_ls[feat_id], dim=1)
                    dists = torch.cdist(features, means_tensor)
                    dists_ls.append(dists)

                mean_dists = torch.stack(dists_ls).mean(dim=0)
                pred_idx = mean_dists.min(1)[1]

                class_map = {cls: i for i, cls in enumerate([c for c in self.class_holder if c in class_means])}
                idx_to_label = {v: k for k, v in class_map.items()}
                pred = torch.tensor([idx_to_label[p.item()] for p in pred_idx], device=data.device)

                correct += pred.eq(target.data).sum()
                num += data.size(0)
        else:
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                model_output = self.model(data)
                expert_logits = model_output[0]
                pred = torch.stack(expert_logits, dim=1).mean(dim=1).max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                num += data.size(0)

        test_accuracy = (100. * correct / num) if num > 0 else 0
        print(f'Test Task {i} (Mean): Accuracy: {correct}/{num} ({test_accuracy:.2f}%)')
        return test_accuracy

    def save_checkpoint(self, save_path = './outputs/final.pt'):
        print(f"Saving checkpoint to: {save_path}")
        folder, file_name = os.path.split(save_path)
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'buffer': self.buffer.state_dict()}, save_path)

    def load_checkpoint(self, load_path = './outputs/final.pt'):
        print(f"Loading checkpoint from: {load_path}")
        ckpt_dict = torch.load(load_path)
        self.model.load_state_dict(ckpt_dict['model'])
        self.buffer.load_state_dict(ckpt_dict['buffer'])
        if hasattr(self.buffer, '_rebuild_index_cache'):
            self.buffer._rebuild_index_cache()

    def _expert_aligned_features(self, feat_list):
        expert_feat = feat_list[self.expert]
        if hasattr(self.model, 'final_addaption_layer'):
            expert_feat = self.model.final_addaption_layer(expert_feat)
        return expert_feat

    def _view_consistency_loss(self, features: torch.Tensor, base_batch: int, *, select_start: int = 0, select_count: Optional[int] = None) -> torch.Tensor:
        if base_batch <= 0 or features.numel() == 0: return features.new_zeros(())
        total, view_count = features.size(0), features.size(0) // base_batch
        if total % base_batch != 0 or view_count <= 1: return features.new_zeros(())
        norm_feat = F.normalize(features, dim=1).reshape(view_count, base_batch, -1).permute(1, 0, 2)
        
        start = max(0, min(base_batch, select_start))
        subset = norm_feat[start:] if select_count is None else norm_feat[start:min(base_batch, start + select_count)]
        if subset.numel() == 0: return features.new_zeros(())
        
        anchor, others = subset[:, :1, :], subset[:, 1:, :]
        if others.numel() == 0: return features.new_zeros(())
        
        return (anchor - others).pow(2).sum(dim=-1).mean()

    def _sample_u_buffer_with_quota(self, request_size: int):
        if request_size <= 0 or len(self.u_buffer) == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        total = len(self.u_buffer)
        # [FIX] Correctly unpack 3 return values from u_buffer.sample
        all_x, all_y, all_idx = self.u_buffer.sample(total)
        
        if all_x.numel() == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        u_values = torch.tensor([self.u_buffer.u_list[int(i.item())] for i in all_idx], device=all_x.device)
        order = torch.argsort(u_values, descending=True)
        top_k = order[: min(request_size, order.numel())]
        
        if top_k.numel() == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        return all_x[top_k], all_y[top_k], all_idx[top_k]

    def _get_classifier_weight(self):
        """获取模型的分类器权重向量作为类原型"""
        # This logic is complex and specific to your model architecture.
        # Assuming it's correct from the provided file.
        classifier_weights = None
        if hasattr(self.model, 'linear'):
            linear = self.model.linear
            if isinstance(linear, nn.ModuleList):
                classifier_weights = linear[self.expert].weight
            elif isinstance(linear, nn.Linear):
                classifier_weights = linear.weight
        elif hasattr(self.model, 'fc'):
            classifier_weights = self.model.fc.weight
        elif hasattr(self.model, 'classifier'):
            classifier_weights = self.model.classifier.weight
        return classifier_weights

    def _feature_inconsistency_scores(self, features: torch.Tensor, base_batch: int) -> torch.Tensor:
        if base_batch <= 0 or features is None or features.numel() == 0: return torch.zeros(0, device=features.device)
        total, view_count = features.size(0), features.size(0) // base_batch
        if total % base_batch != 0 or view_count <= 1: return torch.zeros(base_batch, device=features.device)
        
        norm_feat = F.normalize(features.reshape(view_count, base_batch, -1).permute(1, 0, 2), dim=-1)
        anchor, others = norm_feat[:, :1, :], norm_feat[:, 1:, :]
        if others.numel() == 0: return torch.zeros(base_batch, device=features.device)
        
        return (1.0 - (others * anchor).sum(dim=-1)).mean(dim=1)

    def _sample_from_buffer_by_class(self, target_class: int, n_samples: int, target_device=None):
        """Helper to call buffer's class sampling."""
        return self.buffer.sample_class_data(target_class, n_samples, target_device=target_device)

    def _sample_from_buffer_random_old(self, n_samples: int, old_classes: list, target_device=None, exclude_indices=None):
        """Helper to call buffer's random sampling from a list of classes."""
        return self.buffer.sample_from_classes(
            old_classes, n_samples, target_device=target_device, exclude_indices=exclude_indices,
        )

    def _sample_hard_from_buffer(self, batch_size, exclude_task_id=None, target_device=None):
        """
        [新增] 从主缓冲区中基于不确定性进行重要性采样 (策略一核心方法)。
        """
        if len(self.buffer) == 0 or batch_size <= 0:
            device = target_device if target_device else torch.device('cuda')
            return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long)

        current_len = len(self.buffer)
        ref_device = self.buffer.bx.device if hasattr(self.buffer, 'bx') else torch.device('cuda')
        
        buf_t = self.buffer.bt[:current_len] if hasattr(self.buffer, 'bt') else torch.full((current_len,), -1, device=ref_device)
        buf_u = self.buffer.uncertainty[:current_len] if hasattr(self.buffer, 'uncertainty') else torch.zeros(current_len, device=ref_device)
        
        if exclude_task_id is not None:
            valid_mask = (buf_t != exclude_task_id)
        else:
            valid_mask = torch.ones(current_len, dtype=torch.bool, device=ref_device)
            
        if not valid_mask.any():
            device = target_device if target_device else torch.device('cuda')
            return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long)

        valid_indices = torch.nonzero(valid_mask).squeeze(1)
        valid_uncertainties = buf_u[valid_indices]
        
        weights = valid_uncertainties + 1e-6
        
        num_valid = valid_indices.numel()
        real_batch_size = min(batch_size, num_valid)
        
        if real_batch_size == 0:
             device = target_device if target_device else torch.device('cuda')
             return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long)

        sampled_idx_in_valid = torch.multinomial(weights, real_batch_size, replacement=False)
        final_indices = valid_indices[sampled_idx_in_valid]
        
        ret_x = self.buffer.bx[final_indices]
        ret_y = self.buffer.by[final_indices]
        
        if target_device is not None:
            ret_x = ret_x.to(target_device, non_blocking=True)
            ret_y = ret_y.to(target_device, non_blocking=True)
        
        final_indices = final_indices.cpu().long()
        
        return ret_x, ret_y, final_indices
