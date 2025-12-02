# /ablation_project/ablation_main.py

import argparse
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agent import METHODS
from experiment.dataset import DATASETS
from multi_runs import multiple_run
from multi_runs_joint import multiple_run_joint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

from agent.ablation_mose import AblationMOSE_v3

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')


# ==================== 模拟组件 (为了独立运行) ====================
class MockModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.experts = nn.ModuleList([nn.Linear(8 * 16 * 16, 64) for _ in range(4)])
        self.projectors = nn.ModuleList([nn.Linear(64, 32) for _ in range(4)])
        self.classifiers = nn.ModuleList([nn.Linear(64, num_classes) for _ in range(4)])
        self.adapter = nn.Identity()

    def features(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        return [expert(feat) for expert in self.experts]

    def head(self, feat_list, use_proj=False):
        if use_proj:
            return [proj(feat) for proj, feat in zip(self.projectors, feat_list)]
        else:
            return [cls(feat) for cls, feat in zip(self.classifiers, feat_list)]

    def final_addaption_layer(self, x):
        return self.adapter(x)


class MockBuffer:
    def __init__(self, capacity=200, device='cuda'):
        self.capacity, self.device = capacity, device
        self.x, self.y, self.tasks = torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long), torch.empty(0,
                                                                                                              dtype=torch.long)
        self.current_index = 0

    def __len__(self):
        return self.x.shape[0]

    def add_reservoir(self, x, y, t, **kwargs):
        x, y = x.cpu(), y.cpu()
        for i in range(x.shape[0]):
            if len(self) < self.capacity:
                self.x, self.y, self.tasks = torch.cat((self.x, x[i:i + 1])), torch.cat(
                    (self.y, y[i:i + 1])), torch.cat((self.tasks, torch.tensor([t], dtype=torch.long)))
            else:
                j = random.randint(0, self.current_index)
                if j < self.capacity: self.x[j], self.y[j], self.tasks[j] = x[i], y[i], t
            self.current_index += 1

    def sample(self, batch_size, exclude_task=None, ret_ind=True):
        if len(self) == 0: return None, None, None, None
        indices = torch.arange(len(self))
        if exclude_task is not None: indices = indices[self.tasks != exclude_task]
        if len(indices) == 0: return None, None, None, None
        perm = torch.randperm(len(indices))[:min(batch_size, len(indices))]
        final_indices = indices[perm]
        return self.x[final_indices].to(self.device), self.y[final_indices].to(self.device), None, final_indices


def get_mock_dataloaders(n_tasks, n_classes_per_task, samples_per_task, batch_size):
    task_loaders = []
    for i in range(n_tasks):
        start_class, end_class = i * n_classes_per_task, (i + 1) * n_classes_per_task
        data = torch.randn(samples_per_task, 3, 32, 32)
        labels = torch.randint(start_class, end_class, (samples_per_task,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        task_loaders.append({'train': loader, 'test': loader})
    return task_loaders


# ====================================================================


def get_params():
    parser = argparse.ArgumentParser(description='Ablation Study Runner for MOSE v3')

    # --- 您的原始参数 ---
    parser.add_argument('--dataset', default='cifar100', type=str, choices=DATASETS.keys())
    parser.add_argument('--buffer_size', default=500, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--buffer_batch_size', default=32, type=int)
    parser.add_argument('--ins_t', default=0.07, type=float)
    parser.add_argument('--expert', default='3', type=str)
    parser.add_argument('--classifier', default='ncm', type=str, choices=['linear', 'ncm'])
    parser.add_argument('--augmentation', default='ocm', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)

    # --- 消融实验控制开关 (默认为True, 添加flag则变为False) ---
    parser.add_argument('--no_kd', action='store_false', dest='use_kd', help='Disable Knowledge Distillation (w/o KD)')
    parser.add_argument('--no_idm', action='store_false', dest='use_idm',
                        help='Disable Internal Distillation (w/o IDM)')
    parser.add_argument('--no_udb', action='store_false', dest='use_udb',
                        help='Disable Uncertainty Dual-Buffer (w/o U-DB)')

    # --- 所有模块的超参数 (用于精细控制) ---
    # KD
    parser.add_argument('--distill_weight', type=float, default=1.0, help='Weight for KD loss')
    parser.add_argument('--distill_temperature', type=float, default=2.0, help='Temperature for KD')
    # IDM - D2S
    parser.add_argument('--d2s_weight', type=float, default=0.5, help='Weight for Deep-to-Shallow loss')
    parser.add_argument('--d2s_temperature', type=float, default=2.0, help='Temperature for D2S')
    # IDM - TSD
    parser.add_argument('--tsd_kl_weight', type=float, default=0.5, help='Weight for Teacher-Student KL loss (alpha)')
    parser.add_argument('--tsd_l2_weight', type=float, default=0.1, help='Weight for Teacher-Student L2 loss (lambda)')
    parser.add_argument('--tsd_temperature', type=float, default=4.0, help='Temperature for TSD')
    # U-DB
    parser.add_argument('--u_buffer_capacity', type=int, default=64, help='Capacity of the uncertainty buffer')
    parser.add_argument('--u_buffer_class_quota', type=int, default=7,
                        help='Max samples per class in U-Buffer sampling')

    args = parser.parse_args()

    # 从DATASETS字典填充缺失的参数
    dataset_info = DATASETS[args.dataset]
    args.n_classes = dataset_info['n_classes']
    args.input_size = dataset_info['input_size']
    args.n_tasks = dataset_info['n_tasks']

    return args


def main(args):
    # 环境设置
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args): print(f'\t{arg}: {getattr(args, arg)}')
    print('=' * 100)

    # 随机种子
    np.random.seed(args.seed);
    random.seed(args.seed);
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 模拟您的 `multiple_run` 核心逻辑 ---
    print("Initializing mock components for a single run...")

    # 1. 准备数据
    n_classes_per_task = args.n_classes // args.n_tasks
    task_loaders = get_mock_dataloaders(args.n_tasks, n_classes_per_task, 100, args.batch_size)

    # 2. 准备模型、优化器和缓冲区
    model = MockModel(num_classes=args.n_classes).to(device)
    buffer = MockBuffer(capacity=args.buffer_size, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 3. 实例化 Agent
    agent = AblationMOSE_v3(model, buffer, optimizer, args.input_size, args)

    # 4. 执行持续学习循环
    all_task_accuracies = []
    for i in range(args.n_tasks):
        print(f"************ Learning Task {i + 1}/{args.n_tasks} ************")
        train_loader = task_loaders[i]['train']
        agent.train(task_id=i, train_loader=train_loader)

        acc_list, _ = agent.test(i, task_loaders)
        current_avg_acc = acc_list[:i + 1].mean()
        all_task_accuracies.append(current_avg_acc)
        print(f"Average accuracy after task {i + 1}: {current_avg_acc:.2f}%")

    final_accuracy = all_task_accuracies[-1]
    print("\n======== Experiment Finished ========")
    print(f"Final average accuracy at the end of training: {final_accuracy:.2f}%")
    # ---------------------------------------------


if __name__ == '__main__':
    args = get_params()
    main(args)

# 打开终端，进入 ablation_project 文件夹，然后运行以下命令。
#
# 实验1: 运行完整模型 (Ours)
#
# BASH
# python ablation_main.py --dataset cifar100 --epoch 2 --lr 0.001
# 实验2: 运行 w/o U-DB (移除不确定性双缓冲区)
#
# BASH
# python ablation_main.py --dataset cifar100 --epoch 2 --lr 0.001 --no_udb
# 实验3: 运行 w/o IDM (移除内部蒸馏)
#
# BASH
# python ablation_main.py --dataset cifar100 --epoch 2 --lr 0.001 --no_idm
# 实验4: 运行 w/o KD (移除知识蒸馏)
#
# BASH
# python ablation_main.py --dataset cifar100 --epoch 2 --lr 0.001 --no_kd
# 实验5: 运行 Baseline (最接近原版 MOSE)
#
# BASH
# python ablation_main.py --dataset cifar100 --epoch 2 --lr 0.001 --no_kd --no_idm --no_udb
# 集成到您的项目:

