import argparse
import os
import random
import warnings
import sys
from datetime import datetime

import numpy as np
import torch

# 注意：保持你原有代码的其他导入和类定义不变
from agent import METHODS
from experiment.dataset import DATASETS
from multi_runs import multiple_run
from multi_runs_joint import multiple_run_joint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

# ===== 以下是你原有代码的 Tee 类、get_params、setup_logging 等函数，保持不变 =====
class Tee:
    """同时将输出写入文件和标准输出的类"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 立即刷新确保写入文件
    
    def flush(self):
        for f in self.files:
            f.flush()

def get_params():
    parser = argparse.ArgumentParser()
    # experiment related
    parser.add_argument('--dataset',            default='cifar10',  type=str, choices=DATASETS.keys())
    parser.add_argument('--n_tasks',            default='100',       type=int)
    parser.add_argument('--buffer_size',        default=200,        type=int)
    parser.add_argument('--method',             default='mose',     type=str, choices=METHODS.keys())

    parser.add_argument('--seed',               default=0,          type=int)
    parser.add_argument('--run_nums',           default=10,         type=int)
    parser.add_argument('--epoch',              default=1,          type=int)
    parser.add_argument('--lr',                 default=1e-3,       type=float)
    parser.add_argument('--wd',                 default=1e-4,       type=float)
    parser.add_argument('--batch_size',         default=10,         type=int)
    parser.add_argument('--buffer_batch_size',  default=64,         type=int)

    parser.add_argument('--continual',          default='on',       type=str, choices=['off', 'on'])

    # mose control
    parser.add_argument('--ins_t',              default=0.07,       type=float)
    parser.add_argument('--expert',             default='3',        type=str, choices=['0','1','2','3'])
    parser.add_argument('--n_experts',          default=4,          type=int)
    parser.add_argument('--classifier',         default='ncm',      type=str, choices=['linear', 'ncm'])
    parser.add_argument('--augmentation',       default='ocm',      type=str, choices=['ocm', 'scr', 'none', 'simclr', 'randaug', 'trivial'])

    parser.add_argument('--gpu_id',             default=0,          type=int)
    parser.add_argument('--n_workers',          default=8,          type=int)

    # logging 
    parser.add_argument('--exp_name',           default='tmp',      type=str)
    parser.add_argument('--wandb_project',      default='ocl',      type=str)
    parser.add_argument('--wandb_entity',                           type=str)
    parser.add_argument('--wandb_log',          default='off',      type=str, choices=['off', 'online'])
    
    # 添加日志目录参数
    parser.add_argument('--log_dir',            default='./logs_LJY',   type=str, help='日志文件保存目录')
    # 关键：添加 parse_known_args 支持，避免手动修改参数时报错
    args, _ = parser.parse_known_args()
    return args

def setup_logging(args):
    """设置日志记录"""
    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成日志文件名
    log_filename = f"{args.exp_name}_{args.method}_{timestamp}.txt"
    log_path = os.path.join(args.log_dir, log_filename)
    
    # 保存日志文件路径到args中，供其他地方使用
    args.log_file = log_path
    
    # 创建日志文件
    log_file = open(log_path, 'w', encoding='utf-8')
    
    # 记录实验参数
    log_file.write('=' * 100 + '\n')
    log_file.write(f'Experiment Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    log_file.write('Arguments:\n')
    for arg in vars(args):
        log_file.write(f'\t{arg}: {getattr(args, arg)}\n')
    log_file.write('=' * 100 + '\n\n')
    
    # 重定向标准输出到文件和终端
    original_stdout = sys.stdout
    tee = Tee(original_stdout, log_file)
    sys.stdout = tee
    
    return log_file, original_stdout

def restore_logging(log_file, original_stdout):
    """恢复标准输出并关闭日志文件"""
    sys.stdout = original_stdout
    if log_file:
        # 记录结束时间
        log_file.write('\n' + '=' * 100 + '\n')
        log_file.write(f'Experiment End Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        log_file.write('=' * 100 + '\n')
        log_file.close()
        print(f"Logs saved to: {log_file.name}")

def main(args):
    torch.cuda.set_device(args.gpu_id)
    args.cuda = torch.cuda.is_available()

    # 设置日志记录
    log_file, original_stdout = setup_logging(args)

    try:
        print('=' * 100)
        print('Arguments =')
        for arg in vars(args):
            print('\t' + arg + ':', getattr(args, arg))

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print('[CUDA is unavailable]')

        if args.continual == 'on':
            multiple_run(args)
        else:
            multiple_run_joint(args)
            
    except Exception as e:
        # 记录异常信息
        import traceback
        print(f"\n{'*' * 50} ERROR {'*' * 50}")
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        print('*' * 107)
        
        # 在日志文件中也记录错误
        if log_file:
            log_file.write(f"\n{'*' * 50} ERROR {'*' * 50}\n")
            log_file.write(f"Error occurred: {str(e)}\n")
            traceback.print_exc(file=log_file)
            log_file.write('*' * 107 + '\n')
            
    finally:
        # 恢复标准输出并关闭日志文件
        restore_logging(log_file, original_stdout)

# ===== 关键修改：主函数入口，连续执行两次实验 =====
if __name__ == '__main__':
    # 第一次实验：参数为 cifar100, buffer_size=5000 等
    print("===== 开始执行第一次实验 =====")
    args1 = get_params()
    # 手动修改第一次实验的参数
    args1.dataset = "cifar100"
    args1.buffer_size = 5000
    args1.method = "mose"
    args1.seed = 0
    args1.run_nums = 5
    args1.gpu_id = 0
    args1.n_tasks = 10
    main(args1)  # 执行第一次实验

    # 第二次实验：参数为 cifar10, buffer_size=5000 等
    print("\n===== 第一次实验完成，开始执行第二次实验 =====")
    args2 = get_params()
    # 手动修改第二次实验的参数
    args2.dataset = "cifar10"
    args2.buffer_size = 5000
    args2.method = "mose"
    args2.seed = 0
    args2.run_nums = 5
    args2.gpu_id = 0
    args2.n_tasks = 10
    main(args2)  # 执行第二次实验

    print("\n===== 两次实验均执行完成 =====")
    # 注意：如果不需要自动关机，可删除以下两行
    print("关机")
    os.system("shutdown -h now")