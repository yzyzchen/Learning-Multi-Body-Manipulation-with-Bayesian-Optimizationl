"""
multi_dynamic_main.py - 主运行入口
支持 collect/train/demo 三种模式
"""
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch

from env.panda_pushing_env import PandaPushingEnv

def collect_data(config):
    """独立数据收集模式"""
    print("\n=== 数据收集模式 ===")
    from model.learning_state_dynamics import collect_data_random
    
    # 确保数据目录存在
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    
    env = PandaPushingEnv()
    data = collect_data_random(
        env,
        num_trajectories=config['num_trajectories'],
        trajectory_length=config['trajectory_length']
    )
    np.save(config['save_path'], data)
    print(f"成功收集 {len(data)} 条轨迹，保存至 {config['save_path']}")

def train_model(config):
    """模型训练模式"""
    print("\n=== 模型训练模式 ===")
    from train_multi_step import MultiStepTrainer, DEFAULT_CONFIG
    
    # 合并默认配置和自定义配置
    full_config = DEFAULT_CONFIG.copy()
    full_config.update(config)
    
    # 初始化训练器
    trainer = MultiStepTrainer(full_config)
    
    # 加载已有数据或重新收集
    if os.path.exists(full_config['data_save_path']):
        print("加载已有数据集")
        data = np.load(full_config['data_save_path'], allow_pickle=True)
    else:
        data = trainer.collect_data()
    
    # 执行完整训练流程
    train_loader, val_loader = trainer.prepare_loaders(data)
    trainer.train_model(train_loader, val_loader)
    trainer.evaluate_model()

def run_demo_with_model(config):
    """加载训练模型进行演示"""
    print("\n=== 模型演示模式 ===")
    
    # 初始化环境和模型
    env = PandaPushingEnv(
        visualizer=True,  # 必须启用渲染
        render_every_n_steps=1,
        debug=config['debug']
    )
    from model.learning_state_dynamics import ResidualDynamicsModel
    model = ResidualDynamicsModel(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # 加载训练好的模型
    model.load_state_dict(torch.load(config['model_save_path'], map_location='cpu'))
    model.eval()
    
    # 初始化控制器
    from model.learning_state_dynamics import PushingController, free_pushing_cost_function
    controller = PushingController(
        env=env,
        model=model,
        cost_function=free_pushing_cost_function,
        num_samples=100,
        horizon=10
    )

    for episode in range(config['num_episodes']):
        state = env.reset()
        total_reward = 0
        
        with tqdm(total=config['steps'], desc=f"Episode {episode+1}") as pbar:
            for step in range(config['steps']):
                # 生成控制动作
                action = controller.control(state)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                
                # 更新状态
                state = next_state
                total_reward += reward
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "reward": f"{total_reward:.2f}",
                    "x": f"{state[0]:.2f}",
                    "y": f"{state[1]:.2f}",
                    "theta": f"{state[2]:.2f}"
                })
                
                if done:
                    print(f"\n目标达成! 累计奖励: {total_reward:.2f}")
                    break

        # 显示最终状态
        end_pose = env.get_state()
        target_pose = env.target_state
        distance = np.linalg.norm(end_pose[:2] - target_pose[:2])
        print(f"最终位置: {end_pose[:2]}, 目标位置: {target_pose[:2]}")
        from env.panda_pushing_env import DISK_SIZE
        print(f"距离目标: {distance:.4f} (要求 < {DISK_SIZE})")

if __name__ == "__main__":
    # 主参数解析器
    parser = argparse.ArgumentParser(
        description="多体动力学系统控制平台",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # 数据收集模式
    collect_parser = subparsers.add_parser('collect', help="数据收集")
    collect_parser.add_argument('--num_traj', type=int, default=100,
                              help="轨迹数量")
    collect_parser.add_argument('--traj_len', type=int, default=10,
                               help="单轨迹长度")
    collect_parser.add_argument('--save_path', type=str, 
                               default='model/collected_data.npy',
                               help="数据保存路径")

    # 模型训练模式
    train_parser = subparsers.add_parser('train', help="模型训练") 
    train_parser.add_argument('--num_epochs', type=int, default=1000,
                             help="训练轮次")
    train_parser.add_argument('--data_path', type=str,
                             default='model/collected_data.npy',
                             help="训练数据路径")
    train_parser.add_argument('--model_save', type=str,
                            default='model/trained_model/model.pt',
                            help="模型保存路径")

    # 在main的parser中添加
    demo_model_parser = subparsers.add_parser('demo_model', help="加载模型进行演示")
    demo_model_parser.add_argument('--model_path', type=str, 
                                default='model/trained_model/model.pt',
                                help="模型路径")
    demo_model_parser.add_argument('--debug', action='store_true',
                                help="启用调试模式")
    demo_model_parser.add_argument('--episodes', type=int, default=1,
                                help="演示回合数")
    demo_model_parser.add_argument('--steps', type=int, default=50,
                             help="最大步数")
    demo_model_parser.add_argument('--model_save', type=str, default='model/trained_model/model.pt',
                             help="模型保存路径")

    args = parser.parse_args()
    # 模式分发
    if args.command == 'collect':
        collect_config = {
            'num_trajectories': args.num_traj,
            'trajectory_length': args.traj_len,
            'save_path': args.save_path
        }
        collect_data(collect_config)
        
    elif args.command == 'train':
        train_config = {
            'num_epochs': args.num_epochs,
            'data_save_path': args.data_path,
            'model_save_path': args.model_save
        }
        train_model(train_config)
        
    elif args.command == 'demo_model':
        demo_config = {
            'debug': args.debug,
            'num_episodes': args.episodes,
            'steps': args.steps,
            'model_save_path': args.model_save
        }
        run_demo_with_model(demo_config)

    else:
        parser.print_help()