"""
multi_dynamic_main.py - Main script for multi-body dynamics simulation
Support collect/train/demo modes
"""
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch

from env.panda_pushing_env import PandaPushingEnv

def collect_data(config):
    """Collecting data mode"""
    print("\n=== collecting data mode ===")
    from model.learning_state_dynamics import collect_data_random
    
    # Ensure the save path exists
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    
    env = PandaPushingEnv()
    data = collect_data_random(
        env,
        num_trajectories=config['num_trajectories'],
        trajectory_length=config['trajectory_length']
    )
    np.save(config['save_path'], data)
    print(f"Successfully collected {len(data)} trajectories，saved to {config['save_path']}")

def train_model(config):
    """Model training mode"""
    print("\n=== model training mode ===")
    from train_multi_step import MultiStepTrainer, DEFAULT_CONFIG
    
    # Merge the default configuration with the user configuration
    full_config = DEFAULT_CONFIG.copy()
    full_config.update(config)
    
    # Initialize the trainer
    trainer = MultiStepTrainer(full_config)
    
    # Load the existing data or collect new data
    if os.path.exists(full_config['data_save_path']):
        print("Load the existing data")
        data = np.load(full_config['data_save_path'], allow_pickle=True)
    else:
        data = trainer.collect_data()
    
    # Execute the training process
    train_loader, val_loader = trainer.prepare_loaders(data)
    trainer.train_model(train_loader, val_loader)
    trainer.evaluate_model()

def run_demo_with_model(config):
    """Load the model and run the demo"""
    print("\n=== Load the model and run the demo ===")
    
    # Initialize the model and environment
    env = PandaPushingEnv(
        visualizer=True, 
        render_every_n_steps=1,
        debug=config['debug']
    )
    from model.learning_state_dynamics import ResidualDynamicsModel
    model = ResidualDynamicsModel(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # Load the trained model
    model.load_state_dict(torch.load(config['model_save_path'], map_location='cpu'))
    model.eval()
    
    # Initialize the controller
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
                # Generate action
                action = controller.control(state)
                
                # Execute the action
                next_state, reward, done, _ = env.step(action)
                
                # Update the state and reward
                state = next_state
                total_reward += reward
                
                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "reward": f"{total_reward:.2f}",
                    "x": f"{state[0]:.2f}",
                    "y": f"{state[1]:.2f}",
                    "theta": f"{state[2]:.2f}"
                })
                
                if done:
                    print(f"\nTarget achieved! Accumulated reward: {total_reward:.2f}")
                    break

        # Display the final state and distance to the target
        end_pose = env.get_state()
        target_pose = env.target_state
        distance = np.linalg.norm(end_pose[:2] - target_pose[:2])
        print(f"Final Position: {end_pose[:2]}, Target Position: {target_pose[:2]}")
        from env.panda_pushing_env import DISK_SIZE
        print(f"Final Distance: {distance:.4f} (Requirement < {DISK_SIZE})")

if __name__ == "__main__":
    # 主参数解析器
    parser = argparse.ArgumentParser(
        description="Multi-body dynamics simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # 数据收集模式
    collect_parser = subparsers.add_parser('collect', help="数据收集")
    collect_parser.add_argument('--num_traj', type=int, default=300,
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
    demo_model_parser = subparsers.add_parser('demo_model', help="Load the model and run the demo")
    demo_model_parser.add_argument('--model_path', type=str, 
                                default='model/trained_model/model.pt',
                                help="model path")
    demo_model_parser.add_argument('--debug', action='store_true',
                                help="start the debug mode")
    demo_model_parser.add_argument('--episodes', type=int, default=1,
                                help="demo episodes")
    demo_model_parser.add_argument('--steps', type=int, default=50,
                             help="maximum steps per episode")
    demo_model_parser.add_argument('--model_save', type=str, default='model/trained_model/model.pt',
                             help="model save path")

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