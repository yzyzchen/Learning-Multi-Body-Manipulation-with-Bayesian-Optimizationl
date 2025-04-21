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
from optimizer.panda_pushing_optimizer import PandaBoxPushingStudy
from env.visualizers import GIFVisualizer
from controller.pushing_controller import PushingController, free_pushing_cost_function, obstacle_avoidance_pushing_cost_function


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
        debug=config['debug'],
        include_obstacle=True,
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
    controller = PushingController(
        env=env,
        model=model,
        cost_function=obstacle_avoidance_pushing_cost_function,
        num_samples=100,
        horizon=10
    )

    for episode in range(config['num_episodes']):
        state = env.reset()
        total_reward = 0
        
        with tqdm(total=config['steps'], desc=f"Episode {episode+1}") as pbar:
            # for step in range(config['steps']):
            for step in range(30):
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

def run_opt_demo_with_model():
    """Load the model and run the demo"""
    print("\n=== Load the model and run the demo with bayes optimization===")
    
    # Initialize the model and environment
    test_param_ours_obs = [0.01 , 0.4, 0.4 , 0.4   ] #manual
    # test_param_ours_obs = [0.01827849, 0.39929605, 0.8261565,  0.9678583 ] # bayesian with epoch = 50
    # test_param_ours_obs = [0.5798608491229887, 0.6832310962673614, 0.292713670102513, 0.2677121168629717] # cma with epoch = 50
    # test_param_ours_obs = [0.04671676, 0.7548581, 0.6731052, 0.97566706] # beyasian with obstacle
    # visualizer.reset()
    test_free = PandaBoxPushingStudy(epoch=20, render=True, logdir="logs/", 
                                    study_name="test", 
                                    include_obstacle=True, 
                                    random_target=False,
                                    target_state=np.array([0.95, -0.1, 0.]),
                                    opt_type="test", 
                                    step_scale=0.1, 
                                    device="cpu",
                                    test_params=test_param_ours_obs,
                                    visualizer=None)
    test_free.run()

    cost_mean, cost_var = test_free.get_cost_mean_and_var()

    print("cost_mean", cost_mean)
    print("cost_var", cost_var)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-body dynamics simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Collect data mode
    collect_parser = subparsers.add_parser('collect', help="collecting data")
    collect_parser.add_argument('--num_traj', type=int, default=500,
                              help="number of trajectories")
    collect_parser.add_argument('--traj_len', type=int, default=10,
                               help="trajectory length")
    collect_parser.add_argument('--save_path', type=str, 
                               default='model/collected_data.npy',
                               help="data save path")

    # Model training mode
    train_parser = subparsers.add_parser('train', help="training model") 
    train_parser.add_argument('--num_epochs', type=int, default=1000,
                             help="training epochs")
    train_parser.add_argument('--data_path', type=str,
                             default='model/collected_data.npy',
                             help="training data path")
    train_parser.add_argument('--model_save', type=str,
                            default='model/trained_model/model.pt',
                            help="model save path")

    # Debug mode
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
    
    # Optimizer demo mode
    opt_demo_model_parser = subparsers.add_parser('opt_demo_model', help="Load the model and run the demo with bayes optimization")
    # opt_demo_model_parser.add_argument('--debug', action='store_true',
    #                             help="start the debug mode")

    args = parser.parse_args()
    # mode distribution
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

    elif args.command == 'opt_demo_model':
        # opt_demo_config = {
        #     'debug': args.debug,
        #     'num_episodes': args.episodes,
        #     'steps': args.steps,
        #     'model_save_path': args.model_save
        # }
        run_opt_demo_with_model()

    else:
        parser.print_help()