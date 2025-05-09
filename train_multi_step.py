"""
train_multi_step.py - multi-step residual dynamics model training script
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch import optim
import torch.nn as nn

# Project directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'model')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

# Environment configuration
from env.panda_pushing_env import PandaPushingEnv, TARGET_POSE_FREE, DISK_SIZE
from model.learning_state_dynamics import (
    collect_data_random,
    process_data_multiple_step,
    ResidualDynamicsModel,
    SE2PoseLoss,
    MultiStepLoss
)

DEFAULT_CONFIG = {
    # data configuration
    'num_trajectories': 100,
    'trajectory_length': 10,
    'data_save_path': os.path.join(DATA_DIR, 'collected_data.npy'),
    
    # training configuration
    'batch_size': 500,
    'num_steps': 4,
    'discount': 0.9, 
    'lr': 1e-4,
    'num_epochs': 1000,
    'model_save_path': os.path.join(DATA_DIR, 'multi_step_residual.pt')
}

class MultiStepTrainer:
    def __init__(self, config):
        self.config = config
        self.env = PandaPushingEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def collect_data(self):
        """Data Collection Stage"""
        print("\n=== Data Collection Stage ===")
        data = collect_data_random(
            self.env,
            num_trajectories=self.config['num_trajectories'],
            trajectory_length=self.config['trajectory_length']
        )
        print(f"Collected {len(data)} trajectories")
        np.save(self.config['data_save_path'], data)
        return data

    def prepare_loaders(self, data):
        """Date Processing Stage"""
        print("\n=== Date Processing Stage ===")
        return process_data_multiple_step(
            data,
            batch_size=self.config['batch_size'],
            num_steps=self.config['num_steps']
        )

    def train_model(self, train_loader, val_loader):
        """Model Training Process"""
        print("\n=== Model Training Process ===")
        model = ResidualDynamicsModel(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0]
        ).to(self.device)

        criterion = MultiStepLoss(
            SE2PoseLoss(),
            # SE2PoseLoss(block_width=0.1, block_length=0.1),
            discount=self.config['discount']
        )
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])

        best_loss = float('inf')
        train_losses, val_losses = [], []

        with tqdm(range(self.config['num_epochs']), unit='epoch') as pbar:
            for epoch in pbar:
                # training step
                model.train()
                train_loss = self._run_epoch(model, train_loader, criterion, optimizer)
                train_losses.append(train_loss)

                # validation step
                model.eval()
                val_loss = self._run_epoch(model, val_loader, criterion)
                val_losses.append(val_loss)

                # update progress bar
                pbar.set_postfix({
                    'train': f"{train_loss:.4f}",
                    'val': f"{val_loss:.4f}"
                })

                # save the best trained model
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), self.config['model_save_path'])

        return train_losses, val_losses

    def _run_epoch(self, model, loader, criterion, optimizer=None):
        """Execute a single epoch"""
        total_loss = 0.0
        for batch in loader:
            if optimizer: optimizer.zero_grad()
            
            # Move data to the current device
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            next_states = batch['next_state'].to(self.device)

            # Compute the loss
            loss = criterion(model, states, actions, next_states)
            
            # Backpropagation
            if optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def visualize_results(self, train_losses, val_losses):
        """Visualize Training Results"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.semilogy(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(val_losses)
        plt.title("Validation Loss") 
        plt.xlabel("Epoch")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, 'training_curves.png'))
        plt.close()

    def evaluate_model(self):
        """Model Validation Stage"""
        print("\n=== Model Validation Stage ===")
        env = PandaPushingEnv(render_non_push_motions=False, debug=True)
        initial_state = env.reset() 
        # load the trained model
        model = ResidualDynamicsModel(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        ).to(self.device)
        model.load_state_dict(
            torch.load(self.config['model_save_path'], 
            map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        
        # execute the controller
        from controller.pushing_controller import PushingController, free_pushing_cost_function
        controller = PushingController(
            env=env,
            model=model,
            cost_function=free_pushing_cost_function,
            num_samples=100,
            horizon=10
        )
        
        state = env.reset()
        device = next(model.parameters()).device
        for _ in tqdm(range(20), desc="Testing Steps"):
            state_tensor = torch.from_numpy(state).float().to(device)
            action = controller.control(state_tensor)
            state, _, done, _ = env.step(action)
            if done: break
            
        # validate the results
        end_state = env.get_state()
        goal_distance = np.linalg.norm(end_state[:2] - TARGET_POSE_FREE[:2])
        print(f"\nGoal Distance: {goal_distance:.4f} (Threshold: {DISK_SIZE})")
        print(f"Target achieved: {goal_distance < DISK_SIZE}")

if __name__ == "__main__":
    # training configuration
    config = DEFAULT_CONFIG.copy()

    # execute the training pipeline
    trainer = MultiStepTrainer(config)
    trainer.main()