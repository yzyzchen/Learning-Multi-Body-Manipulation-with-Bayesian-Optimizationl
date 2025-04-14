import numpy as np
import torch
from controller.mppi import MPPI
from functools import partial

from env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_CENTRE, OBSTACLE_HALFDIMS, DISK_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]

class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10, dtype=torch.float, device="cpu"):
        self.env = env
        self.model = model.eval().to(device)
        self.target_state = torch.from_numpy(env.target_state).to(dtype=dtype, device=device)
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = torch.eye(env.action_space.shape[0])
        lambda_value = 1.0

        self.mppi = MPPI(self._compute_dynamics,
                         partial(cost_function, target_pose=self.target_state),
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max,
                         device=device,
                         noise_abs_cost=True)
        self.dtype = dtype
        self.device = device
        self.cost_function = cost_function
        self.parameters = torch.cat([torch.as_tensor(lambda_value).unsqueeze(0), noise_sigma.diagonal()])
        self._default_parameters = self.parameters.clone()

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        with torch.no_grad():
            next_state = self.model(state, action)
        # print(next_state.shape,'next_state')
        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        # state_tensor = torch.from_numpy(state).to(self.device)
        # action_tensor = self.mppi.command(state_tensor)
        # action = action_tensor.detach().cpu().numpy()
        # action = action.clip(self.env.action_space.low, self.env.action_space.high)
        # # ---
        # return action
        device = next(self.model.parameters()).device  # 获取模型所在的设备
        state_tensor = state.clone().detach().to(device)
        action_tensor = self.mppi.command(state_tensor)
        return action_tensor.detach().cpu().numpy()

    def set_parameters(self, hyperparameters):
        # ! Run set_target_state first
        assert self.target_state is not None
        self.mppi.set_parameters(hyperparameters)
        if len(hyperparameters) > 4:
            self.mppi.running_cost = partial(self.cost_function, 
                                             target_pose=self.target_state, 
                                             Q_diag=hyperparameters[4:7])
        else:
            self.mppi.running_cost = partial(self.cost_function, target_pose=self.target_state)
        
        self.parameters = hyperparameters

    def get_parameters(self):
        return self.parameters

    def set_target_state(self, target_state):
        if torch.is_tensor(target_state):
            self.target_state = target_state.to(dtype=self.dtype, device=self.device)
        else:
            self.target_state = torch.from_numpy(target_state).to(dtype=self.dtype, device=self.device)
    
    def get_target_state(self):
        return self.target_state

    def get_cost_total(self):
        return self.mppi.get_cost_total()
    
    def reset(self):
        return self.mppi.reset()
    
    @property
    def default_parameters(self):
        return self._default_parameters


def free_pushing_cost_function(state, action, target_pose, Q_diag=[100, 100, 0.1]):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = target_pose#.to(dtype=state.dtype, device=state.device)  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    state_diff = state[:, :3] - target_pose
    if torch.is_tensor(Q_diag):
        Q = torch.diag(Q_diag).to(dtype=state.dtype, device=state.device)
    else:
        Q = torch.diag(torch.tensor(Q_diag, dtype=state.dtype, device=state.device))
    cost = (state_diff @ Q @ state_diff.T).diagonal()
    # ---
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = DISK_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    x = state[:, 0]
    y = state[:, 1]
    theta = state[:, 2]

    eff_half = (box_size / 2) * (torch.abs(torch.cos(theta)) + torch.abs(torch.sin(theta)))

    obs_half = obstacle_dims / 2

    dx = torch.abs(x - obstacle_centre[0])
    dy = torch.abs(y - obstacle_centre[1])

    collision = (dx <= (eff_half + obs_half[0])) & (dy <= (eff_half + obs_half[1]))
    in_collision = collision.float()
    # ---
    return in_collision.float()


def obstacle_avoidance_pushing_cost_function(state, action, target_pose, Q_diag=[100, 100, 0.1]):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = target_pose#.to(dtype=state.dtype, device=state.device)  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    x = state
    if torch.is_tensor(Q_diag):
        Q = torch.diag(Q_diag).to(dtype=state.dtype, device=state.device)
    else:
        Q = torch.diag(torch.tensor(Q_diag, dtype=state.dtype, device=state.device))
    state_diff = state - target_pose
    cost = (state_diff @ Q @ state_diff.T).diagonal() + 100. * collision_detection(x)
    # ---
    return cost
