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
        noise_sigma = 0.4 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01

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
        # state_tensor = state.clone().detach().to(device)
        if not torch.is_tensor(state):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state_tensor = state.detach().to(device)

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
    # target_pose = target_pose#.to(dtype=state.dtype, device=state.device)  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    # cost = None
    # --- Your code here
    # --- Step 1: Extract xy
    target_xy = state[:, 0:2]     # (B, 2)
    inter_xy = state[:, 3:5]      # (B, 2)
    goal_xy = TARGET_POSE_FREE_TENSOR[:2]  # (2,)

    # --- Step 2: cost for reaching goal
    target_error = target_xy - goal_xy     # (B, 2)
    cost_goal = torch.sum(target_error**2, dim=1)  # (B,)

    # --- Step 3: alignment cost using cross product
    v1 = target_xy - inter_xy              # vector: inter → target
    v2 = goal_xy.unsqueeze(0) - inter_xy   # vector: inter → goal
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # 2D cross product (B,)
    alignment_cost = cross ** 2

    # --- Final cost
    cost = cost_goal + 100.0 * alignment_cost
    return cost


def collision_detection(state, soft_margin=0.01):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    obs_half = obstacle_dims / 2 + soft_margin
    disk_radius = DISK_SIZE / 2 + soft_margin  # inflated disk size

    def check_disk_collision(x, y):
        dx = torch.abs(x - obstacle_centre[0])
        dy = torch.abs(y - obstacle_centre[1])
        return (dx <= (obs_half[0] + disk_radius)) & (dy <= (obs_half[1] + disk_radius))

    # Target disk
    x1 = state[:, 0]
    y1 = state[:, 1]
    target_collision = check_disk_collision(x1, y1)

    # Intermediate disk
    x2 = state[:, 3]
    y2 = state[:, 4]
    inter_collision = check_disk_collision(x2, y2)

    # If either is in collision → overall collision
    return (target_collision | inter_collision).float()


def obstacle_avoidance_pushing_cost_function(state, action, 
                                             target_pose=TARGET_POSE_OBSTACLES_TENSOR, 
                                             Q_diag=[100, 100, 0.1], 
                                             base_weight_collision=100.0, weight_alignment=50.0
                                             ):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    # target_pose = target_pose.to(dtype=state.dtype, device=state.device)  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    # # Dynamically adjust the weight of alignment
    # distance_to_goal = torch.norm(target_error, dim=1)
    # weight_alignment_dynamic = weight_alignment * torch.exp(-distance_to_goal)

    # Ensure goal is on correct device
    goal_xy = target_pose[:2].to(state.device, dtype=state.dtype)

    # Extract object and intermediate positions
    target_xy = state[:, 0:2]     # Object being pushed
    inter_xy = state[:, 3:5]      # Pushing object

    # Step 1: Goal distance (squared Euclidean)
    target_error = target_xy - goal_xy
    cost_goal = torch.sum(target_error ** 2, dim=1)

    def get_distance_to_obstacles(state, soft_margin=0.01):
        # Dummy placeholder for demo
        # You should implement this properly based on obstacle positions
        B = state.size(0)
        dummy_distance = torch.ones(B, device=state.device) * 0.5  # assume safe
        # obstacle_centre = OBSTACLE_CENTRE_TENSOR.to(state.device, dtype=state.dtype)
        # obstacle_halfdims = OBSTACLE_HALFDIMS_TENSOR.to(state.device, dtype=state.dtype)

        # # Compute distances from both disks
        # def compute(pos):
        #     delta = torch.abs(pos - obstacle_centre) - obstacle_halfdims - soft_margin
        #     delta_clamped = torch.clamp(delta, min=0.0)
        #     return torch.norm(delta_clamped, dim=1)

        # dist_target = compute(target_xy)
        # dist_inter = compute(inter_xy)

        # return torch.minimum(dist_target, dist_inter)

        return dummy_distance

    # Step 2: Soft obstacle cost (assumes you have a distance function)
    distance_to_obs = get_distance_to_obstacles(state)  # (B,), must return distance
    # Avoid divide-by-zero by adding epsilon
    soft_collision_cost = torch.exp(-10.0 * distance_to_obs.clamp(min=1e-2))

    # Step 3: Adaptive alignment cost using cross product
    v1 = target_xy - inter_xy  # vector from intermediate → object
    v2 = goal_xy.unsqueeze(0) - inter_xy  # vector from intermediate → goal
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    alignment_cost = cross ** 2

    # Weight alignment cost lower when far from goal
    dist_to_goal = torch.norm(target_error, dim=1)
    adaptive_weight_alignment = weight_alignment * torch.exp(-dist_to_goal)

    # Final cost
    cost = cost_goal + base_weight_collision * soft_collision_cost + adaptive_weight_alignment * alignment_cost
    # ---
    return cost
