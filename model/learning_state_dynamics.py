from os import stat
from threading import current_thread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, DISK_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data = []

    # 添加带进度条的外层循环
    for _ in tqdm(range(num_trajectories), 
                 desc="Collecting Trajectories", 
                 unit="traj",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        states = []
        actions = []
        
        state = env.reset()
        states.append(state)
        
        # 内层循环（轨迹步长）不需要进度条
        for _ in range(trajectory_length):
            action = env.action_space.sample()
            next_state, _, _, _ = env.step(action)
            
            actions.append(action)
            states.append(next_state)
            state = next_state
            
        trajectory = {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32)
        }
        collected_data.append(trajectory)
        
    return collected_data


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    dataset = SingleStepDynamicsDataset(collected_data)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset,[train_size, val_size])

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    # ---
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    dataset = MultiStepDynamicsDataset(collected_data, num_steps=num_steps)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here
        trajectory_index = item // self.trajectory_length
        step_index = item % self.trajectory_length

        state = self.data[trajectory_index]['states'][step_index]
        action = self.data[trajectory_index]['actions'][step_index]
        next_state = self.data[trajectory_index]['states'][step_index+1]

        sample = {
            'state': state,
            'action': action,
            'next_state': next_state,
        }

        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here
        trajectory_index = item // self.trajectory_length
        step_index = item % self.trajectory_length

        state = self.data[trajectory_index]['states'][step_index]
        action = self.data[trajectory_index]['actions'][step_index:step_index+self.num_steps]
        next_state = self.data[trajectory_index]['states'][step_index+1:step_index+1+self.num_steps]

        sample = {
            'state': state,
            'action': action,
            'next_state': next_state
        }


        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self):
        super().__init__()
        self.r = DISK_SIZE/2

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        rg = self.r * 0.7071
        mse_x = F.mse_loss(pose_pred[:,0],pose_target[:,0])
        mse_y = F.mse_loss(pose_pred[:,1], pose_target[:,1])
        mse_theta = F.mse_loss(torch.sin(pose_pred[:,2]), torch.sin(pose_target[:,2]))
        mse_theta = mse_theta + F.mse_loss(torch.cos(pose_pred[:,2]), torch.cos(pose_target[:,2]))
        # se2_pose_loss = mse_x + mse_y + rg * mse_theta
        se2_pose_loss = mse_x + mse_y + rg * mse_theta

        mse_x_inter = F.mse_loss(pose_pred[:,3],pose_target[:,3])
        mse_y_inter = F.mse_loss(pose_pred[:,4], pose_target[:,4])
        mse_theta_inter = F.mse_loss(torch.sin(pose_pred[:,5]), torch.sin(pose_target[:,5]))
        mse_theta_inter = mse_theta + F.mse_loss(torch.cos(pose_pred[:,5]), torch.cos(pose_target[:,5]))
        se2_pose_loss_inter = mse_x_inter + mse_y_inter + rg * mse_theta_inter
        # ---
        return se2_pose_loss + se2_pose_loss_inter


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        predicted_state = model(state, action)

        single_step_loss = self.loss(predicted_state, target_state)


        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        multi_step_loss = 0.0
        current_state = state
        discount = 0.99

        for t in range(actions.shape[1]):
          predicted_state = model(current_state, actions[:, t, :])
          step_loss = self.loss(predicted_state, target_states[:, t, :])
          multi_step_loss += discount * step_loss
          discount *= self.discount
          current_state = predicted_state
        # ---
        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.fc1 = nn.Linear(state_dim + action_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, state_dim)
        self.relu = nn.ReLU()
        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        x = torch.cat((state, action), dim=-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        next_state = self.fc3(x)


        # ---
        return next_state


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.fc1 = nn.Linear(state_dim + action_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, state_dim)
        self.relu = nn.ReLU()



        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        x = torch.cat((state, action), dim=-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        delta = self.fc3(x)

        next_state = state + delta

        # ---
        return next_state


def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    # target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    # cost = None
    # # --- Your code here
    # state_pose = state
    # error = state_pose[:, :3] - target_pose
    # Q = torch.diag(torch.tensor([1.0, 1.0, 0.1], dtype=state.dtype, device=state.device))
    # cost = torch.sum(error @ Q * error, dim=1)
    # # ---
    # return cost

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

    obs_half = obstacle_dims / 1

    dx = torch.abs(x - obstacle_centre[0])
    dy = torch.abs(y - obstacle_centre[1])

    collision = (dx <= (eff_half + obs_half[0])) & (dy <= (eff_half + obs_half[1]))
    in_collision = collision.float()


    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR
  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    target_xy = state[:, 0:2]     # (B, 2)
    inter_xy = state[:, 3:5]
    goal_xy = TARGET_POSE_OBSTACLES_TENSOR[:2]
    target_error = target_xy - goal_xy     # (B, 2)
    cost_goal = torch.sum(target_error**2, dim=1)  # (B,)
    collision_cost = collision_detection(state)

    v1 = target_xy - inter_xy              # vector: inter → target
    v2 = goal_xy.unsqueeze(0) - inter_xy   # vector: inter → goal
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # 2D cross product (B,)
    alignment_cost = cross ** 2

    cost = cost_goal + 10000 * collision_cost + 100 * alignment_cost

    # ---
    return cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.4 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from controller.mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        next_state = self.model(state, action)


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
        state_tensor = torch.tensor(state, dtype=torch.float32)
        


        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.cpu().detach().numpy()


        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here



# ---
# ============================================================
