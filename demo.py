# =====================================
# Demo for Zhiyin Xu, Yuzhou Chen, Chenen Jin
# ******ATTENSION:THIS DEMO WILL TAKE 3 MINUTES******
# =====================================
# This demo includes the following 8 steps:
# (1) Push without obstacle - Bayesian Optimization (EI)
# (2) Push without obstacle - Bayesian Optimization (UCB)
# (3) Push without obstacle - CMA Optimization
# (4) Push without obstacle - Manual Parameters
# (5) Push with obstacle - Bayesian Optimization (EI)
# (6) Push with obstacle - Bayesian Optimization (UCB)
# (7) Push with obstacle - CMA Optimization
# (8) Push with obstacle - Manual Parameters
# =====================================
import torch
import os
import numpy as np
import time
import random
from optimizer.panda_pushing_optimizer import PandaBoxPushingStudy
from env.visualizers import GIFVisualizer

# ========================== #
#        Terminal Colors     #
# ========================== #
class TerminalColors:
    OKGREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_visualization_warning():
    print(TerminalColors.RED + TerminalColors.BOLD + "=====================================" + TerminalColors.ENDC)
    print(TerminalColors.RED + TerminalColors.BOLD + "Attention: Visualization Window May Be Hidden Below!" + TerminalColors.ENDC)
    print(TerminalColors.RED + TerminalColors.BOLD + "=====================================" + TerminalColors.ENDC)

# ========================== #
# Prompt before each method
# ========================== #
import threading

# Shared flag
user_wants_to_skip = False

def wait_for_enter():
    global user_wants_to_skip
    input()
    user_wants_to_skip = True

import sys
import select

def prompt_to_start(task_name, step_number, epoch, wait_seconds=2):
    print(TerminalColors.OKGREEN + f"Step {step_number}: {task_name} for {epoch} epoch(s)!" + TerminalColors.ENDC)
    print(TerminalColors.OKGREEN + f"Waiting {wait_seconds} seconds... (Press Enter now to SKIP this step!)" + TerminalColors.ENDC)

    print(f"You have {wait_seconds} seconds to press Enter to skip...", flush=True)

    # Wait for input with timeout
    i, o, e = select.select([sys.stdin], [], [], wait_seconds)
    if i:
        sys.stdin.readline()
        print(TerminalColors.RED + f"Skipping Step {step_number}: {task_name}..." + TerminalColors.ENDC)
        return False
    else:
        print("Confirmed: Start pushing!")
        print_visualization_warning()
        time.sleep(2)
        return True

# ========================== #
#          Main Program
# ========================== #
if __name__ == "__main__":
    # Basic settings
    EPOCH = 5
    RENDER = False
    LOGDIR = "logs/"
    DEVICE = "cpu"
    visualizer = GIFVisualizer()

    os.makedirs(LOGDIR, exist_ok=True)

    # Parameters for different methods
    PARAM_MANUAL = [0.7, 0.4, 0.5, 0.3]
    PARAM_EI     = [0.00041003036, 0.54347056, 0.99732959, 0.30594563]
    PARAM_UCB    = [0.00183518, 0.82119733, 0.6133683, 0.6450437]
    PARAM_CMA    = [0.137105297951578, 0.1926368343570713, 0.5743182937039544, 0.4428992162255692]

    # ============ Coder Credits ============ #
    print(TerminalColors.BOLD + "=====================================" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "Demo for Zhiyin Xu, Yuzhou Chen, Chenen Jin" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "******ATTENSION:THIS DEMO WILL TAKE 3 MINUTES******" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "=====================================" + TerminalColors.ENDC)

    print(TerminalColors.BOLD + "This demo includes the following 8 steps:" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(1) Push without obstacle - Bayesian Optimization (EI)" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(2) Push without obstacle - Bayesian Optimization (UCB)" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(3) Push without obstacle - CMA Optimization" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(4) Push without obstacle - Manual Parameters" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(5) Push with obstacle - Bayesian Optimization (EI)" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(6) Push with obstacle - Bayesian Optimization (UCB)" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(7) Push with obstacle - CMA Optimization" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "(8) Push with obstacle - Manual Parameters" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "=====================================" + TerminalColors.ENDC)

    # ======================================== #
    #         Pushing WITHOUT obstacle
    # ======================================== #
    print(TerminalColors.BOLD + "============== WITHOUT OBSTACLE ==============" + TerminalColors.ENDC)

    common_no_obs = {
        "epoch": EPOCH,
        "render": RENDER,
        "logdir": LOGDIR, 
        "include_obstacle": False,
        # "random_target": False,
        "target_state": np.array([0.9, 0, 0]),
        "step_scale": 0.1,
        "device": DEVICE,
        "visualizer": visualizer,
    }

    # Step 1: EI without obstacle
    if prompt_to_start("Bayesian Optimization (EI) (No Obstacle)", 1, EPOCH):
        study_ei_no_obs = PandaBoxPushingStudy(**common_no_obs, study_name="bayes_ei_no_obs", opt_type="test", test_params=PARAM_EI)
        study_ei_no_obs.run()

    # Step 2: UCB without obstacle
    if prompt_to_start("Bayesian Optimization (UCB) (No Obstacle)", 2, EPOCH):
        study_ucb_no_obs = PandaBoxPushingStudy(**common_no_obs, study_name="bayes_ucb_no_obs", opt_type="test", test_params=PARAM_UCB)
        study_ucb_no_obs.run()

    # Step 3: CMA without obstacle
    if prompt_to_start("CMA Optimization (No Obstacle)", 3, EPOCH):
        study_cma_no_obs = PandaBoxPushingStudy(**common_no_obs, study_name="cma_no_obs", opt_type="test", test_params=PARAM_CMA)
        study_cma_no_obs.run()

    # Step 4: Manual without obstacle
    if prompt_to_start("Manual Pushing (No Obstacle)", 4, EPOCH):
        study_manual_no_obs = PandaBoxPushingStudy(**common_no_obs, study_name="manual_no_obs", opt_type="test", test_params=PARAM_MANUAL)
        study_manual_no_obs.run()

    # ======================================== #
    #         Pushing WITH obstacle
    # ======================================== #
    print(TerminalColors.BOLD + "============== WITH OBSTACLE ==============" + TerminalColors.ENDC)

    common_with_obs = {
        "epoch": EPOCH,
        "render": RENDER,
        "logdir": LOGDIR, 
        "include_obstacle": True,
        # "random_target": False,
        "target_state": np.array([0.95, -0.1, 0]),
        "step_scale": 0.1,
        "device": DEVICE,
        "visualizer": visualizer,
    }

    # Step 5: EI with obstacle
    if prompt_to_start("Bayesian Optimization (EI) (With Obstacle)", 5, EPOCH):
        study_ei_with_obs = PandaBoxPushingStudy(**common_with_obs, study_name="bayes_ei_with_obs", opt_type="test", test_params=PARAM_EI)
        study_ei_with_obs.run()

    # Step 6: UCB with obstacle
    if prompt_to_start("Bayesian Optimization (UCB) (With Obstacle)", 6, EPOCH):
        study_ucb_with_obs = PandaBoxPushingStudy(**common_with_obs, study_name="bayes_ucb_with_obs", opt_type="test", test_params=PARAM_UCB)
        study_ucb_with_obs.run()

    # Step 7: CMA with obstacle
    if prompt_to_start("CMA Optimization (With Obstacle)", 7, EPOCH):
        study_cma_with_obs = PandaBoxPushingStudy(**common_with_obs, study_name="cma_with_obs", opt_type="test", test_params=PARAM_CMA)
        study_cma_with_obs.run()

    # Step 8: Manual with obstacle
    if prompt_to_start("Manual Pushing (With Obstacle)", 8, EPOCH):
        study_manual_with_obs = PandaBoxPushingStudy(**common_with_obs, study_name="manual_with_obs", opt_type="test", test_params=PARAM_MANUAL)
        study_manual_with_obs.run()

    # ========================= #
    #          End
    # ========================= #
    print(TerminalColors.OKGREEN + TerminalColors.BOLD + "All pushing demos finished successfully!" + TerminalColors.ENDC)