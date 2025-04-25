import re
import os
import matplotlib.pyplot as plt

def parse_results(text):
    steps, costs, success = [], [], []
    blocks = text.split("--------------------------------------------------")
    for block in blocks:
        step_match = re.search(r"STEP:\s*(\d+)", block)
        cost_match = re.search(r"COST:\s*([\d.]+)", block)
        goal_match = re.search(r"GOAL:\s*(True|False)", block)
        detached = "detached" in block.lower()

        if step_match and cost_match:
            step = int(step_match.group(1))
            cost = float(cost_match.group(1))
            goal = (goal_match and goal_match.group(1) == "True")
            # if detached:
            #     goal = False  # force override even if GOAL: True

            steps.append(40 if not goal else step)
            costs.append(cost)
            success.append(goal)
    return steps, costs, success

def plot_comparison(file1, file2, file3, file4,
                      labels=("Method 1", "Method 2", "Method 3", "Method 4"),
                      save_path="results/comparison_plot.png"):
    # Load and parse all 4 files
    with open(file1, "r") as f1, open(file2, "r") as f2, open(file3, "r") as f3, open(file4, "r") as f4:
        data1 = parse_results(f1.read())
        data2 = parse_results(f2.read())
        data3 = parse_results(f3.read())
        data4 = parse_results(f4.read())

    # Set up 4 subplots
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))

    def plot_single(ax, steps, costs, success, title):
        trials = range(len(steps))
        ax2 = ax.twinx()

        # Plot cost with red circles
        ax.scatter(trials, costs, c='tab:red', label='Cost', marker='o')
        ax.set_ylabel("Cost", color='tab:red')
        ax.tick_params(axis='y', labelcolor='tab:red')

        # Plot steps with blue crosses
        ax2.scatter(trials, steps, c='tab:blue', label='Steps', marker='x')
        ax2.set_ylabel("Steps", color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Mark failed trials
        for i, s in enumerate(success):
            if not s:
                ax2.annotate("F", (i, steps[i]), textcoords="offset points", xytext=(0, 5),
                            ha='center', color='black', fontsize=10, weight='bold')

        success_rate = sum(success) / len(success) if len(success) > 0 else 0
        ax.set_title(f"{title} (Success Rate: {success_rate:.2%})")
        ax.set_xlabel("Trial")
        ax.grid(True)

    plot_single(axs[0], *data1, labels[0])
    plot_single(axs[1], *data2, labels[1])
    plot_single(axs[2], *data3, labels[2])
    plot_single(axs[3], *data4, labels[3])

    # Save and close
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_comparison(
    "data/bayes_ei_test.txt",
    "data/bayes_ucb_test.txt",
    "data/cma.txt",
    "data/manual_test.txt",
    labels=("BayesOpt (EI)", "BayesOpt (UCB)", "CMA-ES", "Manual"),
    save_path="results/plot_obstacle.png"
)

# plot_comparison(
#     "data/bayes_ei_no_obstacle_test.txt",
#     "data/bayes_ucb_no_obstacle_test.txt",
#     "data/cma_no_obstacle.txt",
#     "data/manual_no_obstacle_test.txt",
#     labels=("BayesOpt (EI)", "BayesOpt (UCB)", "CMA-ES", "Manual"),
#     save_path="results/plot_free.png"
# )