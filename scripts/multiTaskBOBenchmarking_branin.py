"""
@Time    :   2024/09/30 11:17:24
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   multi-task BO benchmarking using the branin function
"""

#%%
# IMPORT DEPENDENCIES----------------------------------------------------------
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper
from baybe.utils.plotting import create_example_plots
import argparse

def branin(x1, x2):
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

# Shifted and inverted Branin function
def shifted_inverted_branin(x1, x2):
    return -branin(x1 + 2.5, x2 + 2.5) + 300

# #%%
# # ARGPARSER--------------------------------------------------------------------

# parse = argparse.ArgumentParser(
#     description='Run the multi-task BO benchmarking using the branin function'
# )

# parse.add_argument('--saveResults', type=bool, required=True)
# parse.add_argument('--pathToSave', type=str, required=False)
# parse.add_argument('--runName', type=str, required=False)

# boolSaveResults = parse.parse_args().saveResults
# strPathToSave = parse.parse_args().pathToSave
# strRunName = parse.parse_args().runName

# if boolSaveResults:
#     strPathToSave = os.path.join(strPathToSave, 'branin', strRunName)
#     # make a directory to save the results
#     os.makedirs(strPathToSave, exist_ok=True)

boolSaveResults = False
strPathToSave = ''
strRunName = ''

#%%
# SETTINGS---------------------------------------------------------------------

BATCH_SIZE = 1
N_MC_ITERATIONS = 10
N_DOE_ITERATIONS = 10
POINTS_PER_DIM = 11

#%%
# CREATE OPTIMIZATION OBJECTIVE------------------------------------------------

objective = SingleTargetObjective(
    target=NumericalTarget(
        name='Target',
        mode='MIN'
    )
)

#%%
# CREATE SEARCH SPACE----------------------------------------------------------

BOUNDS = torch.tensor([[-5, 5], [-5, 5]])

descrete_param = [
    NumericalDiscreteParameter(
        name=f'x{d}',
        values=np.linspace(lower, upper, POINTS_PER_DIM)
    )
    for d, (lower, upper) in enumerate(BOUNDS)
]

task_param = TaskParameter(
    name="Function",
    values=["Test_Function", "Training_Function"],
    active_values=["Test_Function"]
)

parameters = [*descrete_param, task_param]
searchSpace = SearchSpace.from_product(parameters)

#%%
# GENERATE LOOKUP TABLE--------------------------------------------------------

test_functions = {
    "Test_Function": branin,
    "Training_Function": shifted_inverted_branin
}

grid = np.meshgrid(*[param.values for param in descrete_param])

lookups: dict[str, pd.DataFrame] = {}
for function_name, function in test_functions.items():
    lookup = pd.DataFrame({f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)})
    lookup["Target"] = lookup.apply(lambda row: function(*row.values), axis=1)
    lookup["Function"] = function_name
    lookups[function_name] = lookup
lookup_training_task = lookups["Training_Function"]
lookup_test_task = lookups["Test_Function"] 


#%%
# VISUALIZE LOOKUP TABLE-------------------------------------------------------



# %%
fig, ax = plt.subplots(
    1, 3,
    figsize=(21, 5),
    facecolor='w',
    edgecolor='k',
    dpi = 300,
    constrained_layout = True
)

    # TRAINING TASK
ax[0].scatter(
    lookup_training_task["x0"],
    lookup_training_task["x1"],
    c=lookup_training_task["Target"],
    cmap='viridis'
)
ax[0].set_title("Training Task")
ax[0].set_xlabel("x0")
ax[0].set_ylabel("x1")

# add colorbar
cbar = fig.colorbar(ax[0].collections[0], ax=ax[0])
cbar.set_label("Target")

    # TEST TASK
ax[1].scatter(
    lookup_test_task["x0"],
    lookup_test_task["x1"],
    c=lookup_test_task["Target"],
    cmap='viridis'
)
ax[1].set_title("Test Task")
ax[1].set_xlabel("x0")
ax[1].set_ylabel("x1")

# add colorbar
cbar = fig.colorbar(ax[1].collections[0], ax=ax[1])
cbar.set_label("Target")

    # TRAINING AND TEST TASKS
ax[2].scatter(
    lookup_training_task["Target"],
    lookup_test_task["Target"]
)
ax[2].set_xlabel("Training Target")
ax[2].set_ylabel("Test Target")


# find the correlation between the training and test tasks
pearsonCorrlation_Targets = lookup_training_task["Target"].corr(lookup_test_task["Target"])
r2_Targets = pearsonCorrlation_Targets**2

# add the correlation to the plot
ax[2].text(
    0.05, 0.95,
    f"R^2 = {r2_Targets:.2f}",
    transform=ax[2].transAxes
)
ax[2].text(
    0.05, 0.85,
    f"R = {pearsonCorrlation_Targets:.2f}",
    transform=ax[2].transAxes
)

# add a grid
ax[2].grid()


if boolSaveResults:
    plt.savefig(os.path.join(strPathToSave, "training_test_tasks.png"))

#%%
# SIMULATE SCENARIOS----------------------------------------------------------

results: list[pd.DataFrame] = []
for p in (0.01, 0.02, 0.05, 0.08, 0.2):
    print(f"Simulating {p}...")
    campaign = Campaign(searchspace=searchSpace, objective=objective)
    initial_data = [lookup_training_task.sample(frac=p) for _ in range(N_MC_ITERATIONS)]
    results_fraction = simulate_scenarios(
        {f"{int(100*p)}": campaign},
        lookup_test_task,
        initial_data=initial_data,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
    )
    results.append(results_fraction)

# optimize the function without any initial data
print("Simulating 0...")
result_baseline = simulate_scenarios(
    {"0": Campaign(searchspace=searchSpace, objective=objective)},
    lookup_test_task,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS
)
results = pd.concat([result_baseline, *results])

# optimize the function taskparam
parameters_noTask = [*descrete_param]
searchSpace_noTask = SearchSpace.from_product(parameters_noTask)

# generate the lookup table for the no task parameter
grid_noTask = np.meshgrid(*[param.values for param in parameters_noTask])
lookup_noTask = pd.DataFrame({f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid_noTask)})
lookup_noTask["Target"] = lookup_noTask.apply(lambda row: branin(*row.values), axis=1)

# simulate the scenarios
print("Simulating no task...")
result_noTask = simulate_scenarios(
    {'noTask': Campaign(searchspace=searchSpace_noTask, objective=objective)},
    lookup_noTask,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS
)

results = pd.concat([results, result_noTask])

results.rename(columns={"Scenario": "% of data used"}, inplace=True)
#%%
# intialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(15, 5))

# plot the results
ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="% of data used",
)

# decrese the y limit
ax.set_ylim(0, 50)

# find the minimum value of the test function
minValue = lookup_test_task["Target"].min()
# add a line at the maximum value
ax.axhline(minValue, color='red', linestyle='--', label='Global Minimum')

# add a legend
plt.legend()

# add a title
# plt.title("Multi-Task Learning Optimization")

# add a x-axis label
plt.xlabel("Number of Experiments")

# add a y-axis label
plt.ylabel("Target Cumulative Best")

if boolSaveResults:
    # save the dataframe
    results.to_csv(os.path.join(strPathToSave, "dfResults.csv"))
    # save the plot
    plt.savefig(os.path.join(strPathToSave, "Results.png"))

# %%
