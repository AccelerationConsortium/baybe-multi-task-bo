import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.modelbridge.transforms.choice_encode import ChoiceToNumericChoice
from ax.modelbridge.transforms.unit_x import UnitX
from ax.service.ax_client import AxClient, ObjectiveProperties


# Function to set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)


# Branin function with a categorical variable placeholder
def branin(x1, x2, c1):
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    # Add a penalty based on the categorical variable
    penalty_lookup = {"A": 1.0, "B": 0.0, "C": 2.0}
    y += penalty_lookup[c1]

    return y


# Shifted and inverted Branin function with a categorical variable placeholder
def shifted_inverted_branin(x1, x2, c1):
    return -branin(x1 + 2.5, x2 + 2.5, c1) + 300


set_seeds()  # setting the random seed for reproducibility

transforms = [TaskChoiceToIntTaskChoice, UnitX, ChoiceToNumericChoice]

gs = GenerationStrategy(
    name="MultiTaskOp",
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=10,
            model_kwargs={"deduplicate": True, "transforms": transforms},
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs={"transforms": transforms},
        ),
    ],
)

ax_client = AxClient(generation_strategy=gs, random_seed=42, verbose_logging=False)

ax_client.create_experiment(
    name="MultiTaskOp",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "value_type": "float",
            "bounds": [-5.0, 10.0],
        },
        {
            "name": "x2",
            "type": "range",
            "value_type": "float",
            "bounds": [0.0, 15.0],
        },
        {
            "name": "Task",
            "type": "choice",
            "values": ["A", "B"],
            "is_task": True,
            "target_value": "B",
        },
        {
            "name": "c1",
            "type": "choice",
            "is_ordered": False,
            "values": ["A", "B", "C"],
        },
    ],
    objectives={"Objective": ObjectiveProperties(minimize=False)},
)

for i in range(40):
    p, trial_index = ax_client.get_next_trial(
        fixed_features=ObservationFeatures({"Task": "A" if i % 2 else "B"})
    )

    if p["Task"] == "A":
        u = branin(p["x1"], p["x2"], p["c1"])
    else:
        u = shifted_inverted_branin(p["x1"], p["x2"], p["c1"])

    ax_client.complete_trial(trial_index=trial_index, raw_data={"Objective": u})

df = ax_client.get_trials_data_frame()
df_A = df[df["Task"] == "A"]
df_B = df[df["Task"] == "B"]

# return the parameters as a dict for the row with the highest objective value
optimal_parameters_A = df_A.loc[df_A["Objective"].idxmax()].to_dict()
optimal_parameters_B = df_B.loc[df_B["Objective"].idxmax()].to_dict()

objective_A = optimal_parameters_A["Objective"]
objective_B = optimal_parameters_B["Objective"]

print(f"Optimal parameters for task A: {optimal_parameters_A}")
print(f"Optimal parameters for task B: {optimal_parameters_B}")
print(f"Objective for task A: {objective_A}")
print(f"Objective for task B: {objective_B}")
