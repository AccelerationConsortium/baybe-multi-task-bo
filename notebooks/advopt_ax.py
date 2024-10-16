# load the Advanced Optimization from AC huggingface
from gradio_client import Client

client = Client("AccelerationConsortium/crabnet-hyperparameter")

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.task_encode import TaskEncode, TaskChoiceToIntTaskChoice
from ax.modelbridge.transforms.unit_x import UnitX
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.typeutils import not_none


def set_seeds(seed=42):
    np.random.seed(seed)


# y1 and y2 are correlated
# Adv Opt function for y1
def adv_opt_y1(
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    c1,
    c2,
    c3,
):
    result = client.predict(
        x1,  # float (numeric value between 0.0 and 1.0) in 'x1' Slider component
        x2,  # float (numeric value between 0.0 and 1.0)	in 'x2' Slider component
        x3,  # float (numeric value between 0.0 and 1.0) in 'x3' Slider component
        x4,  # float (numeric value between 0.0 and 1.0) in 'x4' Slider component
        x5,  # float (numeric value between 0.0 and 1.0) in 'x5' Slider component
        x6,  # float (numeric value between 0.0 and 1.0) in 'x6' Slider component
        x7,  # float (numeric value between 0.0 and 1.0) in 'x7' Slider component
        x8,  # float (numeric value between 0.0 and 1.0) in 'x8' Slider component
        x9,  # float (numeric value between 0.0 and 1.0) in 'x9' Slider component
        x10,  # float (numeric value between 0.0 and 1.0) in 'x10' Slider component
        x11,  # float (numeric value between 0.0 and 1.0) in 'x11' Slider component
        x12,  # float (numeric value between 0.0 and 1.0) in 'x12' Slider component
        x13,  # float (numeric value between 0.0 and 1.0) in 'x13' Slider component
        x14,  # float (numeric value between 0.0 and 1.0) in 'x14' Slider component
        x15,  # float (numeric value between 0.0 and 1.0) in 'x15' Slider component
        x16,  # float (numeric value between 0.0 and 1.0) in 'x16' Slider component
        x17,  # float (numeric value between 0.0 and 1.0) in 'x17' Slider component
        x18,  # float (numeric value between 0.0 and 1.0) in 'x18' Slider component
        x19,  # float (numeric value between 0.0 and 1.0) in 'x19' Slider component
        x20,  # float (numeric value between 0.0 and 1.0) in 'x20' Slider component
        c1,  # Literal['c1_0', 'c1_1'] in 'c1' Radio component
        c2,  # Literal['c2_0', 'c2_1'] in 'c2' Radio component
        c3,  # Literal['c3_0', 'c3_1', 'c3_2'] in 'c3' Radio component
        0.5,  # float (numeric value between 0.0 and 1.0) in 'fidelity1' Slider component
        api_name="/predict",
    )
    return result["data"][0][0]  # return y1 value only


# Adv Opt function for y2
def adv_opt_y2(
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    c1,
    c2,
    c3,
):
    result = client.predict(
        x1,  # float (numeric value between 0.0 and 1.0) in 'x1' Slider component
        x2,  # float (numeric value between 0.0 and 1.0)	in 'x2' Slider component
        x3,  # float (numeric value between 0.0 and 1.0) in 'x3' Slider component
        x4,  # float (numeric value between 0.0 and 1.0) in 'x4' Slider component
        x5,  # float (numeric value between 0.0 and 1.0) in 'x5' Slider component
        x6,  # float (numeric value between 0.0 and 1.0) in 'x6' Slider component
        x7,  # float (numeric value between 0.0 and 1.0) in 'x7' Slider component
        x8,  # float (numeric value between 0.0 and 1.0) in 'x8' Slider component
        x9,  # float (numeric value between 0.0 and 1.0) in 'x9' Slider component
        x10,  # float (numeric value between 0.0 and 1.0) in 'x10' Slider component
        x11,  # float (numeric value between 0.0 and 1.0) in 'x11' Slider component
        x12,  # float (numeric value between 0.0 and 1.0) in 'x12' Slider component
        x13,  # float (numeric value between 0.0 and 1.0) in 'x13' Slider component
        x14,  # float (numeric value between 0.0 and 1.0) in 'x14' Slider component
        x15,  # float (numeric value between 0.0 and 1.0) in 'x15' Slider component
        x16,  # float (numeric value between 0.0 and 1.0) in 'x16' Slider component
        x17,  # float (numeric value between 0.0 and 1.0) in 'x17' Slider component
        x18,  # float (numeric value between 0.0 and 1.0) in 'x18' Slider component
        x19,  # float (numeric value between 0.0 and 1.0) in 'x19' Slider component
        x20,  # float (numeric value between 0.0 and 1.0) in 'x20' Slider component
        c1,  # Literal['c1_0', 'c1_1'] in 'c1' Radio component
        c2,  # Literal['c2_0', 'c2_1'] in 'c2' Radio component
        c3,  # Literal['c3_0', 'c3_1', 'c3_2'] in 'c3' Radio component
        0.5,  # float (numeric value between 0.0 and 1.0) in 'fidelity1' Slider component
        api_name="/predict",
    )
    return result["data"][0][1]  # return y2 value only


set_seeds()

transforms = [TaskChoiceToIntTaskChoice, UnitX]

gs = GenerationStrategy(
    name="MultiTaskOp",
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,
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
        {"name": "x1", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x2", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x3", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x4", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x5", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x6", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x7", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x8", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x9", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x10", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x11", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x12", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x13", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x14", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x15", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x16", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x17", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x18", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x19", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "x20", "type": "range", "bounds": [0.0, 1.0]},
        {
            "name": "c1",
            "type": "choice",
            "is_ordered": False,
            "values": ["c1_0", "c1_1"],
        },
        {
            "name": "c2",
            "type": "choice",
            "is_ordered": False,
            "values": ["c2_0", "c2_1"],
        },
        {
            "name": "c3",
            "type": "choice",
            "is_ordered": False,
            "values": ["c3_0", "c3_1", "c3_2"],
        },
        {
            "name": "Task",
            "type": "choice",
            "values": ["y1", "y2"],
            "is_task": True,
            "target_value": "y2",
        },
    ],
    parameter_constraints=[
        "x19 <= x20",
        "x6 + x15 <= 1.0",
    ],
    objectives={
        "Objective": ObjectiveProperties(minimize=False),
    },
    # overwrite_existing_experiment=True,
    # is_test=True,
)

for i in range(10):
    p, trial_index = ax_client.get_next_trial(
        fixed_features=ObservationFeatures({"Task": "y1" if i % 2 else "y2"}),
    )
    if p["Task"] == "y1":
        u = adv_opt_y1(
            p["x1"],
            p["x2"],
            p["x3"],
            p["x4"],
            p["x5"],
            p["x6"],
            p["x7"],
            p["x8"],
            p["x9"],
            p["x10"],
            p["x11"],
            p["x12"],
            p["x13"],
            p["x14"],
            p["x15"],
            p["x16"],
            p["x17"],
            p["x18"],
            p["x19"],
            p["x20"],
            p["c1"],
            p["c2"],
            p["c3"],
        )
    else:
        u = adv_opt_y2(
            p["x1"],
            p["x2"],
            p["x3"],
            p["x4"],
            p["x5"],
            p["x6"],
            p["x7"],
            p["x8"],
            p["x9"],
            p["x10"],
            p["x11"],
            p["x12"],
            p["x13"],
            p["x14"],
            p["x15"],
            p["x16"],
            p["x17"],
            p["x18"],
            p["x19"],
            p["x20"],
            p["c1"],
            p["c2"],
            p["c3"],
        )

    ax_client.complete_trial(trial_index=trial_index, raw_data={"Objective": u})
