# Imports 
from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, CategoricalParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.constraints import ContinuousLinearConstraint
from baybe.simulation import simulate_scenarios

import numpy as np
import pandas as pd
import seaborn as sns
from baybe.utils.random import set_random_seed
# load the Advanced Optimization from AC huggingface
from gradio_client import Client
client = Client("AccelerationConsortium/crabnet-hyperparameter")

# settings 
BATCH_SIZE = 1 
N_DOE_ITERATIONS = 30
N_MC_ITERATIONS = 5

# define the function 
def adv_opt(c1, c2, c3, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20):
    result = client.predict(
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 
        x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,  # Continuous variables
        c1, c2, c3,  # Categorical variables
        0.5,  # Fidelity
        api_name="/predict",
    )
    return result['data'][0][0]  # return y1 value only

#%% 
# define and create the search space
parameters = [
    NumericalContinuousParameter(name="x1", bounds=(0.0, 1.0)), 
    NumericalContinuousParameter(name="x2", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x3", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x4", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x5", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x6", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x7", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x8", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x9", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x10", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x11", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x12", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x13", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x14", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x15", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x16", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x17", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x18", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x19", bounds=(0.0, 1.0)),
    NumericalContinuousParameter(name="x20", bounds=(0.0, 1.0)),

    CategoricalParameter(name='c1', values=['c1_0', 'c1_1'], encoding="OHE"),
    CategoricalParameter(name='c2', values=['c2_0', 'c2_1'], encoding="OHE"),
    CategoricalParameter(name='c3', values=['c3_0', 'c3_1', 'c3_2'], encoding="OHE"),
]

constraints = [
    ContinuousLinearConstraint(parameters=["x19", "x20"], coefficients=[1.0, -1.0], rhs=0.0, operator='<='),
    ContinuousLinearConstraint(parameters=["x6", "x15"], coefficients=[1.0, 1.0], rhs=1.0, operator='<='), 
]

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)

# define objective
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

#%%
# define scenario
df_result = simulate_scenarios(
    {"Default Recommender": Campaign(searchspace=searchspace,objective=objective)},
    adv_opt, 
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)
#%% 

# plot the results
ax = sns.lineplot(
    data=df_result,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
)
