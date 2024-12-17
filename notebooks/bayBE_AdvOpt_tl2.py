### Imports 
from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, CategoricalParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.constraints import ContinuousLinearConstraint
from baybe.simulation import simulate_scenarios

import numpy as np
import pandas as pd
import os
import seaborn as sns
from baybe.utils.random import set_random_seed
# load the Advanced Optimization from AC huggingface
from gradio_client import Client
client = Client("AccelerationConsortium/crabnet-hyperparameter")
import time 

### Define the training and testing functions 
# y1 and y2 are correlated 
# Adv Opt function for y1
def adv_opt_y1(c1, c2, c3, Function, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20):
    
	# adding a Function paramter here because in the transfer learning, a "Function" parameter is needed to specify which function to use
	# delibrately set to None
	Function = None 
    
	# try: 
    #     assert x6 + x15 <= 1.0, f"x6: {x6} + x15: {x15} is not less than 1.0"
    # except AssertionError as e:
    #     raise ValueError(f"Assertion failed for x6 + x15 <= 1.0: {e}")
	
    # try: 
    #     assert x19 <= x20, f"x19: {x19} is not less than x20: {x20}"
    # except AssertionError as e:
    #     raise ValueError(f"Assertion failed for x19 <= x20: {e}")
   
	# handle small numerical errors that may occur
	if x6 + x15 > 1.0: 		# round x6 and x15 to 4 decimal places
		x6 = np.round(x6, 4)
		x15 = np.round(x15, 4)

	if x19 >= x20:  		# subtract 1e-6 from x19 and add 1e-6 to x20
		x19 = x19 - 1e-6
		if x19 < 0: 
			x19 = 0.0
		x20 = x20 + 1e-6
		if x20 > 1: 
			x20 = 1.0

	result = client.predict(
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
        x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
        c1, c2, c3,
        0.5,  # Hardcoded fidelity1 value
        api_name="/predict",
    )
	return result['data'][0][0]


# Adv Opt function for y2
def adv_opt_y2(c1, c2, c3, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, *args, **kwargs): 
	result = client.predict(
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 
        x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,  # Continuous variables
        c1, c2, c3,  # Categorical variables
        0.5,  # Fidelity
        api_name="/predict",
    )
	return result['data'][0][1]			# return y2 value only

# %% 
# # run this cell and above for the first time to generate initial data,  
# # we will use the same initial data for all the runs

# # Define initial data  --------------------
# def generate_parameters():
#     while True:
#         # Random float values for x1 to x20 between 0.0 and 1.0
#         params = {f"x{i}": np.random.uniform(0.0, 1.0) for i in range(1, 21)}
        
#         # Random categorical values for c1, c2, c3
#         params["c1"] = np.random.choice(["c1_0", "c1_1"])
#         params["c2"] = np.random.choice(["c2_0", "c2_1"])
#         params["c3"] = np.random.choice(["c3_0", "c3_1", "c3_2"])
        
#         # Check constraints
#         if params["x19"] < params["x20"] and params["x6"] + params["x15"] <= 1.0:
#             return params

# # Create DataFrame for 1000 input data size in number_init_points
# data = [generate_parameters() for _ in range(1000)]
# initial_points = pd.DataFrame(data)
# # make sure c1, c2, c3 are str type
# initial_points['c1'] = initial_points['c1'].apply(str)
# initial_points['c2'] = initial_points['c2'].apply(str)
# initial_points['c3'] = initial_points['c3'].apply(str)

# # create a dataframe, that has initial points for y1 and y2
# # add a Target column for y1/y2 value, and a Function column for the fucntion used 
# lookup_training_y2 = initial_points.copy()
# lookup_training_y2['Target'] = lookup_training_y2.apply(lambda x: adv_opt_y2(**x), axis=1)
# lookup_training_y2['Function'] = "TrainingY2"

# lookup_testing_y1 = initial_points.copy()
# lookup_testing_y1['Target'] = lookup_testing_y1.apply(lambda x: adv_opt_y1(**x), axis=1)
# lookup_testing_y1['Function'] = "TestingY1"

# # save lookup_training_y2 and lookup_testing_y1 to csv
# lookup_testing_y1.to_csv("CrabNet_lookup_testing_y1.csv", index=False)
# lookup_training_y2.to_csv("CrabNet_lookup_training_y2.csv", index=False)

#%%
### Define and create the search space
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

task_parameters = TaskParameter(name="Function", 
                                values = ["TrainingY2", "TestingY1"], 
                                active_values=["TestingY1"]
                                )

constraints = [
    ContinuousLinearConstraint(parameters=["x19", "x20"], coefficients=[1.0, -1.0], rhs=0.0, operator='<='),
    ContinuousLinearConstraint(parameters=["x6", "x15"], coefficients=[1.0, 1.0], rhs=1.0, operator='<='), 
]

params = [*parameters, task_parameters]
searchspace = SearchSpace.from_product(parameters=params, constraints=constraints)

# define objective
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

### Define the tasks
# consider optimizing y1 using training data from y2
test_functions = {
    "TrainingY2": adv_opt_y2,
    "TestingY1": adv_opt_y1,
}

#%% 
# Read the initial data 
lookup_training_y2 = pd.read_csv("CrabNet_lookup_training_y2.csv")
lookup_testing_y1 = pd.read_csv("CrabNet_lookup_testing_y1.csv")

# settings 
BATCH_SIZE = 1  # batch size of recommendations per DOE iteration
N_DOE_ITERATIONS = 5
N_MC_ITERATIONS = 2

# initialize the results dataframe
results: list[pd.DataFrame] = []
for init_size in (50, 100, 500, 700, 1000):
	# reinitialize the campaign
	campaign = Campaign(searchspace=searchspace, objective=objective)
	# add the initial data
	initial_data = [lookup_training_y2.sample(init_size) for _ in range(N_MC_ITERATIONS)]

	result_temp = simulate_scenarios(
		{f"{init_size}": campaign}, 
		adv_opt_y1,
		initial_data = initial_data,
		batch_size = BATCH_SIZE,
		n_doe_iterations = N_DOE_ITERATIONS,
	)
	results.append(result_temp)


#%%

# plot the results
ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
	hue="% of data used"
)


# #%% 
# # create AdvOptResults folder if not exist
# if not os.path.exists('AdvOpt_LKJPrior_1106'):
# 	os.makedirs('AdvOpt_LKJPrior_1106')
# # save the results 
# results_testing_y1 = results[results['Function'] == 'TestingY1']
# results_testing_y1.to_csv(f"AdvOpt_LKJPrior_1106/init_{init_size}_BayBE_5round.csv", index=False)
	
