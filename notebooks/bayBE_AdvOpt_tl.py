### Imports 
from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter, CategoricalParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.constraints import ContinuousLinearInequalityConstraint
import numpy as np
import pandas as pd
import os
from baybe.utils.random import set_random_seed
# load the Advanced Optimization from AC huggingface
from gradio_client import Client
client = Client("AccelerationConsortium/crabnet-hyperparameter")
import time 

# settings 
BATCH_SIZE = 1  # batch size of recommendations per DOE iteration
N_DOE_ITERATIONS = 30


### Define the training and testing functions 
# y1 and y2 are correlated 
# Adv Opt function for y1
def adv_opt_y1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, c1, c2, c3): 
    result = client.predict(
        x1, # float (numeric value between 0.0 and 1.0) in 'x1' Slider component
		x2,	# float (numeric value between 0.0 and 1.0)	in 'x2' Slider component
		x3,	# float (numeric value between 0.0 and 1.0) in 'x3' Slider component
		x4,	# float (numeric value between 0.0 and 1.0) in 'x4' Slider component
		x5,	# float (numeric value between 0.0 and 1.0) in 'x5' Slider component
		x6,	# float (numeric value between 0.0 and 1.0) in 'x6' Slider component
		x7,	# float (numeric value between 0.0 and 1.0) in 'x7' Slider component
		x8,	# float (numeric value between 0.0 and 1.0) in 'x8' Slider component
		x9,	# float (numeric value between 0.0 and 1.0) in 'x9' Slider component
		x10,	# float (numeric value between 0.0 and 1.0) in 'x10' Slider component
		x11,	# float (numeric value between 0.0 and 1.0) in 'x11' Slider component
		x12,	# float (numeric value between 0.0 and 1.0) in 'x12' Slider component
		x13,	# float (numeric value between 0.0 and 1.0) in 'x13' Slider component
		x14,	# float (numeric value between 0.0 and 1.0) in 'x14' Slider component
		x15,	# float (numeric value between 0.0 and 1.0) in 'x15' Slider component
		x16,	# float (numeric value between 0.0 and 1.0) in 'x16' Slider component
		x17,	# float (numeric value between 0.0 and 1.0) in 'x17' Slider component
		x18,	# float (numeric value between 0.0 and 1.0) in 'x18' Slider component
		x19,	# float (numeric value between 0.0 and 1.0) in 'x19' Slider component
		x20,	# float (numeric value between 0.0 and 1.0) in 'x20' Slider component
		c1,	# Literal['c1_0', 'c1_1'] in 'c1' Radio component
		c2,	# Literal['c2_0', 'c2_1'] in 'c2' Radio component
		c3,	# Literal['c3_0', 'c3_1', 'c3_2'] in 'c3' Radio component
		0.5,	# float (numeric value between 0.0 and 1.0) in 'fidelity1' Slider component
		api_name="/predict",
    )
    return result['data'][0][0]			# return y1 value only 

# Adv Opt function for y2
def adv_opt_y2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, c1, c2, c3): 
    result = client.predict(
        x1, # float (numeric value between 0.0 and 1.0) in 'x1' Slider component
		x2,	# float (numeric value between 0.0 and 1.0)	in 'x2' Slider component
		x3,	# float (numeric value between 0.0 and 1.0) in 'x3' Slider component
		x4,	# float (numeric value between 0.0 and 1.0) in 'x4' Slider component
		x5,	# float (numeric value between 0.0 and 1.0) in 'x5' Slider component
		x6,	# float (numeric value between 0.0 and 1.0) in 'x6' Slider component
		x7,	# float (numeric value between 0.0 and 1.0) in 'x7' Slider component
		x8,	# float (numeric value between 0.0 and 1.0) in 'x8' Slider component
		x9,	# float (numeric value between 0.0 and 1.0) in 'x9' Slider component
		x10,	# float (numeric value between 0.0 and 1.0) in 'x10' Slider component
		x11,	# float (numeric value between 0.0 and 1.0) in 'x11' Slider component
		x12,	# float (numeric value between 0.0 and 1.0) in 'x12' Slider component
		x13,	# float (numeric value between 0.0 and 1.0) in 'x13' Slider component
		x14,	# float (numeric value between 0.0 and 1.0) in 'x14' Slider component
		x15,	# float (numeric value between 0.0 and 1.0) in 'x15' Slider component
		x16,	# float (numeric value between 0.0 and 1.0) in 'x16' Slider component
		x17,	# float (numeric value between 0.0 and 1.0) in 'x17' Slider component
		x18,	# float (numeric value between 0.0 and 1.0) in 'x18' Slider component
		x19,	# float (numeric value between 0.0 and 1.0) in 'x19' Slider component
		x20,	# float (numeric value between 0.0 and 1.0) in 'x20' Slider component
		c1,	# Literal['c1_0', 'c1_1'] in 'c1' Radio component
		c2,	# Literal['c2_0', 'c2_1'] in 'c2' Radio component
		c3,	# Literal['c3_0', 'c3_1', 'c3_2'] in 'c3' Radio component
		0.5,	# float (numeric value between 0.0 and 1.0) in 'fidelity1' Slider component
		api_name="/predict",
    )
    return result['data'][0][1]			# return y2 value only

#%% run this cell and above for the first time to generate initial data, since we want to do benchmarking, 
# we will use the same initial data for all the runs

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

# # Create DataFrame for 20 input data size in number_init_points
# data = [generate_parameters() for _ in range(20)]
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
# lookup_testing_y1.to_csv("BayBE_lookup_testing_y1.csv", index=False)
# lookup_training_y2.to_csv("BayBE_lookup_training_y2.csv", index=False)
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
    ContinuousLinearInequalityConstraint(parameters=["x19", "x20"], coefficients=[-1.0, 1.0], rhs=0.0),
    ContinuousLinearInequalityConstraint(parameters=["x6", "x15"], coefficients=[-1.0, -1.0], rhs=-1.0), 
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
lookup_training_y2 = pd.read_csv("BayBE_lookup_training_y2.csv")
lookup_testing_y1 = pd.read_csv("BayBE_lookup_testing_y1.csv")

# Set up the optimization loop
init_size_list = [0, 1, 5, 10, 20]
random_seed_list = [23, 42, 87, 131, 518]
N_DOE_ITERATIONS = 30

for init_size in init_size_list:
    # reset the results for each init_size
	results = pd.DataFrame()
	for seed in range(len(random_seed_list)): 
		set_random_seed(random_seed_list[seed])

		# create a campaign
		campaign = Campaign(searchspace=searchspace, objective=objective)

		# set initial data
		init_df = lookup_training_y2.iloc[:init_size]       

		# add initial data to campaign
		campaign.add_measurements(init_df)

		for k in range(N_DOE_ITERATIONS):
			recommendation = campaign.recommend(batch_size=BATCH_SIZE)
			# select the numerical columns
			numerical_cols = recommendation.select_dtypes(include='number')
			# replace values less than 1e-6 with 0 in numerical columns
			numerical_cols = numerical_cols.map(lambda x: 0 if x < 1e-6 else x)
			# update the original DataFrame
			recommendation.update(numerical_cols)
			
			# if x6+x15 >1.0, round x6 and x15 to 4 decimal places
			while recommendation['x6'].item() + recommendation['x15'].item() > 1.0: 
				recommendation['x6'] = np.round(recommendation['x6'].item(), 4)
				recommendation['x15'] = np.round(recommendation['x15'].item(), 4)

			# if x19 >= x20, subtract 1e-6 from x19 and add 1e-6 to x20
			while recommendation['x19'].item() >= recommendation['x20'].item():
				recommendation['x19'] = recommendation['x19'].item() - 1e-6
				# if recommendation['x19'] < 0, assign 0 to x19
				if recommendation['x19'].item() < 0:
					recommendation['x19'] = 0
				recommendation['x20'] = recommendation['x20'].item() + 1e-6
				if recommendation['x20'].item() > 1:
					recommendation['x20'] = 1
		
			# target value are looked up via the botorch wrapper
			target_values = []
			for index, row in recommendation.iterrows():
				# print(row.to_dict())
				dict = row.to_dict()
				dict.pop('Function', None)
				target_values.append(adv_opt_y1(**dict))

			recommendation['Target'] = target_values
			time.sleep(15)
			campaign.add_measurements(recommendation)

		results = pd.concat([results, campaign.measurements])

	# create AdvOptResults folder if not exist
	if not os.path.exists('AdvOpt_Modify3_1022'):
		os.makedirs('AdvOpt_Modify3_1022')
	# save the results 
	results_testing_y1 = results[results['Function'] == 'TestingY1']
	results_testing_y1.to_csv(f"AdvOpt_Modify3_1022/init_{init_size}_BayBE_5round.csv", index=False)
		