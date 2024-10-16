# -*-coding:utf-8 -*-
"""
@Time    :   2024/07/14 11:17:24
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   this script is used as a benchmarking tool for the bayesian optimization packages
"""
# %%
# IMPORT DEPENDENCIES------------------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

import seaborn as sns
from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.recommenders import RandomRecommender

from ax.core.observation import ObservationFeatures
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.task_encode import TaskEncode
from ax.modelbridge.transforms.unit_x import UnitX
from ax.service.ax_client import AxClient, ObjectiveProperties

# Function to set seeds for reproducibility
def setSeeds(seed=42):
    '''
    set the seed for reproducibility

    Parameters
    ----------
    seed : int, optional
        the seed to set, by default 42
    '''
    np.random.seed(seed)

setSeeds()

#%%
# LOAD DATA----------------------------------------------------------------------------------------

# get the path to the directory before the current directory
strHomeDir = os.path.dirname(os.getcwd())

dfMP = pd.read_csv(
    os.path.join(strHomeDir, "data", "processed", "mp_bulkModulus_goodOverlap.csv"), index_col=0
)

dfExp = pd.read_csv(
    os.path.join(strHomeDir, "data", "processed", "exp_hardness_goodOverlap.csv"), index_col=0
)

lstElementCols = dfExp.columns.to_list()[4:]

#%%
# FUTHER CLEAN THE DATA BASED ON THE EDA-----------------------------------------------------------

# initialize an empty dataframe to store the integrated hardness values
dfExp_integratedHardness = pd.DataFrame()

# for each unique composition in dfExp, make a cubic spline interpolation of the hardness vs load curve
for strComposition_temp in dfExp["composition"].unique():
    # get the data for the composition
    dfComposition_temp = dfExp[dfExp["composition"] == strComposition_temp]
    # sort the data by load
    dfComposition_temp = dfComposition_temp.sort_values(by="load")
    # if there are any duplicate values for load, drop them
    dfComposition_temp = dfComposition_temp.drop_duplicates(subset="load")
    # if there are less than 5 values, continue to the next composition
    if len(dfComposition_temp) < 5:
        continue

    # make a cubic spline interpolation of the hardness vs load curve
    spSpline_temp = sp.interpolate.CubicSpline(dfComposition_temp["load"], dfComposition_temp["hardness"])
    # integrate the spline from the minimum load to the maximum load
    fltIntegral_temp = spSpline_temp.integrate(0.5, 5, extrapolate = True)

    # make a new dataframe with the lstElementCols from dfComposition_temp
    dfComposition_temp = dfComposition_temp[['strComposition', 'composition'] + lstElementCols]
    dfComposition_temp = dfComposition_temp.drop_duplicates(subset='composition')
    # add a column to dfComposition_temp called 'integratedHardness' and set all values to fltIntegral_temp
    dfComposition_temp["integratedHardness"] = fltIntegral_temp

    # append dfComposition_temp to dfExp_integratedHardness
    dfExp_integratedHardness = pd.concat([dfExp_integratedHardness, dfComposition_temp])

#%%
# CREATE DATAFRAMES FOR THE SEARCH SPACE-----------------------------------------------------------

# ----- TARGET FUNCTION (INTEGRATED HARDNESS) -----
# make a dataframe for the task function (integrated hardness)
dfSearchSpace_target = dfExp_integratedHardness[lstElementCols]
# add a column to dfSearchSpace_task called 'Function' and set all values to 'taskFunction'
dfSearchSpace_target["Function"] = "targetFunction"

# make a lookup table for the task function (integrate hardness) - add the 'integratedHardness' column from dfExp to dfSearchSpace_task
dfLookupTable_target = pd.concat([dfSearchSpace_target, dfExp_integratedHardness["integratedHardness"]], axis=1)
# make the 'integrate hardness' column the 'Target' column
dfLookupTable_target = dfLookupTable_target.rename(columns={"integratedHardness":"Target"})

# ----- SOURCE FUNCTION (VOIGT BULK MODULUS) -----
# make a dataframe for the source function (voigt bulk modulus)
dfSearchSpace_source = dfMP[lstElementCols]
# add a column to dfSearchSpace_source called 'Function' and set all values to 'sourceFunction'
dfSearchSpace_source["Function"] = "sourceFunction"

# make a lookup table for the source function (voigt bulk modulus) - add the 'vrh' column from dfMP to dfSearchSpace_source
dfLookupTable_source = pd.concat([dfSearchSpace_source, dfMP["vrh"]], axis=1)
# make the 'vrh' column the 'Target' column
dfLookupTable_source = dfLookupTable_source.rename(columns={"vrh": "Target"})

# concatenate the two dataframes
dfSearchSpace = pd.concat([dfSearchSpace_target, dfSearchSpace_source])

#%%
# BAYBE EXAMPLE------------------------------------------------------------------------------------
# ----- GENERATE THE SEARCH SPACE -----

lstParameters_bb = []

# for each column in dfSearchSpace except the last one, create a NumericalDiscreteParameter
for strCol_temp in dfSearchSpace.columns[:-1]:
    # create a NumericalDiscreteParameter
    parameter = NumericalDiscreteParameter(
        name=strCol_temp,
        values=np.unique(dfSearchSpace[strCol_temp]),
        tolerance=0.0,
    )
    # append the parameter to the list of parameters
    lstParameters_bb.append(parameter)

# create a TaskParameter
taskParameter = TaskParameter(
    name="Function",
    values=["targetFunction", "sourceFunction"],
    active_values=["targetFunction"],
)

# append the taskParameter to the list of parameters
lstParameters_bb.append(taskParameter)

# create the search space
searchSpace = SearchSpace.from_dataframe(dfSearchSpace, parameters=lstParameters_bb)

# ----- CREATE OPTIMIZATION OBJECTIVE -----

objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MAX"))

# ----- SIMULATE SCENARIOS -----
'''cenarios
# def simulateScenario_manual():
#     # add intial data
use the simulate_scenarios function to simulate the optimization process
'''
intMCIterations = 5
intDOEIterations = 20
intBatchSize = 1

# initialize a list to store the results
lstResults_bb: list[pd.DataFrame] = []

# for each value of n in (5, 15, 30)
for n in (2, 4, 6, 30):
    campaign_temp = Campaign(searchspace=searchSpace, objective=objective)
    lstInitialData_temp = [dfLookupTable_source.sample(n) for _ in range(intMCIterations)] # frac = p
    result_fraction = simulate_scenarios(
        {f"{n}": campaign_temp},                                                # int(100*p)
        dfLookupTable_target,
        initial_data=lstInitialData_temp,
        batch_size=intBatchSize,
        n_doe_iterations=intDOEIterations,
    )
    lstResults_bb.append(result_fraction)

# For comparison, we also optimize the function without using any initial data:

result_baseline = simulate_scenarios(
    {"0": Campaign(searchspace=searchSpace, objective=objective)},
    dfLookupTable_target,
    batch_size=intBatchSize,
    n_doe_iterations=intDOEIterations,
    n_mc_iterations=intMCIterations,
)

# !!!!! ADD A 0 WITHOUT TASK PARAMETER (NOT MULTI-TASK LEARNING)

results_random = simulate_scenarios(
    {"random": Campaign(searchspace=searchSpace, objective=objective, recommender=RandomRecommender())},
    dfLookupTable_target,
    batch_size=intBatchSize,
    n_doe_iterations=intDOEIterations,
    n_mc_iterations=intMCIterations,
)

dfResults_bb = pd.concat([results_random, result_baseline, *lstResults_bb])

# build a function for randomly sampling and 
# All that remains is to visualize the results.
# As the example shows, the optimization speed can be significantly increased by
# using even small amounts of training data from related optimization tasks.

dfResults_bb.rename(columns={"Scenario": "Number of data used"}, inplace=True)
# save the results to a dataframe
dfResults_bb.to_csv(os.path.join(strHomeDir, 'reports', 'results_integratedHardness_final.csv'))

# ----- PLOT RESUTLS -----

# import results
dfResults_bb = pd.read_csv(os.path.join(strHomeDir, 'reports', 'results_integratedHardness_final.csv'),  index_col=0)

# intialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# plot the results
ax = sns.lineplot(
    data=dfResults_bb,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="Number of data used",
)

# add a line at the maximum value
plt.axhline(y=dfLookupTable_target['Target'].max(), color='r', linestyle='--', label='Max Value')

# add a legend
plt.legend()

# add a title
plt.title("Multi-Task Learning Optimization")

# add a x-axis label
plt.xlabel("Number of Experiments")

# add a y-axis label
plt.ylabel("Target Cumulative Best - integratedHardness")

# save the figure
plt.savefig(os.path.join(strHomeDir, 'reports', 'figures', 'multiTaskLearning_hardness.png'))


#%%
# AX EXAMPLE----------------------------------------------------------------------------------------
transforms = [TaskEncode, UnitX]

# ----- INITIALIZE AX CLIENT -----

gs = GenerationStrategy(
    name="MultiTaskOp",
    steps=[
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            max_parallelism=10,
            model_kwargs={"transforms": transforms},
        ),
    ],
)

# make a list of dictionaries for the parameters (each dictionary is one of the elements in the list)
lstParameters_ax = [
    {
        "name": strCol_temp,
        "type": "range",
        "value_type": "float",
        "bounds": [np.unique(dfSearchSpace[strCol_temp]).min(), np.unique(dfSearchSpace[strCol_temp]).max()],
    }
    for strCol_temp in dfSearchSpace.columns[:-1]
]

# add the task parameter
lstParameters_ax.append(
    {
        "name": "Function",
        "type": "choice",
        "values": ["targetFunction", "sourceFunction"],
        "is_task": True,
        "target_value": "targetFunction",
    }
)

strSeparator = " + "
strCompositionConstraint = strSeparator.join(lstElementCols[:-1]) + " <= 1.0"


# ----- SIMULATE SCENARIOS -----

# add the target data
def preTrainWTarget(n: int = 1):
    '''
    randomly select n data points from dfLookupTable_target and add them to the experiment

    Parameters
    ----------

    n : int
        the number of data points to add to the experiment
    '''

    # randomly select n data points from dfLookupTable_target
    dfTarget_temp = dfLookupTable_target.sample(n)
    
    for _, row in dfTarget_temp.iterrows():
        # get the row as a dictionary
        dictRow = row.to_dict()
        # drop the 'Target' key from the dictionary
        dictRow.pop("Target") 
        # add the row to the experiment
        _, trial_index = ax_client.attach_trial(parameters=dictRow)
        # complete the trial
        ax_client.complete_trial(trial_index=trial_index, raw_data={"Objective": row["Target"]})

# create a function to add n data to the experiment
def preTrainWSource(n: int = 5):
    '''
    randomly select n data points from dfLookupTable_source and add them to the experiment

    Parameters
    ----------
    n : int
        the number of data points to add to the experiment
    '''
    # randomly select n data points from dfLookupTable_source
    dfSource_temp = dfLookupTable_source.sample(n)
    
    for _, row in dfSource_temp.iterrows():
        # get the row as a dictionary
        dictRow = row.to_dict()
        # drop the 'Target' key from the dictionary
        dictRow.pop("Target") 
        # add the row to the experiment
        _, trial_index = ax_client.attach_trial(parameters=dictRow)
        # complete the trial
        ax_client.complete_trial(trial_index=trial_index, raw_data={"Objective": row["Target"]})


intDOEIterations = 20

# initialize a dataframe to store the results
dfResults_ax = pd.DataFrame()

# for each value of n in (5, 15, 30)
for n_preTrain in (2, 4, 6):

    # generate a list of 10 random integers between 0 and 100000 
    lstMCIterations = [1337, 1338, 2112, 1233, 99987]
    intMCRun = 0
    # for each value of n in lstMCIterations
    for n_MC in lstMCIterations:
        setSeeds(n_MC)
        
        ax_client = AxClient(generation_strategy=gs, random_seed=n_MC, verbose_logging=False)

        # create the experiment
        ax_client.create_experiment(
            name="MultiTaskOp",
            parameters=lstParameters_ax,
            #parameter_constraints=[strCompositionConstraint],
            objectives={"Objective": ObjectiveProperties(minimize=False)},
        )



        # pre-train the experiment with the source data
        preTrainWSource(n_preTrain)

        # add a measurement for the target data
        preTrainWTarget(2)

        for i in range(intDOEIterations):

            ax_client.fit_model()
            model = ax_client.generation_strategy.model

            obs_features = [
                ObservationFeatures(row.to_dict())
                for _, row in dfSearchSpace_target.iterrows()
            ]

            acqf_values = np.array(
                model.evaluate_acquisition_function(observation_features=obs_features)
            )

            best_index = np.argmax(acqf_values)
            best_parameters = dfSearchSpace_target.iloc[best_index].to_dict()

            # Extract the element columns from best_parameters
            best_elements = {k: v for k, v in best_parameters.items() if k in lstElementCols}

            # Find the row in dfLookupTable_target that matches the best parameters
            matching_row = dfLookupTable_target[
                (dfLookupTable_target[lstElementCols] == pd.Series(best_elements)).all(axis=1)
            ]

            # Extract the target value (hardness or whatever target you're optimizing for)
            result = matching_row["Target"].values[0] if not matching_row.empty else None

            best_feat = dfSearchSpace_target.iloc[best_index].to_dict()
            _, trial_index = ax_client.attach_trial(parameters=best_feat)
            ax_client.complete_trial(trial_index=trial_index, raw_data={"Objective": result})

        # initialize an integer to store the number of trials used to start the experiment
        intNumTrials = n_preTrain + 2

        dfResults_temp = ax_client.get_trials_data_frame()

        # drop the first n_preTrain trials from the dataframe
        dfResults_temp.drop(range(intNumTrials), inplace=True)

        # add a another column to the dataframe called 'Num_Experiments' and set all values to trial_index - intNumTrials
        dfResults_temp["Num_Experiments"] = dfResults_temp["trial_index"] - intNumTrials + 1

        # add a column to the dataframe called 'Num_MC' and set all values to n_MC
        dfResults_temp["Monte_Carlo_Run"] = intMCRun

        # add a column called 'Number of data used' and set all values to n_preTrain
        dfResults_temp["Number of data used"] = n_preTrain

        # add a column called 'Target_CumBest' and set all values to the cumulative maximum of the 'Objective' column for the entries before the current entry
        dfResults_temp["Target_CumBest"] = dfResults_temp["Objective"].cummax()

        # append dfResults_temp to dfResults_ax
        dfResults_ax = pd.concat([dfResults_ax, dfResults_temp])

        intMCRun += 1

        
# #     # for i in range(40):


# # # ----- RUN THE EXPERIMENT -----
# ax_client = AxClient(generation_strategy=gs, random_seed=30, verbose_logging=False)
# ax_client.create_experiment(
#             name="MultiTaskOp",
#             parameters=lstParameters_ax,
#             #parameter_constraints=[strCompositionConstraint],
#             objectives={"Objective": ObjectiveProperties(minimize=False)},
#         )

# preTrainWSource(10)



# preTrainWTarget(1)

# for i in range(40):

#     ax_client.fit_model()
#     model = ax_client.generation_strategy.model

#     obs_features = [
#         ObservationFeatures(row.to_dict())
#         for _, row in dfSearchSpace_target.iterrows()
#     ]

#     acqf_values = np.array(
#         model.evaluate_acquisition_function(observation_features=obs_features)
#     )

#     best_index = np.argmax(acqf_values)
#     best_parameters = dfSearchSpace_target.iloc[best_index].to_dict()

#     # Extract the element columns from best_parameters
#     best_elements = {k: v for k, v in best_parameters.items() if k in lstElementCols}

#     # Find the row in dfLookupTable_target that matches the best parameters
#     matching_row = dfLookupTable_target[
#         (dfLookupTable_target[lstElementCols] == pd.Series(best_elements)).all(axis=1)
#     ]

#     # Extract the target value (hardness or whatever target you're optimizing for)
#     result = matching_row["Target"].values[0] if not matching_row.empty else None

#     best_feat = dfSearchSpace_target.iloc[best_index].to_dict()
#     _, trial_index = ax_client.attach_trial(parameters=best_feat)
#     ax_client.complete_trial(trial_index=trial_index, raw_data={"Objective": result})


# ax_client.get_trials_data_frame()

# # # %%

# # %%

# %%
# intialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# plot the results
ax = sns.lineplot(
    data=dfResults_ax,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="Number of data used",
)

# add a line at the maximum value
plt.axhline(y=dfLookupTable_target['Target'].max(), color='r', linestyle='--', label='Max Value')

# add a legend
plt.legend()

# add a title
plt.title("Multi-Task Learning Optimization")

# add a x-axis label
plt.xlabel("Number of Experiments")

# add a y-axis label
plt.ylabel("Target Cumulative Best - integratedHardness")

# save the figure
plt.savefig(os.path.join(strHomeDir, 'reports', 'figures', 'multiTaskLearning_hardness_ax.png'))
#%%

# add _bb to the end of the Number of data used column of dfResults_bb
dfResults_bb["Number of data used"] = dfResults_bb["Number of data used"].astype(str) + "_bb"
# add _ax to the end of the Number of data used column of dfResults_ax
dfResults_ax["Number of data used"] = dfResults_ax["Number of data used"].astype(str) + "_ax"

# concatenate dfResults_bb and dfResults_ax
dfResults_combined = pd.concat([dfResults_bb, dfResults_ax])
# %%
# intialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# plot the results
ax = sns.lineplot(
    data=dfResults_combined,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="Number of data used",
)

# add a line at the maximum value
plt.axhline(y=dfLookupTable_target['Target'].max(), color='r', linestyle='--', label='Max Value')

# add a legend
plt.legend()

# add a title
plt.title("Multi-Task Learning Optimization")

# add a x-axis label
plt.xlabel("Number of Experiments")

# add a y-axis label
plt.ylabel("Target Cumulative Best - integratedHardness")
# %%
