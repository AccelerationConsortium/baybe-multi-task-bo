# -*-coding:utf-8 -*-
"""
@Time    :   2024/07/14 11:17:24
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :
"""
# %%
# IMPORT DEPENDENCIES------------------------------------------------------------------------------
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import numpy.matlib as nm
import pandas as pd
from pyro import param
import seaborn as sns
from baybe import Campaign
from baybe.constraints import DiscreteSumConstraint, ThresholdCondition
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.searchspace import SubspaceDiscrete
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper
from baybe.utils.plotting import create_example_plots
from botorch.test_functions.synthetic import Hartmann
from baybe.recommenders import RandomRecommender


#%%
# LOAD DATA----------------------------------------------------------------------------------------

# get the path to the directory before the current directory
strHomeDir = os.path.dirname(os.getcwd())

dfMP = pd.read_csv(
    os.path.join(strHomeDir, "data", "processed", "mp_bulkModulus_goodOverlap.csv"), index_col=0
)

# add a column to dfMP called 'load' and set all values to 1
dfMP["load"] = 1

dfExp = pd.read_csv(
    os.path.join(strHomeDir, "data", "processed", "exp_hardness_goodOverlap.csv"), index_col=0
)

lstElementCols = dfExp.columns.to_list()[4:]

# make a list of the unique compositions that are shared between dfMP and dfExp
lstSharedCompositions = list(set(dfMP["composition"]).intersection(set(dfExp["composition"])))

#%%
# MAKE A HISTOGRAM OF THE HARDNESS VALUE ----------------------------------------------------------

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='w', edgecolor='k', dpi = 300, constrained_layout = True)

# plot a histogram of the hardness values
ax.hist(dfExp["hardness"], bins=20)

# add a title, x-aixs label, and y-axis label
ax.set_xlabel("Hardness")
ax.set_ylabel("Frequency")
ax.set_title("Hardness Distribution")

#%%
# FIND THE DUPLICATE COMPOSITIONS------------------------------------------------------------------

# for each unique composition in dfExp, count the number of times it appears
srCompositionCounts = dfExp["composition"].value_counts()

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='w', edgecolor='k', dpi = 300, constrained_layout = True)

# plot a histogram of srCompositionCounts where the x-axis is the number of times a composition appears and the y-axis is the frequency
lstHeight, lstBins, contPatches = ax.hist(srCompositionCounts, bins=6)

# calculate the center of each bin
lstBinCenters = 0.5 * (lstBins[:-1] + lstBins[1:])
# set the x-ticks to the bin centers
ax.set_xticks(lstBinCenters)
# create custom labels based on the bin ranges or integer values
# for instance, using the left edge of each bin as a label
lstTickLabels = [f"{int(lstBins[i+1])}" for i in range(len(lstBins)-1)]
# set the custom tick labels
ax.set_xticklabels(lstTickLabels)


# add a title, x-aixs label, and y-axis label
ax.set_title("Composition Frequency")
ax.set_xlabel("Number of Load Values per Unique Composition")
ax.set_ylabel("Frequency")

# add a grid
ax.grid(True)

#%%
# EXAMPLE CURVE FOR A COMPOSITION------------------------------------------------------------------

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='w', edgecolor='k', dpi = 300, constrained_layout = True)


# plot the hardness vs load for entries with composition Y2 Re0.95 Cr0.05 B6
ax.scatter(dfExp.loc[dfExp["composition"] == "Y2 Re0.95 Cr0.05 B6"]["load"],
           dfExp.loc[dfExp["composition"] == "Y2 Re0.95 Cr0.05 B6"]["hardness"])

# make a cubic spline interpolation of the hardness vs load curve
spSpline = sp.interpolate.CubicSpline(dfExp.loc[dfExp["composition"] == "Y2 Re0.95 Cr0.05 B6"]["load"],
                                      dfExp.loc[dfExp["composition"] == "Y2 Re0.95 Cr0.05 B6"]["hardness"])

# make a range of values from 0.5 to 5 with steps of 0.1
lstX = np.arange(0.5, 5, 0.1)
# evaluate the spline at the values in lstX
lstY = spSpline(lstX)

# plot the spline
ax.plot(lstX, lstY, color='r')

# add a title, x-aixs label, and y-axis label
ax.set_title("Hardness vs Load for Y2 Re0.95 Cr0.05 B6")
ax.set_xlabel("Load")
ax.set_ylabel("Hardness")


# add a grid
ax.grid(True)
# add minor ticks
ax.minorticks_on()

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

# extract the strComposition column from dfExp_integratedHardness
dfExp_integratedHardness["strComposition"].to_csv(os.path.join(strHomeDir, 'data', 'processed', 'strComposition.csv'))
    
#%%
# MAKE A HISTOGRAM OF THE INTEGRATED HARDNESS VALUE ------------------------------------------------

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='w', edgecolor='k', dpi = 300, constrained_layout = True)

# plot a histogram of the hardness values
ax.hist(dfExp_integratedHardness["integratedHardness"], bins=20)

# add a title, x-aixs label, and y-axis label
ax.set_xlabel("Integrated Hardness")
ax.set_ylabel("Frequency")
ax.set_title("Integrated Hardness Distribution")

#%%
# CORRELATION BETWEEN INTEGRATED HARDNESS AND VOIGT BULK MODULUS-----------------------------------

# pull the
dfExp_integratedHardness_shared = dfExp_integratedHardness[dfExp_integratedHardness["composition"].isin(lstSharedCompositions)]
# add the voigt bulk modulus values from dfMP to dfExp_integratedHardness_shared
dfExp_integratedHardness_shared = pd.merge(dfExp_integratedHardness_shared, dfMP, on="composition")
# make a temporary dataframe with only strComposition_x and vrh columns
dfExp_shared = dfExp_integratedHardness_shared[['strComposition_x', 'vrh']]
# rename the columns to 'strComposition' and 'vrh'
dfExp_shared = dfExp_shared.rename(columns={'strComposition_x': 'strComposition', 'vrh': 'vrh'})

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='w', edgecolor='k', dpi = 300, constrained_layout = True)
# plot a scatter plot of the integrated hardness vs voigt bulk modulus
ax.scatter(dfExp_integratedHardness_shared["integratedHardness"], dfExp_integratedHardness_shared["vrh"])
# add a title, x-aixs label, and y-axis label
ax.set_xlabel("Integrated Hardness")
ax.set_ylabel("Voigt Bulk Modulus")
ax.set_title("Integrated Hardness vs Voigt Bulk Modulus")

# calculate the correlation between the integrated hardness and voigt bulk modulus
fltCorrelation = dfExp_integratedHardness_shared["integratedHardness"].corr(dfExp_integratedHardness_shared["vrh"])
# add the correlation to the plot
ax.text(0.1, 0.9, f"Correlation: {fltCorrelation:.2f}", transform=ax.transAxes)

# calculate the r2 value
fltR2 = fltCorrelation**2
# add the r2 value to the plot
ax.text(0.1, 0.8, f"R2: {fltR2:.2f}", transform=ax.transAxes)

# add a grid
ax.grid(True)
# add minor grid lines
ax.minorticks_on()

#%%
# CORRELATION BETWEEN INTEGRATED HARDNESS AND VOIGT BULK MODULUS (predicted) ----------------------

# pull the predicted bulk modulus values from 'Finder' (https://github.com/ihalage/Finder)
dfExp_integratedHardness_bulkModulus_pred = pd.read_csv(
    os.path.join(strHomeDir, "data", "processed", "exp_integratedHardness_bulkModulusPredictions.csv")
)

# for entries in dfExp_integratedHardness, pull the predicted bulk modulus value from dfExp_integratedHardness_bulkModulus_pred
dfExp_integratedHardness_wPred = pd.merge(dfExp_integratedHardness, dfExp_integratedHardness_bulkModulus_pred, on="strComposition")

# make another column called 'vrh_pred' and set all values to the 10^[predicted value]
dfExp_integratedHardness_wPred["vrh_pred"] = 10**dfExp_integratedHardness_wPred["prediction"]

# add the vrh predictions to dfExp_shared
dfExp_shared = pd.merge(dfExp_shared, dfExp_integratedHardness_wPred[["strComposition", "vrh_pred"]], on="strComposition")

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='w', edgecolor='k', dpi = 300, constrained_layout = True)
# plot a scatter plot of the integrated hardness vs voigt bulk modulus
ax.scatter(dfExp_integratedHardness_wPred["integratedHardness"], dfExp_integratedHardness_wPred["vrh_pred"])
# add a title, x-aixs label, and y-axis label
ax.set_xlabel("Integrated Hardness")
ax.set_ylabel("Voigt Bulk Modulus (Predicted)")
ax.set_title("Integrated Hardness vs Voigt Bulk Modulus (Predicted)")

# calculate the correlation between the integrated hardness and voigt bulk modulus
fltCorrelation = dfExp_integratedHardness_wPred["integratedHardness"].corr(dfExp_integratedHardness_wPred["vrh_pred"])
# add the correlation to the plot
ax.text(0.1, 0.9, f"Pearson Correlation: {fltCorrelation:.2f}", transform=ax.transAxes)

# calculate the r2 value
fltR2 = fltCorrelation**2
# add the r2 value to the plot
ax.text(0.1, 0.8, f"R2: {fltR2:.2f}", transform=ax.transAxes)

# add a grid
ax.grid(True)
# add minor grid lines
ax.minorticks_on()




#%%
# CLEAN DATA---------------------------------------------------------------------------------------

# make a dataframe for the task function (hardness) - dfExp [element columns, load]
dfSearchSpace_target = dfExp_integratedHardness[lstElementCols]
# add a column to dfSearchSpace_task called 'Function' and set all values to 'taskFunction'
dfSearchSpace_target["Function"] = "targetFunction"

# make a lookup table for the task function (hardness) - add the 'hardness' column from dfExp to dfSearchSpace_task
dfLookupTable_target = pd.concat([dfSearchSpace_target, dfExp_integratedHardness["integratedHardness"]], axis=1)#/load"]], axis=1)
# make the 'hardness' column the 'Target' column
dfLookupTable_target = dfLookupTable_target.rename(columns={"integratedHardness":"Target"})#/load": "Target"})

# make a dataframe for the source function (voigt bulk modulus) - dfMP [element columns, load]
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
# GENERATE THE SEARCH SPACE------------------------------------------------------------------------

lstParameters = []

# for each column in dfSearchSpace except the last one, create a NumericalDiscreteParameter
for strCol_temp in dfSearchSpace.columns[:-1]:
    # create a NumericalDiscreteParameter
    parameter = NumericalDiscreteParameter(
        name=strCol_temp,
        values=np.unique(dfSearchSpace[strCol_temp]),
        tolerance=0.0,
    )
    # append the parameter to the list of parameters
    lstParameters.append(parameter)

# create a TaskParameter
taskParameter = TaskParameter(
    name="Function",
    values=["targetFunction", "sourceFunction"],
    active_values=["targetFunction"],
)

# append the taskParameter to the list of parameters
lstParameters.append(taskParameter)

# create the search space
searchSpace = SearchSpace.from_dataframe(dfSearchSpace, parameters=lstParameters)

#%%
# CREATE OPTIMIZATION OBJECTIVE--------------------------------------------------------------------
"""
the test functions have a single output (voigt bulk modulus and hardness) that is to be maximized
"""

objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MAX"))

#%%
# SIMULATE SCENARIOS-------------------------------------------------------------------------------
'''cenarios
# def simulateScenario_manual():
#     # add intial data
use the simulate_scenarios function to simulate the optimization process
'''

# # manually simulate the s

#     # in a loop
#         # recomend
#         # add measurement
#     pass

N_MC_ITERATIONS = 10
N_DOE_ITERATIONS = 30
BATCH_SIZE = 1

results: list[pd.DataFrame] = []
for n in (5, 15, 30):
    campaign = Campaign(searchspace=searchSpace, objective=objective)
    initial_data = [dfLookupTable_source.sample(n) for _ in range(N_MC_ITERATIONS)] # frac = p
    result_fraction = simulate_scenarios(
        {f"{n}": campaign},                                                # int(100*p)
        dfLookupTable_target,
        initial_data=initial_data,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
    )
    results.append(result_fraction)

# For comparison, we also optimize the function without using any initial data:

result_baseline = simulate_scenarios(
    {"0": Campaign(searchspace=searchSpace, objective=objective)},
    dfLookupTable_target,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# !!!!! ADD A 0 WITHOUT TASK PARAMETER (NOT MULTI-TASK LEARNING)

results_random = simulate_scenarios(
    {"random": Campaign(searchspace=searchSpace, objective=objective, recommender=RandomRecommender())},
    dfLookupTable_target,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

results = pd.concat([results_random, result_baseline, *results])

# build a function for randomly sampling and 
# All that remains is to visualize the results.
# As the example shows, the optimization speed can be significantly increased by
# using even small amounts of training data from related optimization tasks.

results.rename(columns={"Scenario": "Number of data used"}, inplace=True)
# save the results to a dataframe
results.to_csv(os.path.join(strHomeDir, 'reports', 'results_hardnessOnly.csv'))



#%%
# PLOT RESUTLS-------------------------------------------------------------------------------------

# import results
results = pd.read_csv(os.path.join(strHomeDir, 'reports', 'results_hardnessOnly.csv'),  index_col=0)

# intialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# plot the results
ax = sns.lineplot(
    data=results,
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
'''
TODO LIST
---------------

1. pull out the simulate_scenarios function and make it manually
'''