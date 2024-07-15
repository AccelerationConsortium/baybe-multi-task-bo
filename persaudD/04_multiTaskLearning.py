# -*-coding:utf-8 -*-
'''
@Time    :   2024/07/14 11:17:24
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   
'''

#%%
# IMPORT DEPENDENCIES------------------------------------------------------------------------------
import os
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from botorch.test_functions.synthetic import Hartmann

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.constraints import DiscreteSumConstraint, ThresholdCondition
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper
from baybe.utils.plotting import create_example_plots

# import linear regression model
from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

import ternary
from persaudD.persaudD_general import pltTernary
import numpy.matlib as nm

# SET UP LOGGING-----------------------------------------------------------------------------------

# get the path to the current directory
strWD = os.getcwd()
# get the name of this file
strLogFileName = os.path.basename(__file__)
# split the file name and the extension
strLogFileName = os.path.splitext(strLogFileName)[0]
# add .log to the file name
strLogFileName = os.path.join(f'{strLogFileName}.log')
# join the log file name to the current directory
strLogFilePath = os.path.join(strWD, strLogFileName)

# Initialize logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(strLogFilePath, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)

#%%
# LOAD DATA----------------------------------------------------------------------------------------

# load combinationCount
dfCombinationCounts = pd.read_csv(os.path.join(strWD, "data", "combinationCounts_3.csv"), index_col=0)

# # pull the 5 rows with the highest count_exp
# dfCombinationCounts = dfCombinationCounts.sort_values(by="count_exp", ascending=False).head(5)

intCount_exp = 0 
intCount_mp = 0

# for every row in the dataframe, find the number of valid rows
for idx, row in dfCombinationCounts.iterrows():
    intCount_mp_temp = row["count_mp"]
    intCount_exp_temp = row["count_exp"]

    
    if intCount_mp_temp > intCount_mp and intCount_exp_temp > intCount_exp:
        intCount_mp = intCount_mp_temp
        intCount_exp = intCount_exp_temp
        strId_bestCombiniation = idx

        # remove square brackets and qoutes from row["indices_mp"]
        lstId_mp = row["indices_mp"].replace("[", "").replace("]", "").replace("'", "").split(", ")
        lstId_exp = row["indices_exp"].replace("[", "").replace("]", "").replace("'", "").split(", ")
        lstId_exp = [int(strId) for strId in lstId_exp]
        


dfMP = pd.read_csv(os.path.join(strWD, "data", "mp_bulkModulus_wElementFractions.csv"), index_col=0)
# drop the last column
dfMP = dfMP.iloc[:, :-1]
logging.info("Loaded bulk modulus data from csv file")
# get a list of the elements (from the columns) in the dataframes
lstElementCols = dfMP.columns.tolist()[5:]

dfExp = pd.read_csv(os.path.join(strWD, "data", "exp_hardness_wElementFractions.csv"), index_col=0)
# drop the last column
dfExp = dfExp.iloc[:, :-1]
logging.info("Loaded experimental hardness data from csv file")

# downselect the dataframes to only include the best combination
dfMP = dfMP.loc[lstId_mp]
dfExp = dfExp.loc[lstId_exp]

# -----MP-----
# initialize a list to store the columns with all zeros
lstZeroCols_mp = []
# initialize a list to store the columns with non-zero values
lstNonZeroCols_mp = []

for strCol_temp in lstElementCols:
    if dfMP[strCol_temp].sum() == 0:
        lstZeroCols_mp.append(strCol_temp)
    else:
        lstNonZeroCols_mp.append(strCol_temp)

# drop the columns with all zeros
dfMP.drop(columns=lstZeroCols_mp, inplace=True)

# -----EXP-----
# initialize a list to store the columns with all zeros
lstZeroCols_exp = []
# initialize a list to store the columns with non-zero values
lstNonZeroCols_exp = []
for strCol_temp in lstElementCols:
    if dfExp[strCol_temp].sum() == 0:
        lstZeroCols_exp.append(strCol_temp)
    else:
        lstNonZeroCols_exp.append(strCol_temp)

# drop the columns with all zeros
dfExp.drop(columns=lstZeroCols_exp, inplace=True)

#%%
# PLOT THE DATA-------------------------------------------------------------------------------------

# -----MP-----

pltTernary(dfCompositionData = dfMP[lstNonZeroCols_mp]*100,
           lstElements = lstNonZeroCols_mp,
           srColor = dfMP["voigt"],
           strColorBarLabel = "Voigt Bulk Modulus",
           strTitle = "MP Data",
           intMarkerSize = 100)

# -----EXP-----
pltTernary(dfCompositionData = dfExp[lstNonZeroCols_exp]*100,
              lstElements = lstNonZeroCols_exp,
              srColor = dfExp["hardness"],
              strColorBarLabel = "Hardness",
              strTitle = "EXP Data",
              intMarkerSize = 100)


#%%
# TRAIN A GAUSSIAN PROCESS REGRESSOR---------------------------------------------------------------
'''
the Gaussian Process Regressor is trained on the MP data to predict the voigt bulk modulus and
another one is trained on the Exp data to predict the hardness. Each model will then be used to 
predict the the same values for a more filled in grid of the elements. These predictions will be
used to generate the lookup tables for the test functions. (ie. the test functions will be the
ground truth values of the voigt bulk modulus and hardness)
'''

# make a grid of test points (0 to 1 with steps of 0.02 for each element in lstNonZeroCols_exp)
lstTestPoints = np.meshgrid(*[np.arange(0, 1.02, 0.02) for _ in lstNonZeroCols_exp])
# make a dataframe of the test points
dfX_test_mp = pd.DataFrame({strElement: lstTestPoints[idx].ravel() for idx, strElement in enumerate(lstNonZeroCols_exp)})
# round the values to 2 decimal places
dfX_test_mp = dfX_test_mp.round(2)
# remove rows where the sum of the row is not equal to 1
dfX_test_mp = dfX_test_mp[np.isclose(dfX_test_mp.sum(axis=1), 1, atol=1e-3)]
# sort dfX_test_mp by the columns
dfX_test_mp = dfX_test_mp.sort_values(by=lstNonZeroCols_exp)
# reset the index
dfX_test_mp.reset_index(drop=True, inplace=True)

# make a dataframe of the test points
dfX_test_exp = pd.DataFrame({strElement: lstTestPoints[idx].ravel() for idx, strElement in enumerate(lstNonZeroCols_exp)})
# round the values to 2 decimal places
dfX_test_exp = dfX_test_exp.round(2)
# remove rows where the sum of the row is not equal to 1
dfX_test_exp = dfX_test_exp[np.isclose(dfX_test_exp.sum(axis=1), 1, atol=1e-3)]
# sort dfX_test_exp by the columns
dfX_test_exp = dfX_test_exp.sort_values(by=lstNonZeroCols_exp)
# reset the index
dfX_test_exp.reset_index(drop=True, inplace=True)


# train the Gaussian Process Regressor on the MP data

gp_mp = LinearRegression()
gp_mp.fit(dfMP[lstNonZeroCols_exp], dfMP["voigt"])

# train the Gaussian Process Regressor on the EXP data
gp_exp = LinearRegression()
gp_exp.fit(dfExp[lstNonZeroCols_exp], dfExp["hardness"])

# predict the voigt bulk modulus and make it a series
srY_test_mp = pd.Series(gp_mp.predict(dfX_test_mp), name="voigt")
# predict the hardness and make it a series
srY_test_exp = pd.Series(gp_exp.predict(dfX_test_exp), name="hardness")

# -----PLOT THE PREDICTIONS-----

# -----MP-----
pltTernary(dfCompositionData = dfX_test_mp*100,
              lstElements = lstNonZeroCols_exp,
                srColor = srY_test_mp,
                strColorBarLabel = "Voigt Bulk Modulus",
                strTitle = "LR Voigt Bulk Modulus",
                intMarkerSize = 100)

# -----EXP-----
pltTernary(dfCompositionData = dfX_test_exp*100,
                lstElements = lstNonZeroCols_exp,
                srColor = srY_test_exp,
                strColorBarLabel = "Hardness",
                strTitle = "LR Hardness",
                intMarkerSize = 100)


#%%
# GENERATE LOOKUP TABLES---------------------------------------------------------------------------
'''
the lookup tables are generated from the predictions of the Gaussian Process Regressors
'''

# -----MP-----
lookup_training_task = pd.concat([dfX_test_mp.round(2), srY_test_mp], axis=1)
lookup_training_task["Function"] = "Training_Function"
# change voigt to target
lookup_training_task = lookup_training_task.rename(columns={"voigt": "Target"})
# cast everything in the Target column to float
lookup_training_task["Target"] = lookup_training_task["Target"].astype(float)

# -----EXP-----
lookup_test_task = pd.concat([dfX_test_exp.round(2), srY_test_exp], axis=1)
lookup_test_task["Function"] = "Test_Function"
# change hardness to target
lookup_test_task = lookup_test_task.rename(columns={"hardness": "Target"})
# cast everything in the Target column to float
lookup_test_task["Target"] = lookup_test_task["Target"].astype(float)


#%%
# CREATE OPTIMIZATION OBJECTIVE--------------------------------------------------------------------
'''
the test functions have a single output (voigt bulk modulus and hardness) that is to be maximized
'''

objective = SingleTargetObjective(target = NumericalTarget(name="Target",
                                                           mode="MAX"))

#%%
# CREATE SEARCH SPACE-------------------------------------------------------------------------------
'''
the bounds of the search space are dictated by the upper and lower bounds of each of the elements
'''

lstContinuousParameters = [
        NumericalDiscreteParameter(
        name = f"{strElement}",
        values = np.arange(0, 1.02, 0.02).round(2),
    )for strElement in lstNonZeroCols_exp
]

SumConstraint = [
    DiscreteSumConstraint(
    parameters=lstNonZeroCols_exp,
    condition=ThresholdCondition(  # set condition that should apply to the sum
        threshold=1.0,
        operator="=",
        tolerance=0.001,  # optional; here, everything between 0.999 and 1.001 would also be considered valid
    ),
    )
]

taskParameters = TaskParameter(
    name="Function",
    values=["Test_Function", "Training_Function"],
    active_values=["Test_Function"],
)

lstParameters = [*lstContinuousParameters, taskParameters]

searchspace = SearchSpace.from_product(parameters=lstParameters, constraints=SumConstraint)

#%%

N_MC_ITERATIONS = 10
N_DOE_ITERATIONS = 10
BATCH_SIZE = 1

results: list[pd.DataFrame] = []
for p in (0.01, 0.02, 0.05, 0.08, 0.2):
    campaign = Campaign(searchspace=searchspace, objective=objective)
    initial_data = [lookup_training_task.sample(frac=p) for _ in range(N_MC_ITERATIONS)]
    result_fraction = simulate_scenarios(
        {f"{int(100*p)}": campaign},
        lookup_test_task,
        initial_data=initial_data,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
    )
    results.append(result_fraction)

# For comparison, we also optimize the function without using any initial data:

result_baseline = simulate_scenarios(
    {"0": Campaign(searchspace=searchspace, objective=objective)},
    lookup_test_task,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)
results = pd.concat([result_baseline, *results])

# All that remains is to visualize the results.
# As the example shows, the optimization speed can be significantly increased by
# using even small amounts of training data from related optimization tasks.

results.rename(columns={"Scenario": "% of data used"}, inplace=True)
ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="% of data used",
)
create_example_plots(ax=ax, base_name="basic_transfer_learning")