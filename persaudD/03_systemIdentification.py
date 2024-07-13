# -*-coding:utf-8 -*-
'''
@Time    :   2024/07/13 13:26:50
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   pair down the two datasets  to a managaable size for the BayBE multi-task learning
'''

#%%
# IMPORT DEPENDENCIES------------------------------------------------------------------------------
import config
import os
import logging
import sys
from datetime import datetime
import pandas as pd
import itertools

# SET UP LOGGING-------------------------------------------------------------------------------

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
# LOAD DATA-----------------------------------------------------------------------------------
dfMP = pd.read_csv(os.path.join(strWD, "data", "mp_bulkModulus_wElementFractions.csv"), index_col=0)
logging.info("Loaded bulk modulus data from csv file")

dfExp = pd.read_csv(os.path.join(strWD, "data", "exp_hardness_wElementFractions.csv"), index_col=0)
logging.info("Loaded experimental hardness data from csv file")

#%%
# REMOVE ELEMENTS WITH ALL ZEROS----------------------------------------------------------------
'''
elements that have all zeros in the columns are not useful for the analysis because they cannot
be included in any system which is common to both the MP and EXP datasets (they are not even
available within one dataset). Therefore, we can remove it to reduce the number of checks in 
the subsequent steps.
'''


# get a list of the elements (from the columns) in the dataframes
lstElementCols = dfMP.columns.tolist()[5:-1]
# get a list of information columns in the dataframes
lstInfoCols_mp = dfMP.columns.tolist()[:5]
lstInfoCols_exp = dfExp.columns.tolist()[:4]



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
# REMOVE ELEMENTS THAT ONLY APPEAR IN ONE DATAFRAME----------------------------------------------
'''
elements that only appear in one dataframe can be removed because the goal is to maximize the 
number of entries in the common columns between the two dataframes. Therefore, elements that
only appear in one dataframe are not useful for the analysis and can be removed to reduce the
number of checks in the subsequent steps.
'''


# find the common columns between the two dataframes
lstCommonElementCols = list(set(lstNonZeroCols_mp).intersection(lstNonZeroCols_exp))

# remove element columns that are not in the common columns
lstNonCommonElementCols_mp = [col for col in lstNonZeroCols_mp if col not in lstCommonElementCols]
lstNonCommonElementCols_exp = [col for col in lstNonZeroCols_exp if col not in lstCommonElementCols]

# drop the columns that are not in the common columns
dfMP.drop(columns=lstNonCommonElementCols_mp, inplace=True)
dfExp.drop(columns=lstNonCommonElementCols_exp, inplace=True)

# drop the ElementFraction Exceptions columns
dfMP.drop(columns="ElementFraction Exceptions", inplace=True)
dfExp.drop(columns="ElementFraction Exceptions", inplace=True)



#%%
# IDENTIFY THE BEST COMBINATION--------------------------------------------------------------------
'''
we want to count the number of entries which have at least one non-zero value for an element in the 
combination and zero values for all other elements.
'''

intSystemSize = 6


# generate all combinations of elements from the common columns
lstElementCombinations = list(itertools.combinations(lstCommonElementCols, intSystemSize))

# initialize a dictionary to store the count of entries for each combination
dicCombinationCounts = {}

# iterate over each combination
for tupCombination_temp in lstElementCombinations:
    # get the list of other elements
    lstOtherElements_temp = [col for col in lstCommonElementCols if col not in tupCombination_temp]
    
    # select the columns for the current combination
    dfMP_system_temp = dfMP[list(tupCombination_temp)]
    dfExp_system_temp = dfExp[list(tupCombination_temp)]

    # check for rows where at least one value in the combination is non-zero
    srContainsCombination_mp = dfMP_system_temp.any(axis=1)
    srContainsCombination_exp = dfExp_system_temp.any(axis=1)

    # check for rows where all other elements have zero values
    srOnlyContainsCombination_mp = (dfMP[lstOtherElements_temp] == 0).all(axis=1)
    srOnlyContainsCombination_exp = (dfExp[lstOtherElements_temp] == 0).all(axis=1)

    # count the number of rows satisfying both conditions
    srValidRows_mp = srContainsCombination_mp & srOnlyContainsCombination_mp
    srValidRows_exp = srContainsCombination_exp & srOnlyContainsCombination_exp

    # count the number of valid rows
    intCount_mp = srValidRows_mp.sum()
    intCount_exp = srValidRows_exp.sum()

    # find the indices of the valid rows
    lstIndices_mp = srValidRows_mp[srValidRows_mp].index.tolist()
    lstIndices_exp = srValidRows_exp[srValidRows_exp].index.tolist()

    # extract the entries from the tuple and make them a string
    strCombination_temp = ",".join(tupCombination_temp)


    # store the count of valid rows in the dictionary
    dicCombinationCounts[strCombination_temp] = {
        "count_mp": intCount_mp,
        "count_exp": intCount_exp,
        "indices_mp": lstIndices_mp,
        "indices_exp": lstIndices_exp
    }

#%%

# find the combination with the highest number of valid rows
intMaxCount_mp = 0
intMaxCount_exp = 0
tupBestCombination = None
for tupCombination_temp, dicCounts_temp in dicCombinationCounts.items():
    if dicCounts_temp["count_mp"] > intMaxCount_mp and dicCounts_temp["count_exp"] > intMaxCount_exp:
        intMaxCount_mp = dicCounts_temp["count_mp"]
        intMaxCount_exp = dicCounts_temp["count_exp"]
        tupBestCombination = tupCombination_temp

#%%
# SAVE THE RESULTS-------------------------------------------------------------------------------
'''
save the dictionary of combination counts to a csv file
'''

# create a dataframe from the dictionary
dfCombinationCounts = pd.DataFrame.from_dict(dicCombinationCounts).T

# save the dataframe to a csv file
strCombinationCountsFilePath = os.path.join(strWD, "data", f"combinationCounts_{intSystemSize}.csv")

dfCombinationCounts.to_csv(strCombinationCountsFilePath)
logging.info(f"Saved the combination counts to {strCombinationCountsFilePath}")


# dicCombinationCounts_6 = dfCombinationCounts
# %%
