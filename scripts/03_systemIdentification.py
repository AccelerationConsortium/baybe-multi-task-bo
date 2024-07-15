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
import os
import logging
import sys
import pandas as pd
import itertools
import time

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

# drop the ElementFraction Exceptions columns
dfMP.drop(columns="ElementFraction Exceptions", inplace=True)
dfExp.drop(columns="ElementFraction Exceptions", inplace=True)

# find the common columns between the two dataframes
lstCommonElementCols = list(set(lstNonZeroCols_mp).intersection(lstNonZeroCols_exp))

# remove element columns that are not in the common columns
lstNonCommonElementCols_mp = [col for col in lstNonZeroCols_mp if col not in lstCommonElementCols]
lstNonCommonElementCols_exp = [col for col in lstNonZeroCols_exp if col not in lstCommonElementCols]

# for entries in dfMP with non-zero values in the non-common columns, drop them
dfMP = dfMP[~dfMP[lstNonCommonElementCols_mp].any(axis=1)]
# for entries in dfExp with non-zero values in the non-common columns, drop them
dfExp = dfExp[~dfExp[lstNonCommonElementCols_exp].any(axis=1)]
'''
Entries in the dataframes that have non-zero values in the non-common columns cannot overlap
with any entries in the other dataframe because the non-common columns are not present in the
other dataframe. Therefore, we can remove these entries to reduce the number of checks in the
subsequent steps.
'''

# drop the columns that are not in the common columns
dfMP.drop(columns=lstNonCommonElementCols_mp, inplace=True)
dfExp.drop(columns=lstNonCommonElementCols_exp, inplace=True)


#%%
# IDENTIFY THE BEST COMBINATION--------------------------------------------------------------------
'''
we want to count the number of entries which have at least one non-zero value for an element in the 
combination and zero values for all other elements.
'''

# intSystemSize = 3

def getCombinationCount (intSystemSize, lstCommonElementCols):
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

    # convert the dictionary to a dataframe
    dfCombinationCounts = pd.DataFrame.from_dict(dicCombinationCounts).T
    # save the dataframe to a csv file
    dfCombinationCounts.to_csv(os.path.join(strWD, "data", f"combinationCounts_{intSystemSize}.csv"))
    logging.info(f"Saved the combination counts to {os.path.join(strWD, 'data', f'combinationCounts_{intSystemSize}.csv')}")

    intMaxCount_exp = 0 
    intMaxCount_mp = 0

    # for every row in the dataframe, find the number of valid rows
    for idx, row in dfCombinationCounts.iterrows():
        intCount_mp = row["count_mp"]
        intCount_exp = row["count_exp"]
        if intCount_mp > intMaxCount_mp and intCount_exp > intMaxCount_exp:
            intMaxCount_mp = intCount_mp
            intMaxCount_exp = intCount_exp
            strBestCombination = idx

    return dfCombinationCounts, strBestCombination, intMaxCount_mp, intMaxCount_exp


#%%
# start the timer
timeStart = time.time()
print("Starting the timer for combinations of 2...")
dfCombinationCounts_2, strBestCombination_2, intMaxCount_mp_2, intMaxCount_exp_2 = getCombinationCount(2, lstCommonElementCols)
timeEnd = time.time()
print(f"Finished in {timeEnd - timeStart} seconds\n")

# start the timer
timeStart = time.time()
print("Starting the timer for combinations of 3...")
dfCombinationCounts_3, strBestCombination_3, intMaxCount_mp_3, intMaxCount_exp_3 = getCombinationCount(3, lstCommonElementCols)
timeEnd = time.time()
print(f"Finished in {timeEnd - timeStart} seconds\n")

# start the timer
timeStart = time.time()
print("Starting the timer for combinations of 4...")
dfCombinationCounts_4, strBestCombination_4, intMaxCount_mp_4, intMaxCount_exp_4 = getCombinationCount(4, lstCommonElementCols)
timeEnd = time.time()
print(f"Finished in {timeEnd - timeStart} seconds\n")

# # start the timer
# timeStart = time.time()
# print("Starting the timer for combinations of 5...")
# dfCombinationCounts_5, strBestCombination_5, intMaxCount_mp_5, intMaxCount_exp_5 = getCombinationCount(5, lstCommonElementCols)
# timeEnd = time.time()
# print(f"Finished in {timeEnd - timeStart} seconds\n")

# #%%
# # start the timer
# timeStart = time.time()
# print("Starting the timer for combinations of 6...")
# dfCombinationCounts_6, strBestCombination_6, intMaxCount_mp_6, intMaxCount_exp_6 = getCombinationCount(6, lstCommonElementCols)
# timeEnd = time.time()
# print(f"Finished in {timeEnd - timeStart} seconds\n")


    
# %%
