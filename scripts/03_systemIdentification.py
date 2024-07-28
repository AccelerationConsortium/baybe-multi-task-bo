# -*-coding:utf-8 -*-
"""
@Time    :   2024/07/13 13:26:50
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   pair down the two datasets  to a managaable size for the BayBE multi-task learning
"""

# %%
# IMPORT DEPENDENCIES------------------------------------------------------------------------------
import os
import sys
import time
import itertools
import matplotlib.pyplot as plt

import pandas as pd

# make a function to find the entries in a dataframe which contain the specified elements
def findEntries(lstElements_interest,
                lstElements_total,
                df):
    """
    Find the entries in the dataframe that contain the specified elements

    parameters
    ----------
    lstElements_

    df : pandas.DataFrame
        Dataframe to search for entries.

    returns
    -------
    dfEntries : pandas.DataFrame
        Dataframe containing the entries that contain the specified elements.
    """

    lstEntries = []
    for idx_temp, row_temp in df.iterrows():
        # Check if all specified elements have non-zero values
        all_specified_elements_present = any(row_temp[element] != 0 for element in lstElements_interest)
        
        # Check if all other elements have zero values
        all_other_elements_zero = all(row_temp[element] == 0 for element in lstElements_total if element not in lstElements_interest)
        
        if all_specified_elements_present and all_other_elements_zero:
            lstEntries.append(idx_temp)

    df_wElements = df.loc[lstEntries]

    return df_wElements

def countEntries(lstElements_interest,
                 lstElements_total,
                 df):
    '''
    Count the number of entries in the dataframe that contain the elements in the list

    parameters
    ----------
    lstElements_interest: list
        list of 

    df : pandas.DataFrame
        Dataframe to search for entries.

    returns
    -------
    intCount : int
        Number of entries that contain the elements in the list.
    '''

    intCount = 0
    for idx_temp, row_temp in df.iterrows():
        # Check if all specified elements have non-zero values
        all_specified_elements_present = any(row_temp[element] != 0 for element in lstElements_interest)
        
        # Check if all other elements have zero values
        all_other_elements_zero = all(row_temp[element] == 0 for element in lstElements_total if element not in lstElements_interest)
        
        if all_specified_elements_present and all_other_elements_zero:
            intCount += 1

    return intCount

# %%
# LOAD DATA-----------------------------------------------------------------------------------

# get the path to the directory before the current directory
strHomeDir = os.path.dirname(os.getcwd())

dfMP = pd.read_csv(
    os.path.join(strHomeDir, "data", "processed", "mp_bulkModulus_wElementFractions.csv"), index_col=0
)

dfExp = pd.read_csv(
    os.path.join(strHomeDir, "data", "processed", "exp_hardness_wElementFractions.csv"), index_col=0
)

# %%
# REMOVE ELEMENTS WITH ALL ZEROS----------------------------------------------------------------
"""
elements that have all zeros in the columns are not useful for the analysis because they cannot
be included in any system which is common to both the MP and EXP datasets (they are not even
available within one dataset). Therefore, we can remove it to reduce the number of checks in
the subsequent steps.
"""

# get a list of the elements (from the columns) in the dataframes
lstElementCols = dfMP.columns.tolist()[5:]
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


# %%
# REMOVE ELEMENTS THAT ONLY APPEAR IN ONE DATAFRAME----------------------------------------------
"""
elements that only appear in one dataframe can be removed because the goal is to maximize the
number of entries in the common columns between the two dataframes. Therefore, elements that
only appear in one dataframe are not useful for the analysis and can be removed to reduce the
number of checks in the subsequent steps.
"""

# find the common columns between the two dataframes
lstCommonElementCols = list(set(lstNonZeroCols_mp).intersection(lstNonZeroCols_exp))

# remove element columns that are not in the common columns
lstNonCommonElementCols_mp = [
    col for col in lstNonZeroCols_mp if col not in lstCommonElementCols
]
lstNonCommonElementCols_exp = [
    col for col in lstNonZeroCols_exp if col not in lstCommonElementCols
]

# for entries in dfMP with non-zero values in the non-common columns, drop them
dfMP = dfMP[~dfMP[lstNonCommonElementCols_mp].any(axis=1)]
# for entries in dfExp with non-zero values in the non-common columns, drop them
dfExp = dfExp[~dfExp[lstNonCommonElementCols_exp].any(axis=1)]
"""
Entries in the dataframes that have non-zero values in the non-common columns cannot overlap
with any entries in the other dataframe because the non-common columns are not present in the
other dataframe. Therefore, we can remove these entries to reduce the number of checks in the
subsequent steps.
"""

# drop the columns that are not in the common columns
dfMP.drop(columns=lstNonCommonElementCols_mp, inplace=True)
dfExp.drop(columns=lstNonCommonElementCols_exp, inplace=True)

# cast lstCommonElementCols in bot dataframes to floats
dfMP[lstCommonElementCols] = dfMP[lstCommonElementCols].apply(pd.to_numeric)
dfExp[lstCommonElementCols] = dfExp[lstCommonElementCols].apply(pd.to_numeric)


#%%
# MAKE HISTOGRAM OF THE NUMBER OF NON-ZERO VALUES IN THE ELEMENT COLUMNS---------------------------

# initialize a dictionary to store the number of non-zero values in each column
dicNonZeroCounts = {}

# iterate over each column
for strCol_temp in lstCommonElementCols:
    # count the number of non-zero values in the column
    intNonZeroCount_temp = (dfExp[strCol_temp] != 0).sum()
    # store the count in the dictionary
    dicNonZeroCounts[strCol_temp] = intNonZeroCount_temp

# sort the dictionary by the number of non-zero values
dicNonZeroCounts = dict(sorted(dicNonZeroCounts.items(), key=lambda item: item[1], reverse=True))

# plot the histogram
fig, ax = plt.subplots(1, 1, figsize=(30, 5), facecolor='w', edgecolor='k', dpi = 300, constrained_layout = True)

ax.bar(dicNonZeroCounts.keys(), dicNonZeroCounts.values(), color='b')
ax.set_xlabel("Element")
ax.set_ylabel("Number of non-zero values")
ax.set_title("Number of non-zero values in the element columns")
plt.xticks(fontsize=10);

# Extract elements from chemical formulas and count occurrences
setInitialSelection = set(list(dicNonZeroCounts.keys())[:10])

# -----GREEDY SELECTION-----

setBestElements = setInitialSelection
intBestCount = countEntries(setBestElements,
                            lstCommonElementCols,
                            dfExp)
boolImprovement = True

while boolImprovement:
    boolImprovement = False
    for element_temp in lstCommonElementCols:
        if element_temp not in setBestElements:
            for currentElement_temp in setBestElements:
                lstElements_temp = (setBestElements - {currentElement_temp}) | {element_temp}
                intCount_temp = countEntries(lstElements_temp,
                                             lstCommonElementCols,
                                             dfExp)
                if intCount_temp > intBestCount:
                    setBestElements = lstElements_temp
                    intBestCount = intCount_temp
                    boolImprovement = True
                    break

print(f"Best elements: {setBestElements}"
      f"\nNumber of entries: {intBestCount}")


# find the entries in dfMP and dfExp that contain the best elements
dfEntries_mp = findEntries(list(setBestElements),
                           lstCommonElementCols,
                           dfMP)
dfEntries_exp = findEntries(list(setBestElements),
                            lstCommonElementCols,
                            dfExp)

# drop columns with all zeros
dfEntries_mp = dfEntries_mp.loc[:, (dfEntries_mp != 0).any(axis=0)]
dfEntries_exp = dfEntries_exp.loc[:, (dfEntries_exp != 0).any(axis=0)]

# save the entries to csv files
dfEntries_mp.to_csv(os.path.join(strHomeDir, "data", "processed", "mp_bulkModulus_goodOverlap.csv"))
dfEntries_exp.to_csv(os.path.join(strHomeDir, "data", "processed", "exp_hardness_goodOverlap.csv"))


'''
The following code is legacy and is not used in the current implementation, will be removed
in the future.
'''
# # %%
# # FUNCTION TO FIND THE THE BEST COMBINATION--------------------------------------------------------
# """
# we want to count the number of entries which have at least one non-zero value for an element in the
# combination and zero values for all other elements.
# """

# # CHECK WHATS AVAILABLE AND THEN COUNT

# def getCombinationCount(intSystemSize, lstCommonElementCols):
#     # generate all combinations of elements from the common columns
#     lstElementCombinations = list(
#         itertools.combinations(lstCommonElementCols, intSystemSize)
#     )

#     # initialize a dictionary to store the count of entries for each combination
#     dicCombinationCounts = {}

#     # iterate over each combination
#     for tupCombination_temp in lstElementCombinations:
#         # get the list of other elements
#         lstOtherElements_temp = [
#             col for col in lstCommonElementCols if col not in tupCombination_temp
#         ]

#         # select the columns for the current combination
#         dfMP_system_temp = dfMP[list(tupCombination_temp)]
#         dfExp_system_temp = dfExp[list(tupCombination_temp)]

#         # check for rows where at least one value in the combination is non-zero
#         srContainsCombination_mp = dfMP_system_temp.any(axis=1)
#         srContainsCombination_exp = dfExp_system_temp.any(axis=1)

#         # check for rows where all other elements have zero values
#         srOnlyContainsCombination_mp = (dfMP[lstOtherElements_temp] == 0).all(axis=1)
#         srOnlyContainsCombination_exp = (dfExp[lstOtherElements_temp] == 0).all(axis=1)

#         # count the number of rows satisfying both conditions
#         srValidRows_mp = srContainsCombination_mp & srOnlyContainsCombination_mp
#         srValidRows_exp = srContainsCombination_exp & srOnlyContainsCombination_exp

#         # count the number of valid rows
#         intCount_mp = srValidRows_mp.sum()
#         intCount_exp = srValidRows_exp.sum()

#         # find the indices of the valid rows
#         lstIndices_mp = srValidRows_mp[srValidRows_mp].index.tolist()
#         lstIndices_exp = srValidRows_exp[srValidRows_exp].index.tolist()

#         # extract the entries from the tuple and make them a string
#         strCombination_temp = ",".join(tupCombination_temp)

#         # store the count of valid rows in the dictionary
#         dicCombinationCounts[strCombination_temp] = {
#             "count_mp": intCount_mp,
#             "count_exp": intCount_exp,
#             "indices_mp": lstIndices_mp,
#             "indices_exp": lstIndices_exp,
#         }

#     # convert the dictionary to a dataframe
#     dfCombinationCounts = pd.DataFrame.from_dict(dicCombinationCounts).T
#     # save the dataframe to a csv file
#     dfCombinationCounts.to_csv(
#         os.path.join(strHomeDir, "data", "processed", f"combinationCounts_{intSystemSize}.csv")
#     )

#     intMaxCount_exp = 0
#     intMaxCount_mp = 0

#     # for every row in the dataframe, find the number of valid rows
#     for idx, row in dfCombinationCounts.iterrows():
#         intCount_mp = row["count_mp"]
#         intCount_exp = row["count_exp"]
#         if intCount_mp > intMaxCount_mp and intCount_exp > intMaxCount_exp:
#             intMaxCount_mp = intCount_mp
#             intMaxCount_exp = intCount_exp
#             strBestCombination = idx

#     return dfCombinationCounts, strBestCombination, intMaxCount_mp, intMaxCount_exp

# # %%
# # start the timer
# timeStart = time.time()
# print("Starting the timer for combinations of 2...")
# dfCombinationCounts_2, strBestCombination_2, intMaxCount_mp_2, intMaxCount_exp_2 = (
#     getCombinationCount(2, lstCommonElementCols)
# )
# timeEnd = time.time()
# print(f"Finished in {timeEnd - timeStart} seconds\n")

# # start the timer
# timeStart = time.time()
# print("Starting the timer for combinations of 3...")
# dfCombinationCounts_3, strBestCombination_3, intMaxCount_mp_3, intMaxCount_exp_3 = (
#     getCombinationCount(3, lstCommonElementCols)
# )
# timeEnd = time.time()
# print(f"Finished in {timeEnd - timeStart} seconds\n")
# # find the maximum number in dfCombinationCounts_3 in the count_mp column
# print(dfCombinationCounts_3["count_mp"].max())

# # start the timer
# timeStart = time.time()
# print("Starting the timer for combinations of 4...")
# dfCombinationCounts_4, strBestCombination_4, intMaxCount_mp_4, intMaxCount_exp_4 = (
#     getCombinationCount(4, lstCommonElementCols)
# )
# timeEnd = time.time()
# print(f"Finished in {timeEnd - timeStart} seconds\n")

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
