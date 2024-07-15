# -*-coding:utf-8 -*-
'''
@Time    :   2024/07/12 18:43:07
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   data injest and cleaning for BayBE multi-task learning (Merck call it transfer learning)
'''

#%%
# IMPORT DEPENDENCIES------------------------------------------------------------------------------
from mp_api.client import MPRester
from emmet.core.summary import HasProps
import config
import os
import logging
import sys
from datetime import datetime
import pandas as pd

#%%
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
# -----LOAD MATERIALS PROJECT BULK MODULUS DATA WITH MP-API-----

# try to load the data from local csv file
try:
    dfMP = pd.read_csv(os.path.join(strWD, "data", "mp_bulkModulus.csv"), index_col=0)
    logging.info("Loaded bulk modulus data from csv file")

# if the file does not exist, get the data from the MP-API
except:
    logging.info("Could not find bulk modulus data csv file, getting data from MP-API")
    # get the data from the MP-API
    with MPRester(config.mp_apiKey) as mpr:
        # get all materials with bulk modulus data
        lstMPDocs = mpr.materials.summary.search(
            has_props = [HasProps.elasticity],
            fields = ["material_id",
                      "formula_pretty",
                      "bulk_modulus"]
            )
        
    # initialize dictionary to store data
    dicMP = {}
    # loop through each material and store data in dictionary
    for mpDoc_temp in lstMPDocs:
        try:
            # make a new dictionary entry for each material
            dicMP[mpDoc_temp.material_id] = {"formula": mpDoc_temp.formula_pretty,
                                             "voigt": mpDoc_temp.bulk_modulus["voigt"],
                                             "reuss": mpDoc_temp.bulk_modulus["reuss"],
                                             "vrh": mpDoc_temp.bulk_modulus["vrh"]}
        except:
            # check if there is bulk modulus data
            if mpDoc_temp.bulk_modulus is None:
                logging.error(f"No bulk modulus data for {mpDoc_temp.material_id}")
            else:
                logging.error(f"Error with {mpDoc_temp.material_id}")
            continue

    dfMP = pd.DataFrame.from_dict(dicMP, orient="index")

    # save the data to a csv file
    dfMP.to_csv(os.path.join(strWD, "data", "mp_bulkModulus.csv"))
    logging.info("Saved bulk modulus data to csv file")

# change the formula column name to 'strComposition'
dfMP.rename(columns={"formula": "strComposition"}, inplace=True)

# -----LOAD EXPERIMENTAL HARDNESS DATA-----

# load the data from the csv file
dfExp = pd.read_csv(os.path.join(strWD, "data", "Vickers Hardness Load Data.csv"))
logging.info("Loaded experimental hardness data from local csv file")
# change the formula column name to 'strComposition'
dfExp.rename(columns={"composition": "strComposition"}, inplace=True)


#%%
# CLEAN DATA---------------------------------------------------------------------------------------

# -----CLEAN MATERIALS PROJECT DATA-----
# remove any entries with NaN values
dfMP.dropna(inplace=True)
# for duplicate formulae, keep the first entry
dfMP.drop_duplicates(subset="strComposition", keep="first", inplace=True)

# -----CLEAN EXPERIMENTAL HARDNESS DATA-----
# remove any entries with NaN values
dfExp.dropna(inplace=True)
# # for duplicate formulae, take the mean of the hardness values
# dfExp = dfExp.groupby("Formula").mean().reset_index()

#%%
# SAVE DATA LOCALLY--------------------------------------------------------------------------------

# save the cleaned data to a csv file
dfMP.to_csv(os.path.join(strWD, "data", "mp_bulkModulus_cleaned.csv"))
dfExp.to_csv(os.path.join(strWD, "data", "exp_hardness_cleaned.csv"))