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