# -*-coding:utf-8 -*-
"""
@Time    :   2024/07/13 12:56:02
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   use matminer to generate features for the MP bulk modulus data and the experimental data
"""

import logging
import os
import sys
from datetime import datetime

# IMPORT DEPENDENCIES------------------------------------------------------------------------------
import config
import pandas as pd
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.conversions import StrToComposition

if __name__ == "__main__":

    # SET UP LOGGING-------------------------------------------------------------------------------

    # get the path to the current directory
    strWD = os.getcwd()
    # get the name of this file
    strLogFileName = os.path.basename(__file__)
    # split the file name and the extension
    strLogFileName = os.path.splitext(strLogFileName)[0]
    # add .log to the file name
    strLogFileName = os.path.join(f"{strLogFileName}.log")
    # join the log file name to the current directory
    strLogFilePath = os.path.join(strWD, strLogFileName)

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(strLogFilePath, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # LOAD DATA-----------------------------------------------------------------------------------
    dfMP = pd.read_csv(
        os.path.join(strWD, "data", "mp_bulkModulus_cleaned.csv"), index_col=0
    )
    logging.info("Loaded bulk modulus data from csv file")

    dfExp = pd.read_csv(
        os.path.join(strWD, "data", "exp_hardness_cleaned.csv"), index_col=0
    )
    logging.info("Loaded experimental hardness data from csv file")

    # USE MATMINER TO GET COMPOSITIONAL DATA-------------------------------------------------------

    # -----GET COMPOSITIONAL DATA FOR MATERIALS PROJECT DATA-----
    # convert the strComposition to a composition object
    dfMP = StrToComposition().featurize_dataframe(dfMP, "strComposition")
    # get the element fractions
    dfMP = ElementFraction().featurize_dataframe(
        dfMP, col_id="composition", ignore_errors=True, return_errors=True
    )

    # -----GET COMPOSITIONAL DATA FOR EXPERIMENTAL HARDNESS DATA-----
    # convert the strComposition to a composition object
    dfExp = StrToComposition().featurize_dataframe(dfExp, "strComposition")
    # get the element fractions
    dfExp = ElementFraction().featurize_dataframe(
        dfExp, col_id="composition", ignore_errors=True, return_errors=True
    )

    # EXPORT DATA---------------------------------------------------------------------------------
    dfMP.to_csv(os.path.join(strWD, "data", "mp_bulkModulus_wElementFractions.csv"))
    logging.info("Exported materials project data with element fractions to csv file")

    dfExp.to_csv(os.path.join(strWD, "data", "exp_hardness_wElementFractions.csv"))
    logging.info(
        "Exported experimental hardness data with element fractions to csv file"
    )
