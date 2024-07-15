# -*-coding:utf-8 -*-
"""
@Time    :   2024/07/13 12:56:02
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   use matminer to generate features for the MP bulk modulus data and the experimental data
"""



# IMPORT DEPENDENCIES------------------------------------------------------------------------------
import os
import pandas as pd
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.conversions import StrToComposition

if __name__ == "__main__":
    # get the path to the directory before the current directory
    strHomeDir = os.path.dirname(os.getcwd())

    # LOAD DATA-----------------------------------------------------------------------------------
    dfMP = pd.read_csv(
        os.path.join(strHomeDir, "data", "processed", "mp_bulkModulus_cleaned.csv"), index_col=0
    )

    dfExp = pd.read_csv(
        os.path.join(strHomeDir, "data", "processed", "exp_hardness_cleaned.csv"), index_col=0
    )

    # USE MATMINER TO GET COMPOSITIONAL DATA-------------------------------------------------------

    # -----GET COMPOSITIONAL DATA FOR MATERIALS PROJECT DATA-----
    # convert the strComposition to a composition object
    dfMP = StrToComposition().featurize_dataframe(
        dfMP, "strComposition", ignore_errors=True, return_errors=True
    )
    # get the element fractions
    dfMP = ElementFraction().featurize_dataframe(
        dfMP, col_id="composition", ignore_errors=True, return_errors=True
    )

    # -----GET COMPOSITIONAL DATA FOR EXPERIMENTAL HARDNESS DATA-----
    # convert the strComposition to a composition object
    dfExp = StrToComposition().featurize_dataframe(
        dfExp, "strComposition", ignore_errors=True, return_errors=True
    )
    # get the element fractions
    dfExp = ElementFraction().featurize_dataframe(
        dfExp, col_id="composition", ignore_errors=True, return_errors=True
    )

    # drop the StrToComposition Exceptions column
    dfMP.drop(columns=["StrToComposition Exceptions"], inplace=True)
    dfExp.drop(columns=["StrToComposition Exceptions"], inplace=True)

    # drop the ElementFraction Exceptions column
    dfMP.drop(columns=["ElementFraction Exceptions"], inplace=True)
    dfExp.drop(columns=["ElementFraction Exceptions"], inplace=True)

    # drop entries with missing data
    dfMP.dropna(inplace=True)
    dfExp.dropna(inplace=True)

    # EXPORT DATA---------------------------------------------------------------------------------
    dfMP.to_csv(os.path.join(strHomeDir, "data", "processed", "mp_bulkModulus_wElementFractions.csv"))

    dfExp.to_csv(os.path.join(strHomeDir, "data", "processed", "exp_hardness_wElementFractions.csv"))