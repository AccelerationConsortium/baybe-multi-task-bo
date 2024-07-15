# -*-coding:utf-8 -*-
"""
@Time    :   2024/07/09 09:38:28
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   test working with BayBE using the example on github
"""

from baybe.objectives import SingleTargetObjective

# %%
from baybe.targets import NumericalTarget

target = NumericalTarget(
    name="Yield",
    mode="MAX",
)
objective = SingleTargetObjective(target=target)

# %%
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)

parameters = [
    CategoricalParameter(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",  # one-hot encoding of categories
    ),
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,  # allows experimental inaccuracies up to 0.2 when reading values
    ),
    SubstanceParameter(
        name="Solvent",
        data={
            "Solvent A": "COC",
            "Solvent B": "CCC",  # label-SMILES pairs
            "Solvent C": "O",
            "Solvent D": "CS(=O)C",
        },
        encoding="MORDRED",  # chemical encoding via mordred package
    ),
]

# %%
from baybe.searchspace import SearchSpace

searchspace = SearchSpace.from_product(parameters)

# %%
from baybe.recommenders import (  # BotorchRecommender,                     # -- NOT WORKING
    FPSRecommender,
    TwoPhaseMetaRecommender,
)

recommender = TwoPhaseMetaRecommender(
    initial_recommender=FPSRecommender(),  # farthest point sampling
    # recommender=BotorchRecommender(),       # Bayesian model-based optimization -- NOT WORKING
)

# %%
from baybe import Campaign

campaign = Campaign(searchspace, objective, recommender)

# %%
df = campaign.recommend(batch_size=3)
print(df)
# %%
df["Yield"] = [79.8, 54.1, 59.4]
campaign.add_measurements(df)
# %%
