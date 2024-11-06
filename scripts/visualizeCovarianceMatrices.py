# -*-coding:utf-8 -*-
'''
@Time    :   2024/11/05 11:51:11
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   script to visualize the covariance matrices of the different tasks
'''
#%%
# IMPORT DEPENDENCIES----------------------------------------------------------
import torch
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact

# Set a fixed random seed for reproducibility
torch.manual_seed(0)

# define function to plot covariance matrix
def plotCovarianceMatrix(
    tensorCovarianceMatrix: torch.Tensor,
    strTitle: str
):
    '''
    function to plot the covariance matrix of the kernel

    arguments
    ---------------

    tensorCovarianceMatrix: torch.Tensor
        the covariance matrix to be plotted

    strTitle: str
        the title of the plot

    returns
    ---------------
    NONE

    '''

    plt.figure(
        figsize=(7, 5),
        facecolor='w',
        edgecolor='k',
        dpi = 300,
        constrained_layout = True)
    
    sns.heatmap(tensorCovarianceMatrix.detach().numpy(), cmap="coolwarm", square=True)
    plt.title(strTitle)
    intNumTasks = tensorCovarianceMatrix.size(0)
    lstTickLabels = [f"Task {i}" for i in range(intNumTasks)]
    plt.xticks(ticks=range(intNumTasks), labels=lstTickLabels, rotation=45)
    plt.yticks(ticks=range(intNumTasks), labels=lstTickLabels, rotation=0)
    plt.show()

    return

def createKernel_index(
    intNumTasks: int,
    intRank: int,
    boolWithPrior: bool,
    fltEta: float = None,
):
    '''
    function to create an index kernel

    arguments
    ---------------

    intNumTasks: int
        the number of tasks

    intRank: int
        the rank of the kernel

    boolWithPrior: bool
        whether to include the LKJ prior

    fltEta: float
        the eta parameter for the LKJ prior

    returns
    ---------------
    kernel: gpytorch.kernels.IndexKernel
        the index kernel

    '''

    # check that an eta value is provided if the prior is included
    if boolWithPrior and fltEta is None:
        raise ValueError('An eta value must be provided if the LKJ prior is included')
    
    if boolWithPrior:
        # define the LKJ Covariance Prior
        kernel = gpytorch.kernels.IndexKernel(
            num_tasks=intNumTasks,
            # active_dims=0,
            rank=intRank,
            prior=gpytorch.priors.LKJCovariancePrior(
                n=intNumTasks,
                eta=fltEta,
                sd_prior=gpytorch.priors.NormalPrior(0.0, 1.0)
            )
        )
    else:
        kernel = gpytorch.kernels.IndexKernel(
            num_tasks=intNumTasks,
            # active_dims=0,
            rank=intRank
        )

    return kernel

def updateKernel(
    intNumTasks: int = 20,
    intRank: int = 5,
    fltEta: float = 1.0
):
    '''
    function to update the kernel and plot the covariance matrix

    arguments
    ---------------

    intNumTasks: int
        the number of tasks

    intRank: int
        the rank of the kernel

    fltEta: float
        the eta parameter for the LKJ prior

    returns
    ---------------
    NONE

    '''

    # create the task indices
    tensorTaskIndices = torch.arange(intNumTasks)

    # create the index kernel
    kernel = createKernel_index(
        intNumTasks,
        intRank,
        True,
        fltEta
    )
    tensorCovarianceMatrix = kernel(tensorTaskIndices).evaluate()

    # plot the covariance matrix
    plotCovarianceMatrix(
        tensorCovarianceMatrix,
        f'IndexKernel with Rank={intRank}, Eta={fltEta}'
    )



intNumTasks = 2
intRank = intNumTasks
tensor = torch.arange(intNumTasks)

# without LKJ prior
indexKernel_noPrior = createKernel_index(intNumTasks, intRank, False)
covMatrix_noPrior = indexKernel_noPrior(tensor).evaluate()
# plot the covariance matrices
plotCovarianceMatrix(covMatrix_noPrior, 'IndexKernel without LKJ Prior')

interact(updateKernel,
         intNumTasks=widgets.IntSlider(value=20, min=2, max=50, step=1, description='Num Tasks'),
         intRank=widgets.IntSlider(value=5, min=1, max=10, step=1, description='Rank'),
         fltEta=widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Eta (LKJ Prior)'))

# #%%
# # IMPORT DEPENDENCIES----------------------------------------------------------
# from operator import index
# from sklearn import covariance
# import torch
# import gpytorch
# import matplotlib.pyplot as plt
# import seaborn as sns


# #%%
# kernel = gpytorch.kernels.IndexKernel(
#     num_tasks=2,
#     active_dims=0,
#     rank=2,
#     prior=gpytorch.priors.LKJCovariancePrior(
#         n=10,
#         eta=0.1,
#         sd_prior=gpytorch.priors.NormalPrior(0.1, 0.01),
#     )
# )

# task_indices = torch.arange(2)

# covarianceMatrix = kernel(task_indices).evaluate()

# # plot the covariance matrix
# fig, ax = plt.subplots(
#     1, 1,
#     figsize=(7, 5),
#     facecolor='w',
#     edgecolor='k',
#     dpi = 300,
#     constrained_layout = True
# )

# sns.heatmap(covarianceMatrix.detach().numpy(), cmap='coolwarm', ax=ax)
# ax.set_title('Covariance Matrix')
# ax.set_xlabel('Task 1')
# ax.set_ylabel('Task 2')
# # %%
# import torch
# import gpytorch
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define function to plot covariance matrix
# def plot_covariance_matrix(cov_matrix, title):
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cov_matrix.detach().numpy(), cmap="viridis", square=True)
#     plt.title(title)
#     plt.show()

# # Basic IndexKernel
# def create_index_kernel(num_tasks, rank, with_prior=False):
#     kernel = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)
    
#     if with_prior:
#         # Define the LKJ Covariance Prior
#         kernel.covar_factor_prior = gpytorch.priors.LKJCovariancePrior(
#             n=num_tasks, eta=1.0, sd_prior=gpytorch.priors.NormalPrior(0.0, 1.0)
#         )
#     return kernel

# # Example setup
# num_tasks = 10  # example number of tasks
# rank = num_tasks  # rank can be adjusted
# task_indices = torch.arange(num_tasks)

# # Without LKJ prior
# kernel_no_prior = create_index_kernel(num_tasks, rank, with_prior=False)
# cov_matrix_no_prior = kernel_no_prior(task_indices).evaluate()
# plot_covariance_matrix(cov_matrix_no_prior, title="IndexKernel without LKJ Prior")

# # With LKJ prior
# kernel_with_prior = create_index_kernel(num_tasks, rank, with_prior=True)
# cov_matrix_with_prior = kernel_with_prior(task_indices).evaluate()
# plot_covariance_matrix(cov_matrix_with_prior, title="IndexKernel with LKJ Prior")
# # %%
# import torch
# import gpytorch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import ipywidgets as widgets
# from ipywidgets import interact

# # Function to plot the covariance matrix
# def plot_covariance_matrix(cov_matrix, title="Covariance Matrix"):
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cov_matrix.detach().numpy(), cmap="viridis", square=True)
#     plt.title(title)
#     plt.show()

# # Function to create IndexKernel with optional LKJCovariancePrior
# def create_index_kernel(num_tasks, rank, eta=None):
#     kernel = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)
    
#     if eta is not None:
#         # Add LKJ covariance prior
#         kernel.covar_factor_prior = gpytorch.priors.LKJCovariancePrior(
#             n=num_tasks, eta=eta, sd_prior=gpytorch.priors.NormalPrior(0.0, 1.0)
#         )
    
#     return kernel

# # Interactive function to update kernel and plot
# def update_kernel(num_tasks=100, rank=5, eta=1.0):
#     # num_tasks = 100  # Fixed number of tasks
#     task_indices = torch.arange(num_tasks)
    
#     # Create IndexKernel with the specified rank and LKJ prior
#     kernel = create_index_kernel(num_tasks=num_tasks, rank=rank, eta=eta)
#     cov_matrix = kernel(task_indices).evaluate()
    
#     # Plot the covariance matrix
#     plot_covariance_matrix(cov_matrix, title=f"IndexKernel with Rank={rank}, Eta={eta}")

# # Create interactive sliders for rank and eta
# interact(update_kernel, 
#          rank=widgets.IntSlider(value=5, min=1, max=10, step=1, description='Rank'),
#          eta=widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Eta (LKJ Prior)'));


# %%
