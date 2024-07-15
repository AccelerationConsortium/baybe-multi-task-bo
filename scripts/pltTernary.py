def pltTernary(dfCompositionData,
               lstElements,
               srColor = [],
               strColorBarLabel = '', 
               strTitle = '',
               intMarkerSize = 5,
               strSavePath = ''):
    '''
    Plot a ternary diagram of the composition data

    Arguments
    ---------
    dfCompositionData : pandas.DataFrame
        dataframe of the composition data (normalized to 100%)
        shape = (n_samples, 3)

    lstElements : list
        list of the elements in the composition data

    srColor : pandas.Series
        series of the color data
        shape = (n_samples,)
        default = []

    strColorBarLabel : str
        label for the color bar
        default = ''

    strTitle : str
        title for the plot
        default = ''


    Returns:
    --------
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    import ternary
    import numpy as np
    import numpy.matlib as nm
    import pandas as pd

    # --- DATA CHECKS ---
    if len(srColor) > 0:
        # cast all the entries in the series to floats
        srColor = srColor.astype(float)
        # make sure that the color series is the same length as the composition data
        if len(srColor) != len(dfCompositionData):
            raise ValueError('The color series must be the same length as the composition data!')
        # check if a color bar label was passed
        if len(strColorBarLabel) == 0:
            raise ValueError('A color bar label must be passed if a color series is passed!')


    # --- PROCESS COMPOSITION DATA ---
    # make a copy of the dataframe
    dfCompositionData_copy = dfCompositionData.copy()
    npCompositionData = dfCompositionData_copy.to_numpy()
    # convert the data to cartesian coordinates
    npCompositionData_cartesian = nm.vstack((npCompositionData[:,0].T,npCompositionData[:,2].T)).T

    # --- PLOT TERNARY DIAGRAM ---
    # setup subplot with 1 row and 1 column
    fig, ax = plt.subplots(1, 1, facecolor='w', edgecolor='k', constrained_layout=True, dpi=500)
    
    # set the scale of the plot
    scale = 100
 
    # setup the ternary plot
    grid = plt.GridSpec(10, 12, wspace=2, hspace=1)
    ax = plt.subplot(grid[:,:9])
    
    # make the ternary plot
    figure, tax = ternary.figure(scale=scale, ax=ax)
    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=10)

    # Set Axis labels and Title
    fontsize = 12
    offset = 0.14
    tax.right_corner_label(lstElements[0], 
                           fontsize=fontsize, 
                           offset=0.2, 
                           fontweight='bold')
    tax.top_corner_label(lstElements[2], 
                         fontsize=fontsize, 
                         offset=0.23, 
                         fontweight='bold')
    tax.left_corner_label(lstElements[1], 
                          fontsize=fontsize, 
                          offset=0.2, 
                          fontweight='bold')
    
    tax.left_axis_label(lstElements[2] + ' at.%',
                        fontsize=fontsize, 
                        offset=offset)
    tax.right_axis_label(lstElements[0] + 'at.%', 
                         fontsize=fontsize, 
                         offset=offset)
    tax.bottom_axis_label(lstElements[1] + ' at.%', 
                          fontsize=fontsize, 
                          offset=offset)
    
    tax.ticks(axis='lbr', 
              multiple=10, 
              linewidth=1, 
              offset=0.025, 
              clockwise= True)
    
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()

    # set the title
    if len(strTitle) > 0:
        tax.set_title(strTitle, 
                      fontsize=18, 
                      y=1.20)

    # check if a color series was passed
    if len(srColor) > 0:
        # Create color map and plot color bar
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(srColor.min(), srColor.max())

        # plot the data with color
        tax.scatter(npCompositionData_cartesian,
                    marker = 'o',
                    c = cmap(norm(srColor)), 
                    vmin = srColor.min(),
                    vmax = srColor.max(),
                    s = intMarkerSize)
        
        # create color bar and plot color bar
        ax = plt.subplot(grid[1:-1,-2:])
        cb1 = matplotlib.colorbar.ColorbarBase(ax, 
                                               cmap = cmap, 
                                               norm = norm, 
                                               orientation='vertical', 
                                               label = strColorBarLabel)
        cb1.set_label(label = strColorBarLabel, 
                      size=18)
    else:
        # plot the data without color
        tax.scatter(npCompositionData_cartesian,
                    marker='o',
                    c='tab:blue',
                    s = intMarkerSize,
                    alpha = 0.5)
        
    # save the plot
    if len(strSavePath) > 0:
        plt.savefig(strSavePath, dpi=300)
    
    return
