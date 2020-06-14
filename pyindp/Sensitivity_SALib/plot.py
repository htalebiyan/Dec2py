import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import plot
sns.set(context='notebook', style='darkgrid', font_scale=0.6)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.close('all')

def plot_radar(raw_data,row_titles,col_titles):
    # initialize the figure
    my_dpi=300
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    # Initialise the spider plot
    ax = plt.subplot(1,1,1, polar=True)
    # Create a color palette:
    my_palette = ['darkorange','forestgreen','crimson','midnightblue','c','k']
    #correct labels
    labels = [x for x in row_titles]
    labels = [r'$R_c$' if x == 'Rc' else x for x in labels]
    labels = [r'$P_d$' if x == 'Pd' else x for x in labels]
    labels = [r'$P_i$' if x == 'Pi' else x for x in labels]
    labels = [r'$\Upsilon$' if x == 'gamma' else x for x in labels]
    means = raw_data.mean(axis=1)
    # Loop to plot
    for row in range(len(raw_data.index)):
        title=row_titles[row]
        labels[row] = labels[row]+'(%1.2f)'%means[title]
        color=my_palette[row]
        # number of variable
        categories=list(raw_data)
        N = len(categories)
         
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
         
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
         
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], col_titles, color='k', y=0.0)
        # Draw ylabels
        ax.set_rlabel_position(10)
        plt.yticks([1,2,3,4,5,6], ["1","2","3","4","5","6"], color="gray")
        plt.ylim(1,7)
        ax.invert_yaxis()
         
        values=raw_data.iloc[row].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=0.4, linestyle='solid', zorder=6)
        ax.fill(angles, values, color=color, alpha=0.2, lw=0.01,zorder=6)
    # Add legend and save to file
    leg = plt.figlegend(ax.lines,labels,loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=3, fontsize='medium',frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)
    plt.savefig('SensRadar_variables.png', dpi=my_dpi, bbox_inches="tight",bbox_extra_artists=(leg, ))