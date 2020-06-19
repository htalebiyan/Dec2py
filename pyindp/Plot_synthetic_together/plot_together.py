import plots
import seaborn as sns
import os.path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

plt.close('all')
sns.set(context='notebook',style='darkgrid', font_scale=1.2)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
dpi = 300

[LAMBDA_DF, ALLOC_GAP_DF, RUN_TIME_DF] = pd.read_pickle('postprocess_dicts_all_topo.pkl')

""" Plot Lambda results """    
selected_df = LAMBDA_DF[(LAMBDA_DF['lambda_U']!='nan')]
selected_df["lambda_TC"] = pd.to_numeric(selected_df["lambda_U"])

selected_df = selected_df.rename(columns={"lambda_U": "lambda U",
                                          "auction_type": "Auction Type",
                                          "topology":"Topology"})
g = sns.catplot(x=' No. Layers', y='lambda U', hue='Auction Type',
                  col='Topology',data=selected_df, kind='bar',palette="Reds",
                  linewidth=0.5,edgecolor=[.25,.25,.25], capsize=.05,
                  errcolor=[.25,.25,.25],errwidth=1, height=6, aspect=0.5)
g.fig.set_size_inches(4000/dpi,1200/dpi)
g._legend.remove()
g.axes[0,0].set_ylabel(r'$E[\lambda_U]$')
g.axes[0,0].set_xlabel(r'$L$')
g.axes[0,1].set_xlabel(r'$L$')
g.axes[0,2].set_xlabel(r'$L$')
g.set_titles(col_template = 'Topology: {col_name}')
handles, labels = g.axes[0,2].get_legend_handles_labels()
lgd = g.axes[0,2].legend(handles, labels,loc='lower left', bbox_to_anchor=(0, 0),
            frameon =True,framealpha=0.9, ncol=1, title='Res. Alloc. Type') 
plt.savefig('Relative_Performance_synthetic.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')
""" Plot allocation gap """    
selected_df = ALLOC_GAP_DF[(ALLOC_GAP_DF['gap']!='nan')&
                      (ALLOC_GAP_DF['auction_type']!='UNIFORM')]
selected_df["gap"] = pd.to_numeric(selected_df["gap"])
plots.plot_relative_allocation_synthetic(selected_df,distance_type='gap')

''' time plot '''  
selected_df = RUN_TIME_DF[(RUN_TIME_DF['decision_time']<1000)&(RUN_TIME_DF['t']>0)] 
selected_df["decision_time"] = pd.to_numeric(selected_df["decision_time"])
#normalize the valuation time by the number of resources
selected_df["valuation_time"] = selected_df["valuation_time"]/selected_df["no_resources"]
selected_df["Total Time"] = selected_df["valuation_time"]+selected_df["decision_time"]+\
                            selected_df["auction_time"]
selected_df = selected_df.rename(columns={"auction_type": "Auction Type",
                                    "valuation_time": "Valuation Time",
                                    'topology':'Topology',
                                    'auction_time':'Auction Time'})

time_names = ['Valuation Time','decision_time','Auction Time','Total Time']
id_vars = [x for x in list(selected_df.columns) if x not in time_names]
selected_df = pd.melt(selected_df, id_vars=id_vars, value_vars=time_names)
# Removing non-informative lines
selected_df = selected_df[~((selected_df['decision_type']=='indp')&(selected_df['variable']=='Valuation Time'))]
selected_df = selected_df[~((selected_df['decision_type']=='indp')&(selected_df['variable']=='decision_time'))]
selected_df = selected_df[~((selected_df['decision_type']=='indp')&(selected_df['variable']=='Auction Time'))]
selected_df = selected_df[~((selected_df['Auction Type']=='UNIFORM')&(selected_df['variable']=='Valuation Time'))]
selected_df = selected_df[~((selected_df['Auction Type']=='UNIFORM')&(selected_df['variable']=='decision_time'))]
selected_df = selected_df[~((selected_df['Auction Type']=='UNIFORM')&(selected_df['variable']=='Auction Time'))]
selected_df = selected_df[~((selected_df['Auction Type']=='MCA')&(selected_df['variable']=='decision_time'))]
selected_df = selected_df[~((selected_df['Auction Type']=='MDA'))]
selected_df = selected_df[~((selected_df['Auction Type']=='MAA'))]

#Plot
flatui = ["#fc7e2f", "#377eb8", "#E41a1c", "#649d66", "k"]
clrs=['#e9937c', '#a6292d', '#000000']
with sns.color_palette(clrs):#sns.color_palette("Set1", 5)
    g = sns.relplot(x="t", y="value", col="Topology", hue="Auction Type", style='variable',
                    kind="line", ci=None, data=selected_df, legend='full', 
                    markers=True, ms=7,
                    style_order=['Total Time',"Valuation Time",'Auction Time'])

# Correct the legend
labels_objs = g._legend.texts
rep_dict = {'Auction Type':'Allocation Type','variable':'Time Type', 'nan':'iINDP',}
for x in labels_objs:
    if x._text in rep_dict.keys():
        x._text = rep_dict[x._text]
    x.set_fontsize(14)
g._legend._loc = 1
g._legend.set_bbox_to_anchor((0.59, 0.88))
g._legend.set_frame_on(True)

g.axes[0][0].set_ylabel(r'Mean Time (sec)')
g.axes[0][0].set_xlabel(r'Time Step')
g.axes[0][1].set_xlabel(r'Time Step')
g.axes[0][2].set_xlabel(r'Time Step')
g.axes[0][0].set_title(r'Topology: Random')
g.axes[0][1].set_title(r'Topology: Scale Free')
g.axes[0][2].set_title(r'Topology: Grid')
g.axes[0][0].xaxis.set_ticks(np.arange(1, 11, 1.0))#ax.get_xlim()

dpi = 600
g.fig.set_size_inches(8000/dpi, 3000/dpi)
plt.savefig('time_synthetic.png', dpi=dpi, bbox_inches='tight') 