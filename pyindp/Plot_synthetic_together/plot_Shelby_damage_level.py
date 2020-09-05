import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from scipy.stats.stats import pearsonr

sns.set(context='notebook', style='darkgrid', font_scale=1.2)
plt.close('all')
FILTER_SCE = '../../data/damagedElements_sliceQuantile_0.90.csv'
df_config = pd.read_csv(FILTER_SCE,header=0)
OUTPUT_DIR = 'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/results/'
# Getting back the objects ###
with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'rb') as f:
    [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
      ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE] = pickle.load(f)
    
LAMBDA_DF_ext=pd.merge(LAMBDA_DF, df_config,
                       left_on=['Magnitude','sample'],right_on=['sce','set'])
selected_data = LAMBDA_DF_ext[(LAMBDA_DF_ext['lambda_U'] != 'nan')]
                              # &(LAMBDA_DF_ext['auction_type'] != 'MDA')
                              # &(LAMBDA_DF_ext['auction_type'] != 'MAA')]
selected_data["lambda_U"] = pd.to_numeric(selected_data["lambda_U"])
# selected_data = selected_data[(selected_data['lambda_U'] >-2)]
sns.lmplot(x="value", y="lambda_U", ci=95,hue="auction_type", data=selected_data)

'''Compute correlation '''
params = ['no_resources', 'value']
auction_types=selected_data.auction_type.unique().tolist()
corr= pd.DataFrame(columns = ['y', 'config_param', 'auction_type', 'pearson_corr', 'p_value'])
y = "lambda_U"
for c in auction_types:
    for x in params:
        print(c, x)
        df_sel = selected_data[(selected_data['auction_type']==c)]
        pc, p = pearsonr(df_sel[x], df_sel[y])
        print('res',c, pc, p)
        corr = corr.append({'y':y, 'config_param':x, 'auction_type':c, 'pearson_corr':pc,
                            'p_value':p}, ignore_index=True)
        
ALLOC_GAP_DF_ext=pd.merge(ALLOC_GAP_DF, df_config,
                       left_on=['Magnitude','sample'],right_on=['sce','set'])
selected_data = ALLOC_GAP_DF_ext[(ALLOC_GAP_DF_ext['gap'] != 'nan')]
                              # &(LAMBDA_DF_ext['auction_type'] != 'MDA')
                              # &(LAMBDA_DF_ext['auction_type'] != 'MAA')]
selected_data["gap"] = pd.to_numeric(selected_data["gap"])
# selected_data = selected_data[(selected_data['lambda_U'] >-2)]
sns.lmplot(x="value", y="gap", ci=95,hue="auction_type", data=selected_data)

'''Compute correlation '''
params = ['no_resources', 'value']
auction_types=selected_data.auction_type.unique().tolist()
y = "gap"
for c in auction_types:
    for x in params:
        print(c, x)
        df_sel = selected_data[(selected_data['auction_type']==c)]
        pc, p = pearsonr(df_sel[x], df_sel[y])
        print('res',c, pc, p)
        corr = corr.append({'y':y, 'config_param':x, 'auction_type':c, 'pearson_corr':pc,
                            'p_value':p}, ignore_index=True)
        
'''Correlation plot'''
sns.set(font_scale=1.2)
dpi = 300
corr['Resource allocation']= corr['auction_type']
corr['config param'] = corr['config_param'].replace({'value': r'$P_d$',
                                                    'no_resources': r'$R_c$'})
for y in corr.y.unique():
    corr_fig = corr[(corr['y']==y)&(corr['pearson_corr']!='nan')]
    corr_fig = corr_fig.pivot_table(values='pearson_corr',
                                    index='Resource allocation',
                                    columns='config param')
    plt.figure(figsize=[1500/dpi,1000/dpi])
    ax = sns.heatmap(corr_fig, annot=False, fmt="1.2f", vmin=-1, vmax=1,
                      cmap="RdYlGn")
    # plt.savefig('corr_'+y+'.png', dpi=dpi, bbox_inches='tight')