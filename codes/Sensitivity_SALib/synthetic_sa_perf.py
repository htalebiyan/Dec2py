import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import delta
from scipy.stats.stats import pearsonr
import pickle
import scipy.stats as ss
import statsmodels.api as sa
import scikit_posthocs as sp

# %% read and add configuration data
config_list_folder = 'C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v4.1/GeneralNetworks/'
config_data = pd.read_csv(config_list_folder + 'List_of_Configurations.txt', header=0, sep="\t")
config_data = config_data.assign(topology='general')

results_folder = 'C:/Users/ht20/Documents/Files/Game_synthetic/v4.1/postprocess/'
dfs = pd.read_pickle(results_folder + 'postprocess_dicts_EDM10.pkl')
comp_df = pd.merge(dfs[4], config_data, left_on=['Magnitude'], right_on=['Config Number'])
comp_df['lambda_U'] = pd.to_numeric(comp_df['lambda_U'], errors='coerce')

# %% Compute sensitivity
sens_perf = pd.DataFrame(columns=['config_param', 'auction', 'decision', 'delta', 'delta_CI'])
delta_dict_perf = {}
decision_types = ['ng', 'bgCCUU', 'bgCNUU', 'bgNCUU', 'bgNNUU']
auction_types = ['UNIFORM', 'OPTIMAL']  # 'MCA', 'MDA', 'MAA'
for decision in decision_types:
    for auc in auction_types:
        print('Resilience', decision, auc)
        problem = {
            'num_vars': 4,
            'names': ['N', 'pi', 'pd', 'Rc'],
            'bounds': [[10, 50], [0.001, 0.05], [0.05, 0.5], [2, 60]]}
        comp_df['lambda_U'] = comp_df['lambda_U'].astype(float)
        sel_LAMBDA_DF = comp_df[(comp_df['decision_type'] == decision) & (comp_df['auction_type'] == auc)
                                & (comp_df['layer'] == 'nan')]
        X = pd.concat([sel_LAMBDA_DF[' No. Nodes'], sel_LAMBDA_DF[' Interconnection Prob'],
                       sel_LAMBDA_DF[' Damage Prob'], sel_LAMBDA_DF[' Resource Cap ']],
                      axis=1, keys=['N', 'pi', 'pd', 'Rc']).to_numpy()
        Y = pd.concat([sel_LAMBDA_DF['lambda_U']], axis=1, keys=['lambda_U']).to_numpy().ravel()
        delta_perf = delta.analyze(problem, X, Y, num_resamples=100, conf_level=0.95,
                                   print_to_console=True, seed=None)
        for idx, para in enumerate(delta_perf['names']):
            sens_perf = sens_perf.append({'config_param': para, 'auction': auc,
                                          'decision': decision, 'delta': delta_perf['delta'][idx],
                                          'delta_CI': delta_perf['delta_conf'][idx]},
                                         ignore_index=True)
        delta_dict_perf[auc, decision] = pd.DataFrame.from_dict(
            {x: delta_perf['delta_vec'][idx] for idx, x in enumerate(delta_perf['names'])})

# %% Compute correlation
col = "auction_type"
row = "decision_type"
params = [' No. Nodes', ' Interconnection Prob', ' Damage Prob', ' Resource Cap ']
corr = pd.DataFrame(columns=['y', 'config_param', col, row, 'pearson_corr', 'p_value'])
for r in decision_types:
    for c in auction_types:
        for x in params:
            print(c, r, x)
            df_sel = comp_df[(comp_df[col] == c) & (comp_df[row] == r)]
            pc, p = pearsonr(df_sel[x], df_sel["lambda_U"])
            print('perf', c, r, pc, p)
            corr = corr.append({'y': "lambda_U", 'config_param': x, col: c, row: r, 'pearson_corr': pc,
                                'p_value': p}, ignore_index=True)

# %%  compute ranking
sens_perf.loc[:, 'delta_CI/delta'] = sens_perf['delta_CI'] / sens_perf['delta']
for decision in decision_types:
    for auc in auction_types:
        cond = (sens_perf['decision'] == decision) & (sens_perf['auction'] == auc)
        sens_perf.loc[cond, 'rank'] = sens_perf.loc[cond, 'delta'].rank(method='average', ascending=False)

# %%  ANOVA and posthoc tests
anova_perf = {}
for decision in decision_types:
    for auc in auction_types:
        data = [delta_dict_perf[auc, decision][x].values for x in delta_perf['names']]
        H, p = ss.kruskal(*data)
        df = pd.melt(delta_dict_perf[auc, decision], id_vars=[], value_vars=delta_dict_perf[auc, decision].columns)
        ph = sp.posthoc_conover(df, val_col='value', group_col='variable', p_adjust='holm')
        anova_perf[auc, decision] = {'anova_p': p, 'posthoc_matrix': ph}

# %%  Manually correct ranks
corrected = pd.read_csv('Results_perf.csv')
sens_perf['rank_corrected'] = corrected['rank_corrected']
with open('postprocess_dicts_sens_synth.pkl', 'wb') as f:
    pickle.dump([corr, sens_perf, delta_dict_perf, anova_perf], f)
