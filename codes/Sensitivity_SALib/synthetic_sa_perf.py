import numpy as np
import pandas as pd
from SALib.analyze import delta
from scipy.stats.stats import pearsonr
import pickle
import scipy.stats as ss
import statsmodels.api as sa
import scikit_posthocs as sp

'''Compute sensitivity'''
[LAMBDA_DF, ALLOC_GAP_DF, RUN_TIME_DF] = pd.read_pickle('postprocess_dicts_all_topo.pkl')

sens_perf= pd.DataFrame(columns = ['config_param', 'auction', 'topology','delta', 'delta_CI'])
sens_res= pd.DataFrame(columns = ['config_param', 'auction', 'topology','delta', 'delta_CI'])
delta_dict_perf = {}
delta_dict_res = {}
topologies = ['Random', 'ScaleFree', 'Grid']
auction_types = ['UNIFORM', 'MCA', 'MDA', 'MAA']
for topo in topologies:
    for auc in auction_types:
        print(topo,auc)
        gamma_dict={'ScaleFree':[2, 3], 'Random':[0.02, 0.62], 'Grid':[3, 10]}
        problem = {
            'num_vars': 6,
            'names': ['L', 'N', 'gamma', 'pi', 'pd', 'Rc'],
            'bounds': [[2, 4], [10, 50], gamma_dict[topo], [0.001, 0.1], [0.05, 0.5], [2, 400]]}
        print('Resilience')
        LAMBDA_DF['lambda_U'] = LAMBDA_DF['lambda_U'].astype(float)
        sel_LAMBDA_DF = LAMBDA_DF[(LAMBDA_DF['topology']==topo)&\
                                  (LAMBDA_DF['auction_type']==auc)]
        X = pd.concat([sel_LAMBDA_DF[' No. Layers'], sel_LAMBDA_DF[' No. Nodes'],
                        sel_LAMBDA_DF[' Topology Parameter'], sel_LAMBDA_DF[' Interconnection Prob'],
                        sel_LAMBDA_DF[' Damage Prob'], sel_LAMBDA_DF[' Resource Cap']],
                      axis=1, keys=['L', 'N', 'gamma', 'pi', 'pd', 'Rc']).to_numpy()
        Y = pd.concat([sel_LAMBDA_DF['lambda_U']], axis=1, keys=['lambda_U']).to_numpy().ravel()
        delta_perf = delta.analyze(problem, X, Y, num_resamples=100, conf_level=0.95,
                                    print_to_console=True, seed=None)
        for idx, para in enumerate(delta_perf['names']):
            sens_perf = sens_perf.append({'config_param':para, 'auction':auc,
                                          'topology':topo, 'delta':delta_perf['delta'][idx],
                                          'delta_CI':delta_perf['delta_conf'][idx]},
                                          ignore_index=True)
        delta_dict_perf[auc, topo] = pd.DataFrame({x:delta_perf['delta_vec'][idx]\
                                                    for idx,x in enumerate(delta_perf['names'])})
        print('Allocation')
        ALLOC_GAP_DF['gap'] = ALLOC_GAP_DF['gap'].astype(float)
        sel_ALLOC_GAP_DF = ALLOC_GAP_DF[(ALLOC_GAP_DF['topology']==topo)&\
                                        (ALLOC_GAP_DF['auction_type']==auc)]
        X = pd.concat([sel_ALLOC_GAP_DF[' No. Layers'], sel_ALLOC_GAP_DF[' No. Nodes'],
                        sel_ALLOC_GAP_DF[' Topology Parameter'], sel_ALLOC_GAP_DF[' Interconnection Prob'],
                        sel_ALLOC_GAP_DF[' Damage Prob'], sel_ALLOC_GAP_DF[' Resource Cap']],
                      axis=1, keys=['L', 'N', 'gamma', 'pi', 'pd', 'Rc']).to_numpy()
        Y = pd.concat([sel_ALLOC_GAP_DF['gap']], axis=1, keys=['gap']).to_numpy().ravel()
        delta_res = delta.analyze(problem, X, Y, num_resamples=100, conf_level=0.95,
                                  print_to_console=True, seed=None)
        for idx, para in enumerate(delta_res['names']):
            sens_res = sens_res.append({'config_param':para, 'auction':auc,
                                        'topology':topo, 'delta':delta_res['delta'][idx],
                                        'delta_CI':delta_res['delta_conf'][idx]},
                                        ignore_index=True)
        delta_dict_res[auc, topo] = pd.DataFrame({x:delta_res['delta_vec'][idx]\
                                                    for idx,x in enumerate(delta_res['names'])})
'''Compute correlation '''
col = "auction_type"
row = "topology"
params = [' No. Layers', ' No. Nodes', ' Topology Parameter',
          ' Interconnection Prob', ' Damage Prob', ' Resource Cap']
corr= pd.DataFrame(columns = ['y', 'config_param', col, row, 'pearson_corr', 'p_value'])
y = "gap"
for r in topologies:
    for c in auction_types:
        for x in params:
            print(c, r, x)
            df_sel = ALLOC_GAP_DF[(ALLOC_GAP_DF[col]==c)&(ALLOC_GAP_DF[row]==r)]
            pc, p = pearsonr(df_sel[x], df_sel[y])
            print('res',c, r, pc, p)
            corr = corr.append({'y':y, 'config_param':x, col:c, row:r, 'pearson_corr':pc,
                                'p_value':p}, ignore_index=True)

y = "lambda_U"
for r in topologies:
    for c in auction_types:
        for x in params:
            print(c, r, x)
            df_sel = LAMBDA_DF[(LAMBDA_DF[col]==c)&(LAMBDA_DF[row]==r)]
            pc, p = pearsonr(df_sel[x], df_sel[y])
            print('perf',c, r, pc, p)
            corr = corr.append({'y':y, 'config_param':x, col:c, row:r, 'pearson_corr':pc,
                                'p_value':p}, ignore_index=True)

''' compute ranking and save '''
sens_res.loc[:, 'delta_CI/delta'] = sens_res['delta_CI']/sens_res['delta']
for topo in topologies:
    for auc in auction_types:
        cond = (sens_res['topology']==topo)&(sens_res['auction']==auc)
        sens_res.loc[cond, 'rank'] = sens_res.loc[cond, 'delta'].rank(method='average', ascending=False)
sens_perf.loc[:, 'delta_CI/delta'] = sens_perf['delta_CI']/sens_perf['delta']
for topo in topologies:
    for auc in auction_types:
        cond = (sens_perf['topology']==topo)&(sens_perf['auction']==auc)
        sens_perf.loc[cond, 'rank'] = sens_perf.loc[cond, 'delta'].rank(method='average', ascending=False)
''' ANOVA and posthoc tests'''
anova_perf = {}
anova_res = {}
for topo in topologies:
    for auc in auction_types:
        names = delta_perf['names']
        data =[delta_dict_perf[auc, topo][x].values for x in names]
        H, p = ss.kruskal(*data)
        df = pd.melt(delta_dict_perf[auc, topo], id_vars=[], value_vars=delta_dict_perf[auc, topo].columns)
        ph = sp.posthoc_conover(df, val_col='value', group_col='variable', p_adjust='holm')
        anova_perf[auc, topo] = {'anova_p': p, 'posthoc_matrix': ph}

        data =[delta_dict_res[auc, topo][x].values for x in names]
        H, p = ss.kruskal(*data)
        df = pd.melt(delta_dict_res[auc, topo], id_vars=[], value_vars=delta_dict_res[auc, topo].columns)
        ph = sp.posthoc_conover(df, val_col='value', group_col='variable', p_adjust='holm')
        anova_res[auc, topo] = {'anova_p': p, 'posthoc_matrix': ph}

''' Manually correct ranks'''
corrected = pd.read_csv('Results_perf.csv')
sens_perf['rank_corrected'] = corrected['rank_corrected']
corrected = pd.read_csv('Results_res.csv')
sens_res['rank_corrected'] = corrected['rank_corrected']

with open('postprocess_dicts_sens_synth.pkl', 'wb') as f:
    pickle.dump([corr, sens_perf, sens_res, delta_dict_perf, delta_dict_res,
                  anova_perf, anova_res], f)
