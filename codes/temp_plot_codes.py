import matplotlib.pyplot as plt
import matplotlib as mpl
import plots
import importlib
import pickle
import gameutils
import seaborn as sns

plt.close('all')
importlib.reload(plots)

# results_dir = 'C:/Users/ht20/Documents/Files/Game_synthetic/v4.1/postprocess/'
# # 'C:/Users/ht20/Documents/Files/Game_synthetic/v4.1/postprocess/'
# # 'C:/Users/ht20/Documents/Files/Game_Shelby_County/postprocess/'
# with open(results_dir + 'postprocess_dicts_EDM10.pkl', 'rb') as f:  # postprocess_dicts
#     [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
#      ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE, ANALYZE_NE_DF, REL_ACTION_DF] = pickle.load(f)

# plots.plot_performance_curves(BASE_DF,
#                               cost_type='Total', normalize=True, ci=None,
#                               deaggregate=False, plot_resilience=False)
# # # [((BASE_DF['decision_type'] == 'indp') | (BASE_DF['decision_type'] == 'ng')) & \
# # #                                       (BASE_DF['rationality'] != 'unbounded') & \
# # #                                       (BASE_DF['auction_type'] != 'OPTIMAL') & \
# # #                                       (BASE_DF['Magnitude'] == 77) & (BASE_DF['sample'] == 0)]

# plots.plot_relative_performance(LAMBDA_DF, lambda_type='U', layer='nan')
# # # [(LAMBDA_DF['auction_type'] == 'UNIFORM')]

# plots.plot_ne_analysis(ANALYZE_NE_DF, ci=None)
# plots.plot_ne_cooperation(ANALYZE_NE_DF, ci=None)
# # [((ANALYZE_NE_DF['decision_type'] == 'indp') | \
# #                                          (ANALYZE_NE_DF['decision_type'] == 'ng')) & \
# #                                         (ANALYZE_NE_DF['rationality'] != 'unbounded') & \
# #                                         (ANALYZE_NE_DF['auction_type'] != 'OPTIMAL') & \
# #                                         (ANALYZE_NE_DF['Magnitude'] == 77) & (ANALYZE_NE_DF['sample'] == 0)]
# plots.plot_relative_actions(REL_ACTION_DF)
# [(REL_ACTION_DF['rationality'] != 'unbounded')]

# # # plots.plot_separated_perform_curves(BASE_DF, x='t', y='cost', cost_type='Total',
# # #                                     ci=95, normalize=False)
# # plots.plot_auction_allocation(RES_ALLOC_DF, ci=95)
# # plots.plot_relative_allocation(ALLOC_GAP_DF, distance_type='gap')
# # plots.plot_run_time(RUN_TIME_DF, ci=95)
# plots.plot_payoff_hist(ANALYZE_NE_DF, compute_payoff_numbers=True, outlier=False)

# [(REL_ACTION_DF['auction_type']!='UNIFORM')]

''' Plot all cooperation gains '''
# COOP_GAIN, COOP_GAIN_TIME = gameutils.cooperation_gain(BASE_DF, LAMBDA_DF, COMBS, ref_state='bgNNUU',
#                                                        states=['bgCCUU', 'bgCNUU', 'bgNCUU'])
# with open('cooperation_gains_dicts.pkl', 'wb') as f:
#     pickle.dump([COOP_GAIN, COOP_GAIN_TIME], f)

with open('cooperation_gains_dicts.pkl', 'rb') as f:  # postprocess_dicts
    [COOP_GAIN, COOP_GAIN_TIME] = pickle.load(f)
plots.plot_cooperation_gain(COOP_GAIN, COOP_GAIN_TIME, ref_state = 'bgNNUU',
                            states = ['bgCCUU', 'bgCNUU', 'bgNCUU'])
# [COOP_GAIN['auction_type']!='UNIFORM']

''' Plot all performance curves '''
# palette = sns.color_palette("Set1", 3)
# ax = sns.lineplot(x='t', y='cost', hue='Magnitude', markers=False,
#              ci=None, data=BASE_DF[(BASE_DF['decision_type'] == 'indp') & (BASE_DF['layer'] == 'nan') & \
#                                    (BASE_DF['cost_type'] == 'Under Supply Perc')],
#              estimator='mean', alpha=.5, legend=True, lw=.8)
# sns.lineplot(x='t', y='cost', hue='Magnitude', style='sample', markers=True,
#              ci=None, data=BASE_DF[(BASE_DF['decision_type'] == 'indp') & (BASE_DF['layer'] == 'nan') & \
#                                    (BASE_DF['cost_type'] == 'Under Supply Perc') & \
#                                    ((BASE_DF['Magnitude'] == 63) | (BASE_DF['Magnitude'] == 80) | \
#                                     (BASE_DF['Magnitude'] == 56))],
#              estimator=None, alpha=1, palette=palette, lw=.5, ax=ax, legend=True,
#              **{'markersize': 4, 'markeredgewidth': .25})
# sns.lineplot(x='t', y='cost', hue='Magnitude', markers=False,
#              ci=None, data=BASE_DF[(BASE_DF['decision_type'] == 'indp') & (BASE_DF['layer'] == 'nan') & \
#                                    (BASE_DF['cost_type'] == 'Under Supply Perc')& \
#                                    ((BASE_DF['Magnitude'] == 63) | (BASE_DF['Magnitude'] == 80) | \
#                                     (BASE_DF['Magnitude'] == 56))],
#              estimator='mean', alpha=1, palette=palette, lw=1.5, ax=ax, legend=False)
# ax.set(xlabel=r'time step $t$', ylabel=r'\% Unmet Demand')
# handles, labels = ax.get_legend_handles_labels()
# labels = ['Config.' if x == 'Magnitude' else x for x in labels]
# lgd = ax.legend(handles, labels, loc='lower left', ncol=1, framealpha=0.35, bbox_to_anchor=(1, 0.12),
#                 fontsize='small', title='Config.', title_fontsize='small')
# plt.savefig('Performance_curves.png', dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))