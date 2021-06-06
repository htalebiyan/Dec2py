import pandas as pd

results_folder = 'C:/Users/ht20/Documents/Files/Game_synthetic/v4.1/postprocess/'
dfs = pd.read_pickle(results_folder + 'postprocess_dicts_BAYESGAME_bgNNUU_EDM10_OPTIMALandUNIFORM_CF125.pkl')

base_df = dfs[2]
base_df = base_df[base_df['t'] == 0]
for idx, row in base_df.iterrows():
    if row['decision_type'] != 'indp':
        ref_cost = base_df[(base_df['Magnitude'] == row['Magnitude']) & \
                           (base_df['sample'] == row['sample']) & \
                           (base_df['decision_type'] == 'indp') & \
                           (base_df['cost_type'] == row['cost_type']) & \
                           (base_df['layer'] == row['layer'])]
        if ref_cost['cost'].values[0] != row['cost']:
            print(row['Magnitude'], row['sample'], row['decision_type'], row['auction_type'],
                  row['layer'], row['cost'], ref_cost['cost'].values[0], row['cost_type'])
