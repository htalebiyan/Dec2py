import numpy as np
import pandas as pd
import pickle

root = 'C:/Users/ht20/Documents/Files/NIST_testbeds/Seaside/'
data_1000yr = pd.read_pickle(root+'1000yr/postprocess_dicts.pkl')
data_1000yr[2]['return_period'] = 1000
data_500yr = pd.read_pickle(root+'500yr/postprocess_dicts.pkl')
data_500yr[2]['return_period'] = 500
data_5000yr = pd.read_pickle(root+'5000yr/postprocess_dicts.pkl')
data_5000yr[2]['return_period'] = 5000
data = pd.concat([data_1000yr[2], data_500yr[2], data_5000yr[2]])
data = data.reset_index()
with open('postprocess_all.pkl', 'wb') as f:
    pickle.dump([data_1000yr[0], data_1000yr[0], data, data_1000yr[0],
                 data_1000yr[0]], f)

