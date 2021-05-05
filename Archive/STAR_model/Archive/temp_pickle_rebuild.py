import pickle
import gzip
import numpy
rooy_folder = 'C:/Users/ht20\Documents/Files/STAR_models/Shelby_final_all_Rc/old_data_python2'
with open(rooy_folder+'/train_test_data_nodes.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    
pickle.dump([p[0],p[1]], open('train_test_data_nodes.pkl', "wb" ),
             protocol=pickle.HIGHEST_PROTOCOL)
