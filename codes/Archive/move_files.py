import os
import pandas as pd
import re
import shutil
main_dir = 'C:/Users/ht20/Documents/Files/Shelby_data_paper/'
results_dir = main_dir+'Restoration_results_90_perc/' 
subfolders = [f.name for f in os.scandir(results_dir) if f.is_dir()]

sce_list = 'C:/Users/ht20/Documents/GitHub/td-DINDP/data/damagedElements_sliceQuantile_0.90.csv'
list_high_dam = pd.read_csv(sce_list)

for f in subfolders:
    sce_num = int(f.split('m')[1].split('_')[0])
    for file in os.listdir(results_dir+f):
        if file not in ['agents','Model']:
            sample_num = int(re.findall(r'\d+', file)[0])
            is_in_list = len(list_high_dam.loc[(list_high_dam.set == sample_num)&\
                                                (list_high_dam.sce == sce_num)].index) != 0
            if not is_in_list:
                new_folder = main_dir+'temp/'+f
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                os.rename(results_dir+f+'/'+file, new_folder+'/'+file)
        elif file == 'Model':
            shutil.rmtree(results_dir+f+'/Model')
        elif file == 'agents':
            for agefile in os.listdir(results_dir+f+'/agents'):
                sample_num = int(re.findall(r'\d+', agefile)[0])
                is_in_list = len(list_high_dam.loc[(list_high_dam.set == sample_num)&\
                                    (list_high_dam.sce == sce_num)].index) != 0
                if not is_in_list:
                    new_folder = main_dir+'temp/'+f+'/agents'
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    os.rename(results_dir+f+'/agents/'+agefile, new_folder+'/'+agefile)

for f in subfolders:
    if os.listdir(results_dir+f) == ['agents']:
        os.rmdir(results_dir+f+'/agents')
        os.rmdir(results_dir+f)