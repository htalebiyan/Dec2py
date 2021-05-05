import os
import pandas as pd
import re
import shutil
main_dir = 'C:/Users/ht20/Documents/Files/Game_Shelby_County/results_NE - Copy/'
subfolders = [f.name for f in os.scandir(main_dir) if f.is_dir()]

for f in subfolders:
    for file in os.listdir(main_dir+f):
        if file not in ['agents','Model']:
            if file[:4] != 'objs':
                os.remove(main_dir+f+'/'+file)
        else:
            shutil.rmtree(main_dir+f+'/'+file)

for f in subfolders:
    os.rmdir(main_dir+f)