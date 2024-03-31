from calcium_signal import CalciumSignal
from config import FILEDICT
import utils as utils

import pandas as pd 
import numpy as np 
import os 
from tqdm import tqdm

def get_file_list(root):
    file_list = os.listdir(root)
    for file in file_list:
        if file.startswith('._'):
            file_list.remove(file)
    return file_list

def get_path_list(root): 
    file_list = get_file_list(root)
    path_list = [os.path.join(root, file) for file in file_list]
    return path_list

def mk_new_dir(path): 
    if not os.path.exists(path):
        os.makedirs(path)

 
def main():
    trial = 'time-1105'
    save_path = f'/Volumes/zhongzh/Data/3D-Ca/save_corr/'
    mk_new_dir(save_path)
    root = FILEDICT[trial]
    path_list = get_path_list(root)
    mean_list = []
    name_list = []
    for path in tqdm(path_list):
        name = path.split('/')[-1]
        print(name)
        signal = CalciumSignal(path)
        # try: 
        #     signal.plot_center(peak_only=True, save=save_path)
        # except:
        #     print(f'Failed to process {name}')
        #     continue
        mean_corr = signal.plot_heat()
        mean_list.append(mean_corr)
        name_list.append(name)
    
    df = pd.DataFrame({
        'Name': name_list, 
        'Mean Correlation': mean_list
    })
    df.to_csv(os.path.join(save_path, f'{trial}.csv'))

if __name__ == '__main__':
    main()

