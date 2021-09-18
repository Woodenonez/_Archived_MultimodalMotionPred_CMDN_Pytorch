import os, sys
from pathlib import Path

import math, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from data_handle.sad_object import *
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data_handle.sad_object import *


def gen_csv_trackers(data_dir):
    # data_dir  -  objf
    obj_folders = os.listdir(data_dir)
    for objf in obj_folders:
        obj_files   = os.listdir(data_dir+objf) # all files/images under this folder
        t_list = []
        x_list = []
        y_list = []
        other_list = []
        invalid_files = []
        for f in obj_files:
            info = f[:-4]
            try:
                t_list.append(int(info.split('_')[1]))
                x_list.append(float(info.split('_')[2]))
                y_list.append(float(info.split('_')[3]))
                other_list.append(info.split('_')[0]+'_'+info.split('_')[1]+'.png')
            except:
                invalid_files.append(f)
                continue
        for f in invalid_files:
            obj_files.remove(f)
        df = pd.DataFrame({'f':obj_files,'t':t_list,'x':x_list,'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(data_dir+objf+'/data.csv', index=False)

def gather_all_data(data_dir, past, maxT, save_dir=None, minT=1):
    # data_dir  -  objf(1,2,...)
    #           -  All(all in one folder)
    if save_dir is None:
        save_dir = data_dir

    column_name = [f'f{i}' for i in range(0,past+1)] + ['T', 'x', 'y']
    df_all = pd.DataFrame(columns=column_name)
    obj_folders = os.listdir(data_dir)
    for objf in obj_folders:
        df_obj = pd.read_csv(data_dir+objf+'/data.csv')
        for T in range(minT,maxT+1):
            sample_list = []
            for i in range(len(df_obj)-past-T): # each sample
                sample = []
                ################## Sample START ##################
                for j in range(past+1):
                    sample.append(df_obj.iloc[i+j]['f'])
                sample.append(T)
                sample.append(df_obj.iloc[i+past+T]['x'])
                sample.append(df_obj.iloc[i+past+T]['y'])
                ################## Sample E N D ##################
                sample_list.append(sample)
            df_T = pd.DataFrame(sample_list, columns=df_all.columns)
            df_all = pd.concat([df_all, df_T], ignore_index=True)
    df_all.to_csv(save_dir+'all_data.csv', index=False)

def save_SAD_data(save_path, ts, sim_time):
    boundary_coords = [(0.1, 0.1), (9.9, 0.1), (9.9, 9.9), (0.1, 9.9)]
    obstacle_list = [ 
        [(0.0, 0.0), (0.0, 10.0), (2.0, 10.0), (2.0, 0.0)],
        [(8.0, 0.0), (8.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
        [(4.0, 4.0), (4.0, 6.0), (6.0, 6.0), (6.0, 4.0)],
     ]
    for i in range(sim_time):
        print(f'\rSimulating: {i+1}/{sim_time}', end='')
        target_size = 0.5
        inflation = 0.5 + (random.randint(0, 20)/10-1) * 0.2
        stagger = 0.2   + (random.randint(0, 20)/10-1) * 0.1
        vmax = 1        + (random.randint(0, 20)/10-1) * 0.2
        ref_path = get_ref_path()
        start = ref_path[0]

        graph = Graph(boundary_coords, obstacle_list, inflation=inflation)

        obj = MovingObject(start, stagger=stagger)
        obj.run(ref_path, ts, vmax)
        traj = obj.traj
        for j, tr in enumerate(obj.traj):
            # images containing everything
            shape = patches.Circle(tr, radius=target_size/2, fc='k')
            fig, ax = plt.subplots()
            graph.plot_map(ax, clean=True) ### NOTE change this
            ax.add_patch(shape)
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)
            ax.axis('off')
            if save_path is None:
                plt.show()
                sys.exit(0)
            else:
                folder = os.path.join(save_path,f'{i}/')
                Path(folder).mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.join(folder,f'{i}_{j}_{round(tr[0],4)}_{round(tr[1],4)}.png'), bbox_inches='tight')
                plt.close()
    print()

print("Generate synthetic segmentation dataset.")

save_path = os.path.join(Path(__file__).parent.parent.parent, 'Data/SimpleAvoid1m1c/') # save in folder
past = 3
T_range = (10, 10)
ts = 0.2
sim_time = 100 # [second] or times

save_SAD_data(save_path, ts, sim_time)

gen_csv_trackers(save_path) # generate CSV tracking files first
print('CSV records for each object generated.')

gather_all_data(save_path, past, maxT=T_range[1], minT=T_range[0]) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')