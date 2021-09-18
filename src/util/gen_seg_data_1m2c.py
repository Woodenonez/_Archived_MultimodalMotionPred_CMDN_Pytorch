import os, sys
from pathlib import Path

import math, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from data_handle.sad_generator import *
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data_handle.sad_generator import *

# 2c stands for 2 channels per image

def gen_csv_trackers(data_dir):
    # data_dir  -  objf(1,2,...)  -  obj&other(2 channels)
    obj_folders = os.listdir(data_dir)
    for objf in obj_folders:
        obj_files   = os.listdir(data_dir+objf+'/obj/')
        other_files = os.listdir(data_dir+objf+'/other/')
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
        df = pd.DataFrame({'f':obj_files,'f_other':other_list,'t':t_list,'x':x_list,'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(data_dir+objf+'/obj/'+'data.csv', index=False)

def gather_all_data(data_dir, past, maxT, save_dir=None, minT=1):
    # data_dir  -  objf(1,2,...)  -  obj&other
    #           -  All(all in one folder)
    if save_dir is None:
        save_dir = data_dir

    column_name = [f'f{i}' for i in range(1,2*past+1)] + ['T', 'x', 'y']
    df_all = pd.DataFrame(columns=column_name)
    obj_folders = os.listdir(data_dir)
    for objf in obj_folders:
        df_obj = pd.read_csv(data_dir+objf+'/obj/'+'data.csv')
        for T in range(minT,maxT+1):
            sample_list = []
            for i in range(len(df_obj)-past-T+1): # each sample
                sample = []
                ################## Sample START ##################
                for j in range(past):
                    sample.append(df_obj.iloc[i+j]['f'])
                    sample.append(df_obj.iloc[i+j]['f_other'])
                sample.append(T)
                sample.append(df_obj.iloc[i+past+T-1]['x'])
                sample.append(df_obj.iloc[i+past+T-1]['y'])
                ################## Sample E N D ##################
                sample_list.append(sample)
            df_T = pd.DataFrame(sample_list, columns=df_all.columns)
            df_all = pd.concat([df_all, df_T], ignore_index=True)
    df_all.to_csv(save_dir+'all_data.csv', index=False)

def save_FTD_data(save_path, past, maxT, ts, map_unit, sim_time):
    ID = 0
    obj_dict_a = {}
    obj_dict_h = {}
    print()
    obj_id_track = []
    obj_shape_track = []
    for k in range(int(sim_time/ts)+1): # k is the [time step] (maybe not in second)

        print(f'\r{int(k*ts)}/{sim_time}s. Active/Not:{len(list(obj_dict_a))}/{len(list(obj_dict_h))}.  ', end='')

        if (np.random.poisson(5,1)>8): # Poisson distribution to decide if there is a new pedestrian
            ID += 1
            key = 'obj{}'.format(ID)
            idx = random.randint(0,pedestrian_path()-1) # pedestrian_path() no input -> length of path list
            info = pedestrian_path(idx)
            obj_dict_a[key] = Moving_Object(0, ID, info[1], [1,0], info=info)

        if (np.random.poisson(5,1)>9.5): # Poisson distribution to decide if there is a new forklift
            ID += 1
            key = 'obj{}'.format(ID)
            vv = 2 # speed
            choice = random.choice([1,2,3,4,5])
            if choice in [1]:
                obj_dict_a[key] = Moving_Object(1,ID,[0,1.0],[vv,0],info=['w','n'])
            elif choice in [2]:
                obj_dict_a[key] = Moving_Object(1,ID,[4.5,10.0],[0,-vv],info=['n','w'])
            elif choice in [3]:
                obj_dict_a[key] = Moving_Object(1,ID,[10.0,2.0],[-vv,0],info=['e','n'])
            elif choice in [4]:
                obj_dict_a[key] = Moving_Object(1,ID,[0,1.0],[vv,0],info=['w','e'])
            elif choice in [5]:
                obj_dict_a[key] = Moving_Object(1,ID,[4.5,10.0],[0,-vv],info=['n','e'])
            elif choice in [6]:
                obj_dict_a[key] = Moving_Object(1,ID,[10.0,2.0],[-vv,0],info=['e','w'])
            # choice += 1

        if (k%int(8/ts)==0):# & (yes): # MPs run in a regular period (8s)
            ID += 1
            key = 'obj{}'.format(ID)
            obj_dict_a[key] = Moving_Object(2,ID,[0,5.5],[1,0],info=None)
            yes = 0

        for name in list(obj_dict_a): # check for active objects
            if out_of_border(obj_dict_a[name].p[0], obj_dict_a[name].p[1]) | obj_dict_a[name].terminate:
                obj_dict_h[name] = obj_dict_a[name]
                del obj_dict_a[name]

        for name1, obj1 in obj_dict_a.items(): # check interactions
            for name2, obj2 in obj_dict_a.items():
                obj_dict_a[name1] = interaction(obj1,obj2)
                if obj_dict_a[name1].stop:
                    break

        obj_p_list = []     # obj position
        obj_id_list = []    # obj ID
        obj_tp_list = []    # obj type
        obj_shape_list = [] # obj shape
        for name, obj in obj_dict_a.items(): # go one step for all active objects
            obj.one_step(sampling_time=ts, map_unit=map_step, ax=None)

            obj_p_list.append(obj.p)
            obj_id_list.append(obj.id)
            obj_tp_list.append(obj.tp)
            obj_shape_list.append(obj.shape)

        for i in range(len(obj_id_list)): # create segmentation
            this_pos = obj_p_list[i]
            mag_seg_obj = gen_seg([obj_tp_list[i]], [obj_shape_list[i]])
            try: # get all non-targeted objects
                mag_seg_others = gen_seg(obj_tp_list[:i]+obj_tp_list[i+1:], obj_shape_list[:i]+obj_shape_list[i+1:])
            except:
                mag_seg_others = gen_seg(obj_tp_list[:i], obj_shape_list[:i])
            mag_seg_obj    = np.rot90(mag_seg_obj)
            mag_seg_others = np.rot90(mag_seg_others)
            
            plt.figure() # images only containing the target
            plt.imshow(mag_seg_obj,    cmap='Greys')
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            Path('./Seg/{}/obj/'.format(obj_id_list[i])).mkdir(parents=True, exist_ok=True)
            plt.savefig('./Seg/{}/obj/{}_{}_{}_{}.png'.format(obj_id_list[i],obj_id_list[i], k, round(this_pos[0],4), round(this_pos[1],4)), bbox_inches='tight')
            plt.close()
            plt.figure() # images only containing non-targeted objects
            plt.imshow(mag_seg_others, cmap='Greys')
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            Path('./Seg/{}/other/'.format(obj_id_list[i])).mkdir(parents=True, exist_ok=True)
            plt.savefig('./Seg/{}/other/{}_{}.png'.format(obj_id_list[i],obj_id_list[i],k), bbox_inches='tight')
            plt.close()
            
            # if obj.id not in obj_id_track:
            #     new_dict = {'id':obj.id, 'tp':obj.tp, 'pos':[np.append(k,obj.p)], 'shape':[obj.shape]}
            #     obj_shape_track.append(new_dict)
            #     obj_id_track.append(obj.id)
            # else:
            #     idx = obj_id_track.index(obj.id)
            #     obj_shape_track[idx]['pos'].append(np.append(k,obj.p))
            #     obj_shape_track[idx]['shape'].append(np.append(k,obj.p))
    ###############################################################################

def save_SAD_data(save_path, ts, sim_time):
    boundary_coords = [(0.1, 0.1), (9.9, 0.1), (9.9, 9.9), (0.1, 9.9)]
    obstacle_list = [ 
        [(0.0, 0.0), (0.0, 10.0), (2.0, 10.0), (2.0, 0.0)],
        [(8.0, 0.0), (8.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
        [(4.0, 6.0), (4.0, 8.0), (6.0, 8.0), (6.0, 6.0)],
     ]
    for i in range(sim_time):
        print(f'\rSimulating: {i+1}/{sim_time}', end='')
        target_size = 0.5
        inflation = 0.5 + (random.randint(0, 20)/10-1) * 0.2
        stagger = 0.2   + (random.randint(0, 20)/10-1) * 0.1
        vmax = 1        + (random.randint(0, 20)/10-1) * 0.2
        ref_path = get_ref_path()
        start = ref_path[0]

        graph = Graph(boundary_coords, obstacle_list, start, end, inflation=inflation)

        obj = MovingObject(start, stagger=stagger)
        obj.run(ref_path, ts, vmax)
        for j, tr in enumerate(obj.traj):
            # images only containing the target
            shape = patches.Circle(tr, radius=target_size/2, fc='k')
            fig1, ax1 = plt.subplots()
            graph.plot_map(ax1, empty=True)
            ax1.add_patch(shape)
            ax1.set_xlim(0,10)
            ax1.set_ylim(0,10)
            ax.axis('off')
            if save_path is None:
                pass
            else:
                folder = os.path.join(save_path,f'{i}/obj/')
                Path(folder).mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.join(folder,f'{i}_{j}_{round(tr[0],4)}_{round(tr[1],4)}.png'), bbox_inches='tight')
                plt.close()
            # images containing everything
            shape = patches.Circle(tr, radius=target_size/2, fc='k')
            fig2, ax2 = plt.subplots()
            graph.plot_map(ax2, clean=True)
            ax2.add_patch(shape)
            ax2.set_xlim(0,10)
            ax2.set_ylim(0,10)
            ax.axis('off')
            if save_path is None:
                plt.show()
                sys.exit(0)
            else:
                folder = os.path.join(save_path,f'{i}/other/')
                Path(folder).mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.join(folder,f'{i}_{j}.png'), bbox_inches='tight')
                plt.close()
    print()


print("Generate synthetic segmentation dataset.")

save_path = os.path.join(Path(__file__).parent.parent.parent, 'Data/SimpleAvoid/') # save in folder
past = 3
T_range = (10, 10)
ts = 0.2
sim_time = 50 # [second] or times

save_SAD_data(save_path, ts, sim_time)

gen_csv_trackers(save_path) # generate CSV tracking files first
print('CSV records for each object generated.')

gather_all_data(save_path, past, maxT=T_range[1], minT=T_range[0]) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')