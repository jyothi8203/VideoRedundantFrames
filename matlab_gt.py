import csv
import os
# from summe import *
import numpy as np
import random
import scipy.io as scio
import json

''' PATHS '''
HOMEDATA = '../GT/';
HOMEVIDEOS = '../videos/';


def convert_numpy_to_list(obj, key=None):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v, k) for k, v in obj.items()}
    elif isinstance(obj, list):
        # print(obj, type(obj), isinstance(obj, np.ndarray))
        return [convert_numpy_to_list(elem) for elem in obj]
    elif hasattr(obj, 'tolist') or isinstance(obj,np.ndarray):  # if obj.shape[0] == 1 if obj.shape[1] == 1 then obj.flatten().list(); if obj
        if key == 'user_score' or key == 'segments' or key == 'all_userIDs':
            ndarry_lst = obj.tolist()
            new_lst = []
            # new_lst.append([ele if isinstance(item,list) else None for item in ndarry_lst for ele in item ] )
            if len(ndarry_lst) == 1:
                if isinstance(ndarry_lst[0], list) and isinstance(ndarry_lst[0][0], np.ndarray):
                    for ele in ndarry_lst[0]:
                        sub_ele = ele.tolist()
                        new_lst.append(sub_ele)
            else:
                new_lst = ndarry_lst
            return new_lst
        else:
            new_lst = []
            ndarry_lst = obj.flatten().tolist()
            if key == 'nFrames' or key == 'video_duration' or key == 'FPS':
                if len(ndarry_lst) == 1:  # obj.shape[0] == 1
                    # if isinstance(ndarry_lst[0], int) or isinstance(ndarry_lst[0], float) or isinstance(ndarry_lst[0], str): # isinstance float
                    new_lst = ndarry_lst[0]
            elif key == 'gt_score':
                if len(ndarry_lst) > 1:  # obj.shape[1] == 1
                    if isinstance(ndarry_lst[0], np.ndarray):
                        new_lst = [ele for ele in ndarry_lst if ele.size == 1]
                    else:
                        new_lst = ndarry_lst
            # print(obj, type(obj), isinstance(obj, np.ndarray))
            return new_lst
    elif isinstance(obj, bytes) or isinstance(obj, bytearray):
        str_d = obj.decode('utf-8')
        # print(str_d, type(str_d), isinstance(str_d, str))
        return str(str_d)
    else:
        return obj

def load_matlab_video_files(vid_name,wrtr):
    # vid_name = vid_name.split('.')[0]
    ext = vid_name.split('.')[1]
    if ext != 'mat':
        gt_file = HOMEDATA + '/' + vid_name + '.mat'
    else:
        gt_file = vid_name
    gt_data = scio.loadmat(gt_file)  # ,spmatrix=False)

    # Convert any non-JSON-serializable types (e.g., NumPy arrays)
    json_compatible_data = convert_numpy_to_list(gt_data)

    with open('./matlab_data_files/'+vid_name + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_compatible_data, json_file)
    no_users = gt_data['user_score'].shape[1]
    wrtr.writerow([vid_name+'.mp4',json_compatible_data['nFrames'],json_compatible_data['segments'],json_compatible_data['all_userIDs'],json_compatible_data['video_duration'],json_compatible_data['FPS'],no_users])

    # csv_writer.writerow([vid_name+'.mp4',json_compatible_data['nFrames'],json_compatible_data['segments'],json_compatible_data['all_userIDs'],json_compatible_data['gt_score'],json_compatible_data['user_score'],])

def process_summe_mat():
    # Take a random video and create a random summary for it
    included_extenstions = ['mp4']

    videoList = [fn for fn in os.listdir(HOMEVIDEOS) if any([fn.endswith(ext) for ext in included_extenstions])]
    # videoName = videoList[int(round(random.random() * 24))]
    fldr_pth = './matlab_data_files'
    if os.path.exists(fldr_pth):
        print("exists")
    else:
        os.mkdir(fldr_pth)
    csv_path = fldr_pth+'/summe_videos_matlab_data1.csv'
    csv_file = open(csv_path,'w',newline='')

    csv_writer = csv.writer(csv_file,dialect=csv.excel)
    csv_writer.writerow(['VideoName','Frames', 'Segments', 'All UserIDs','Video Duration', 'FPS', 'No Users'])

    load_matlab_video_files(videoList[5],csv_writer)


def process_tvsumm_mat():
    root_pth = 'D:\\Project\\Datasets\\VideoSummaries\\tvsumm\\matlab'
    fldr_pth = os.path.join(root_pth,'json')
    if os.path.exists(fldr_pth):
        print("exists")
    else:
        os.mkdir(fldr_pth)
    inp_mat = os.path.join(root_pth,'ydata-tvsum50.mat')
    csv_path = fldr_pth+'/tvsumm_videos_matlab_data.csv'
    csv_file = open(csv_path,'w',newline='')

    csv_writer = csv.writer(csv_file,dialect=csv.excel)
    csv_writer.writerow(['VideoName','Frames', 'Segments', 'All UserIDs','Video Duration', 'FPS', 'No Users'])
    load_matlab_video_files(inp_mat,csv_writer)


if __name__ == "__main__":
    process_summe_mat()
    # process_tvsumm_mat()