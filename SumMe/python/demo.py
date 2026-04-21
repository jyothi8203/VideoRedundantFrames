#!/usr/bin/env python
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo for the evaluation of video summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script takes a random video, selects a random summary
% Then, it evaluates the summary and plots the performance compared to the human summaries
%
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
% date:        05-16-2014
'''
import os 
from summe import *
import numpy as np
import random
import scipy.io as scio
import json

''' PATHS ''' 
HOMEDATA='../GT/';
HOMEVIDEOS='../videos/';


def convert_numpy_to_list(obj,key=None):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v,k) for k, v in obj.items()}
    elif isinstance(obj, list):
        print(obj,type(obj),isinstance(obj,np.ndarray))
        return [convert_numpy_to_list(elem) for elem in obj]
    elif hasattr(obj, 'tolist') or isinstance(obj, np.ndarray):#if obj.shape[0] == 1 if obj.shape[1] == 1 then obj.flatten().list(); if obj
        if key == 'user_score' or key == 'segments' or key == 'all_userIDs':
            ndarry_lst = obj.tolist()
            new_lst = []
            # new_lst.append([ele if isinstance(item,list) else None for item in ndarry_lst for ele in item ] )
            # for item in ndarry_lst:
            if len(ndarry_lst) == 1:
                if isinstance(ndarry_lst[0], list) and isinstance(ndarry_lst[0][0], np.ndarray):
                    ele15_lst = []
                    for ele in ndarry_lst[0]:
                    #     if ele.size == 1 and isinstance(ele, np.ndarray):
                        sub_ele = ele.tolist()
                        new_lst.append(sub_ele)
            else:
                new_lst = ndarry_lst
            return new_lst
        else:
            new_lst = []
            ndarry_lst = obj.flatten().tolist()
            if key == 'nFrames' or key == 'video_duration' or key == 'FPS':
                if len(ndarry_lst) == 1: # obj.shape[0] == 1
                    # if isinstance(ndarry_lst[0], int) or isinstance(ndarry_lst[0], float) or isinstance(ndarry_lst[0], str): # isinstance float
                    new_lst = ndarry_lst[0]
            elif key == 'gt_score':
                if len(ndarry_lst) > 1: # obj.shape[1] == 1
                    if isinstance(ndarry_lst[0], np.ndarray):
                        if obj.shape[1] == 1:
                            new_lst = obj.reshape(-1).tolist()
                        # new_lst = [ele for ele in ndarry_lst if ele.size == 1]

                    else:
                        new_lst = ndarry_lst
            print(obj,type(obj),isinstance(obj,np.ndarray))
            return new_lst
    elif isinstance(obj, bytes) or isinstance(obj, bytearray):
        str_d = obj.decode('utf-8')
        print(str_d, type(str_d),isinstance(str_d, str))
        return str(str_d)
    else:
        return obj

if __name__ == "__main__":
    # Take a random video and create a random summary for it
    included_extenstions=['mp4']
    videoList=[fn for fn in os.listdir(HOMEVIDEOS) if any([fn.endswith(ext) for ext in included_extenstions])]
    videoName = videoList[int(round(random.random()*24))]
    videoName=videoName.split('.')[0]                                    
    
    #In this example we need to do this to now how long the summary selection needs to be
    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scio.loadmat(gt_file)#,spmatrix=False)
    # import scipy.io
    # import json

    # Load the .mat file
    # mat_data = scipy.io.loadmat('your_file.mat')

    # Convert any non-JSON-serializable types (e.g., NumPy arrays)
    json_compatible_data = convert_numpy_to_list(gt_data)
    with open(videoName+'.json', 'w',encoding='utf-8') as json_file:
        json.dump(json_compatible_data, json_file)

    # Save the data to a JSON file
    # with open(videoName+'.json', 'w') as json_file:
    #     # json.dump(json_compatible_data, json_file, indent=4)
    #     for key, val in json_compatible_data.items():
    #         print(key,type(key),type(val))
    #         if isinstance(val, str) or isinstance(key, str):
    #             if isinstance(val,list):
    #                 val = str(val)
    #             json_file.write(key)
    #             json_file.write(val)
    #             # json_file.write(json_compatible_data[key])
    # print("Conversion complete: your_file.mat converted to output.json")
    # print(gt_data.keys())
    nFrames=gt_data.get('nFrames')[0][0]
    user_score=gt_data.get('user_score')

    print(nFrames)
    # json_data = {}
    # 'create a gt_matlab_json.json file'
    # with open(videoName+'.json','w',encoding='utf-8') as fil:
    #     for itm_ky, itm_val in gt_data.items():
    #         if isinstance(itm_val, np.ndarray):
    #             if itm_val.shape[0]==1 or itm_val.shape[1]==1:
    #                 json_data[str(itm_ky)] = itm_val.flatten().tolist()
    #             elif itm_val.shape[0]>1:
    #                 ndarry_lst = itm_val.tolist()
    #                 json_data[str(itm_ky)] = ndarry_lst
    #                 new_lst = []
    #                 # new_lst.append([ele if isinstance(item,list) else None for item in ndarry_lst for ele in item ] )
    #                 for item in ndarry_lst:
    #                     if isinstance(item, list):
    #                         ele15_lst = []
    #                         for ele in item:
    #                             ele15_lst.append(ele)
    #                         new_lst.append(ele15_lst)
    #                 json_data["user_scores_loop"] = new_lst
    #         else:
    #             key = str(itm_ky)
    #             json_data[str(itm_ky)] =  str(itm_val)
    #     #     json.dump(itm_val, fil)
    #     fil.write(json.dumps(json_data, indent=4))
    '''Example summary vector''' 
    #selected frames set to n (where n is the rank of selection) and the rest to 0
    summary_selections={};

    summary_selections[0] = np.random.random(size=(nFrames,1));
    # summary_selections[0] = 1950
    print(summary_selections[0])
    lst1 = summary_selections[0]
    summ_selc_lst = lst1


    '''Evaluate'''
    #get f-measure at 15% summary length
    [f_measure,summary_length]=evaluateSummary(summ_selc_lst,videoName,HOMEDATA,user_score)
    print('F-measure : %.3f at length %.2f' % (f_measure, summary_length))
    
    '''plotting'''
    methodNames={'Random'};
    plotAllResults(summary_selections,methodNames,videoName,HOMEDATA);
