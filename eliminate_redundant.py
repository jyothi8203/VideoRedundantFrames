# #canny edge filtering techniue
# #dis similarity - euclidean & mahalanabos
# #pixel wise
# #L2 similarity
# #pairwise frame smilarity in attention approach
# from statistics import correlation
# #Optical flow
# #find correlation with SLH Sparse correspondence algorithm
# #DataSets - MSCOCO, Cityscapes, nuscenes, ADE20K, NYUV2,LVIS, BDD100K, KITTI-360, yOUtUBE-vis2019, Objects365

import os
import shutil
import cv2
import numpy as np

# Initialize video capture
import ffmpegcv
import glob
import cv2 as cv
import os,shutil
from datetime import datetime
import csv
import scipy.io as scio
import numba as nb
import multiprocessing
# from numba import jit, cuda
MAX_THRES = 0.9
MIN_THRES = 0.7
height = width = 0


def mse(img1, img2):
    return ((img1 - img2) ** 2).mean()

def ssim_frames(inp_vid, opt_frms):
    # print(nb.get_num_threads())
    # nb.set_num_threads(multiprocessing.cpu_count())
    missing_frames_lst = []
    procsd_frms_lst = []
    summ_vect = []
    mse_score_arry = []
    tot_cnt = miss_cnt = 0
    vidin = ffmpegcv.VideoCapture(inp_vid)
    opt_frm = os.path.join(opt_frms, 'frame')
    opt_pth = opt_frm + str(tot_cnt) + '.jpg'
    res, prev = vidin.read()
    height, width, layers = prev.shape
    prev_gray = cv.cvtColor(prev,cv.COLOR_BGR2GRAY)
    cv.imwrite(opt_pth, prev)
    summ_vect.append(1)
    res, frame = vidin.read()

    while res is True:
        tot_cnt += 1
        curr = frame
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
        mse_score = mse(prev_gray, curr_gray)
        mse_score_arry.append(int(mse_score))
        frm_score = f'frame{tot_cnt}.jpg:{mse_score}'
        # print(frm_score)
        if mse_score>15: # can apply threshold 20, 30, 50
            prev = curr
            prev_gray = curr_gray
            procsd_frms_lst.append(frm_score)
            opt_pth = opt_frm + str(tot_cnt) + '.jpg'
            cv.imwrite(opt_pth,prev)
            summ_vect.append(1)
        else:
            missing_frames_lst.append(frm_score)
            miss_cnt += 1
            summ_vect.append(0)
        res, frame = vidin.read()
    set_mse = set(mse_score_arry)
    for scr in set_mse:
        print(f"frequency of {scr} in mse scores", mse_score_arry.count(scr))
    return miss_cnt, missing_frames_lst, procsd_frms_lst, summ_vect


def video_to_frames(inp_vid,opt_fldr):
    frm_cnt = 0
    diff_lst = []
    summry_vect = []
    opt_pth = os.path.join(opt_fldr, "ssim"+ inp_file[:-4])
    print(opt_pth)
    if os.path.exists(opt_pth) is True:
        shutil.rmtree(opt_pth)  # Delete folder and contents
    os.makedirs(opt_pth)
    miss_frm_cnt, diff_lst, prcsd_lst, summry_vect = ssim_frames(inp_vid, opt_pth)
    return miss_frm_cnt,diff_lst, prcsd_lst, opt_pth, summry_vect

def frames_to_video(inp_vid,opt_fldr):
    opt_pth = os.path.join(opt_fldr, "ssim"+ inp_file[:-4])
    list_fls = os.listdir(opt_pth)
    # Video writer to create .avi file
    video = cv2.VideoWriter(inp_vid, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    # Appending images to video
    for image in list_fls:
        video.write(cv2.imread(os.path.join(opt_pth, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

if __name__ == '__main__':
    base_fldr = '/DataSets/SumMe/'
    GTPATH = os.path.join(base_fldr, 'GT/')
    VIDEOPATH = os.path.join(base_fldr, 'videos/')
    inp_fldr = VIDEOPATH
    inp_fldr = inp_fldr.replace('\\', '/')
    op_fldr = os.path.join(inp_fldr,'output_l2')
    lst_mp4_files = glob.glob(f'{inp_fldr}/*.mp4')
    csv_path = os.path.join(inp_fldr,'record_time_summe.csv')
    with open(csv_path, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["VideoName", "FrameCount", "FrameExtractionTime"])

    for vid_file in lst_mp4_files:
      # print(vid_file)
      inp_file = os.path.basename(vid_file)
      # print(inp_file)
      vid_nm = inp_file.split('.')[0]
      # Load GroundTruth file
      gt_file = GTPATH + '/' + vid_nm + '.mat'
      gt_data = scio.loadmat(gt_file)
      user_score=gt_data.get('user_score')
      nFrames = user_score.shape[0]
      nFPS = gt_data.get('nFPS')
      start_time = datetime.now()
      ms_frame_cnt, frms_lst, prcsd_lst, op_pth,summary_vect = video_to_frames(vid_file,op_fldr)
      end_time = datetime.now()
      process_time = end_time-start_time
      print("processing time:",process_time)
      frames_to_video(vid_file,op_fldr)
      # record_log_data(csv_path,inp_file,op_pth,ms_frame_cnt,frms_lst,prcsd_lst, process_time)