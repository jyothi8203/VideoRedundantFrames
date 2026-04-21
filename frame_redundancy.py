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
inp_vid = 'F:\\Jyothi\\projects\\DataSets\\VideoSummarization\\SumMe\\video\\ABase jumping.mp4'
# cap = cv2.VideoCapture(inp_vid)
# https://github.com/chenxinfeng4/ffmpegcv
# VideoWriter: Write a video file.
# VideoCapture: Read a video file.
# VideoCaptureNV: Read a video file by NVIDIA GPU.
# VideoCaptureQSV: Read a video file by Intel QuickSync Video.
# VideoCaptureCAM: Read a camera.
# VideoCaptureStream: Read a RTP/RTSP/RTMP/HTTP stream.
# VideoCaptureStreamRT: Read a RTSP stream (IP Camera) in real time low latency as possible.
# noblock: Read/Write a video file in background using mulitprocssing.
# toCUDA: Translate a video/stream as CHW/HWC-float32 format into CUDA device, >2x faster.
# Deeplearning pipeline.
#
# """
#           ——————————  NVIDIA GPU accelerating ⤴⤴ ———————
#           |                                              |
#           V                                              V
# video -> decode -> crop -> resize -> RGB -> CUDA:CHW float32 -> model
# """
# cap = ffmpegcv.toCUDA(
#     ffmpegcv.VideoCaptureNV(file, pix_fmt='nv12', resize=(W,H)),
#     tensor_format='chw')
#
# for frame_CHW_cuda in cap:
#     frame_CHW_cuda = (frame_CHW_cuda - mean) / std
#     result = model(frame_CHW_cuda)
##################33

# Parameters for Lucas-Kanade optical flow
# lk_params = dict(winSize=(50, 50),
#                  maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# # Create some random colors
# color = np.random.randint(0, 255, (100, 3))
#
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# if not ret:
#     exit()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
# frame_count = 0
# # Threshold for optical flow magnitude to consider a frame important
# flow_threshold = 2#5
# cnt = 0
# missing_frms = []
# inp_file = os.path.basename(inp_vid)
# opt_fldr = f'optcflw{inp_file[:-4]}'
# opt_frms = os.path.join(os.path.dirname(inp_vid),"output",opt_fldr)
# if os.path.exists(opt_frms) is True:
#     shutil.rmtree(opt_frms)  # Delete folder and contents
# os.makedirs(opt_frms)
# opt_frm = os.path.join(opt_frms, 'frame')
# opt_pth = opt_frm + str(cnt) + '.jpg'
# cv2.imwrite(opt_pth,old_frame)
# while True:
#     ret, frame = cap.read()
#     cnt += 1
#     if not ret:
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
#
#     if st.max() != 0:
#         good_new = p1[st == 1]
#         good_old = p0[st == 1]
#         # Calculate the magnitude of optical flow vectors
#         flow_magnitudes = np.sqrt(np.sum((good_new - good_old)**2, axis=1))
#         mean_flw_magn = np.mean(flow_magnitudes)
#
#         print("FM",flow_magnitudes,"Mean",mean_flw_magn)
#         # Check if the average flow magnitude exceeds the threshold
#         if mean_flw_magn > flow_threshold:
#             # Draw optical flow tracks
#             for i, (new, old) in enumerate(zip(good_new, good_old)):
#                 a, b = new.ravel()
#                 c, d = old.ravel()
#                 mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
#                 frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
#             img = cv2.add(frame, mask)
#             # Perform object detection here on the 'img' frame
#             cv2.imshow('Optical Flow and Object Detection', img)
#
#             cv2.imwrite(opt_frm+f'{cnt}.jpg',img)#img has the mask with track points
#             print(f'frame{cnt}.jpg')
#         else:
#             missing_frms.append(f'frame{cnt}.jpg:{mean_flw_magn}')
#         p0 = good_new.reshape(-1, 1, 2)
#
#     # Now update the previous frame and previous points
#     old_gray = gray.copy()
#     p0 = p1
#
#
#     k = cv2.waitKey(25) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()
# cap.release()
import ffmpegcv
import glob
import cv2 as cv
import os,shutil
from datetime import datetime
import csv
import scipy.io as scio
#import numba as nb
#import multiprocessing

#from numba import jit, cuda, vectorize, guvectorize
import skimage.metrics as metrics

MAX_THRES = 0.9
MIN_THRES = 0.7

#@jit(nopython=True,nogil=True,parallel=True,fastmath=True)
def mse(img1, img2):
    return ((img1 - img2) ** 2).mean()

# @vectorize
# def mse(img1, img2):
#     res = ((img1 - img2) ** 2).mean()
#     return res

def ssim_frames(inp_vid, opt_frms):
    nthrds = multiprocessing.cpu_count()
    os.environ['NUMBA_NUM_THREADS'] = str(nthrds)
    print(nb.get_num_threads())
    nb.set_num_threads(nthrds)
    missing_frames_lst = []
    procsd_frms_lst = []
    summ_vect = []
    mse_score_arry = []
    tot_cnt = miss_cnt = 0
    vidin = ffmpegcv.VideoCapture(inp_vid)
    opt_frm = os.path.join(opt_frms, 'frame')
    opt_pth = opt_frm + str(tot_cnt) + '.jpg'
    res, prev = vidin.read()
    prev_gray = cv.cvtColor(prev,cv.COLOR_BGR2GRAY)
    # cv.imwrite(opt_pth, prev)
    summ_vect.append(1)
    res, frame = vidin.read()

    while res is True:
        tot_cnt += 1
        curr = frame
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
        mse_score = mse(prev_gray, curr_gray)
        # mse_score = metrics.structural_similarity(prev_gray,curr_gray,multichannel=True)
        mse_score_arry.append(int(mse_score))
        frm_score = f'frame{tot_cnt}.jpg:{mse_score}'
        print(frm_score)
        if mse_score>15: # can apply threshold 20, 30, 50
            prev = curr
            prev_gray = curr_gray
            procsd_frms_lst.append(frm_score)
            opt_pth = opt_frm + str(tot_cnt) + '.jpg'
            # cv.imwrite(opt_pth,prev)
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
    opt_pth = os.path.join(opt_fldr, "ssim_cuda"+ inp_file[:-4])
    print(opt_pth)
    # if os.path.exists(opt_pth) is True:
    #     shutil.rmtree(opt_pth)  # Delete folder and contents
    # os.makedirs(opt_pth)
    miss_frm_cnt, diff_lst, prcsd_lst, summry_vect = ssim_frames(inp_vid, opt_pth)
    return miss_frm_cnt,diff_lst, prcsd_lst, opt_pth, summry_vect


if __name__ == '__main__':
    base_fldr = 'D:/Project/DataSets/SumMe/'
    GTPATH = os.path.join(base_fldr, 'GT/')
    VIDEOPATH = os.path.join(base_fldr, 'videos/')
    inp_fldr = VIDEOPATH
    # inp_fldr = 'D:/Projects/DataSets/SumMe/videos/'
    # inp_fldr = 'D:\\Project\\Datasets\\tvsumm'
    inp_fldr = inp_fldr.replace('\\', '/')
    op_fldr = os.path.join(inp_fldr,'output_l2')
    # lst_vid_files = os.listdir(inp_fldr)
    # lst_mp4_files = lst_vid_files
    lst_mp4_files = glob.glob(f'{inp_fldr}/*.mp4')#(f'{directory}/*.txt')
    csv_path = os.path.join(inp_fldr,'record_time_summe_l2_THRES_1.csv')
    with open(csv_path, 'w', newline='') as file:
      writer = csv.writer(file)
      # writer.writerow(["VideoName", "FrameCount", "OutputPath","Missing Frame Scores", "Processed Frames Score", "Min_Avg_Thrshld","Max_Avg_Thrshld","FrameExtractionTime"])
      writer.writerow(["VideoName", "FrameCount", "OutputPath", "Min_Avg_Thrshld","Max_Avg_Thrshld","FrameExtractionTime"])

    #lst_mp4_files = lst_mp4_files[-3:-1]
    for vid_file in lst_mp4_files:
      print(vid_file)
      inp_file = os.path.basename(vid_file)
      print(inp_file)
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
      record_log_data(csv_path,inp_file,op_pth,ms_frame_cnt, process_time)