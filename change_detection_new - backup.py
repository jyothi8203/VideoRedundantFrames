import glob
# import cv2 as cv
import ffmpegcv
import os,shutil
import numpy as np
from datetime import datetime
import csv,json
import scipy.io as scio
# from summe import evaluate_summary
import numba as nb
# import multiprocessing

# from skimage.metrics import structural_similarity as ssim
from sklearn import metrics
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from numba import njit, prange


@njit(fastmath=True)
def ssim_numba(img1, img2):
    # print("inside ssim_numba")
    C1 = 6.5025
    C2 = 58.5225

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    # print("result: ", numerator/denominator)
    return numerator / denominator


# @njit(parallel=True, fastmath=True)
@njit(fastmath=True)
def compute_ssim_batch(frames):
    # print("inside compute_ssim_batch")
    n_frames = len(frames)
    scores = np.empty(n_frames)
    # sim_score = 0.9
    for i in prange(n_frames-1):
        # print(i)
        sim_score = ssim_numba(frames[i], frames[i + 1])
        # print("sim_score", sim_score)
        scores[i] = sim_score
    # scores[n_frames-1] = sim_score
    return scores

@njit(fastmath=True)
def func_adapt_thres(scrs_arr):
    mean_threshold = np.mean(scrs_arr)
    min_thres = np.min(scrs_arr)
    max_thres = np.max(scrs_arr)

    std_thres = np.std(scrs_arr)
    adpt_thres = mean_threshold - (0.5 * std_thres)
    if adpt_thres >= 0.9:
        if 0.9 > min_thres > 0.7:
            adpt_thres = min_thres
        else:
            adpt_thres = 0.9
    elif adpt_thres <= 0.7:
        if 0.9 > max_thres > 0.7:
            adpt_thres = max_thres
        else:
            if 0.9 > mean_threshold > 0.7:
                adpt_thres = mean_threshold
            else:
                adpt_thres = 0.7
    return adpt_thres

@njit(fastmath=True)
def adaptive_threshold(scores_arr,wind_len):
    idx = 0
    miss_cnt = 0
    adapt_thrs_arr_x = []
    adapt_thrs_arr_y = []
    summ_vect_arr = []
    adpt_thres = scores_arr[idx]

    indx_len = scores_arr.shape[0] - 1
    scores_arr[indx_len] = adpt_thres
    for i,sim_score in enumerate(scores_arr):
        if i > 0 and (i % wind_len ==  0 or i == indx_len):
            ssim_score_aary = scores_arr[idx:i]
            adapt_thres = func_adapt_thres(ssim_score_aary)
            adapt_thrs_arr_x.append(i + 1)
            adapt_thrs_arr_y.append(np.round(adapt_thres, 2))
            idx += wind_len
        # ssim_score_aary = np.zeros(len_thrshold)

        if sim_score <= adpt_thres:
            summ_vect_arr.append(1)
        else:
            miss_cnt += 1
            summ_vect_arr.append(0)
    return miss_cnt,adapt_thrs_arr_x,adapt_thrs_arr_y,summ_vect_arr

class ChangeDetection:
    def __init__(self):
        self.MAX_THRES = 0.9
        self.MIN_THRES = 0.7
        self.ALPHA = 0.5
        self.opt_path = "opt_path"
        self.adapt_thrs_arry = []
        self.algor = 1
        self.every = 2
        self.nFPS = 25
        self.nFrames = 1000
        self.adapt_thrs_ar_x = []
        self.missing_frames_lst = []
        self.procsd_frms_lst = []
        self.summ_vect = []
        self.TEST = True

    def orig_frames(self,inp_vid,opt_frms):
        # cap = cv.VideoCapture(inp_vid)
        cap = ffmpegcv.VideoCapture(inp_vid)
        cnt = 0
        res, frame = cap.read()
        opt_frm = os.path.join(opt_frms, 'frame')
        writer = ffmpegcv.VideoWriter(opt_frm, 'mp4v')#, cap.fps,#size=cap.size)
        while res:
            opt_pth = opt_frm + str(cnt) + '.jpg'
            writer.write(opt_pth, frame)
            cnt += 1
            res, frame = cap.read()
        return cnt - 1

    def optc_flw_frames(self,inp_vid, opt_frms):
        missing_frames = []
        lk_params = dict(winSize=(50, 50),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        cap = cv.VideoCapture(inp_vid)
        cnt = 0
        ret, frame = cap.read()
        # Take first frame and find corners in it
        if not ret:
            exit()
        color = np.random.randint(0, 255, (100, 3))
        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame)
        frame_count = 0
        # Threshold for optical flow magnitude to consider a frame important
        flow_threshold = 2  # 5

        opt_frm = os.path.join(opt_frms, 'frame')
        opt_pth = opt_frm + str(cnt) + '.jpg'
        prev = frame
        prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.summ_vect.append(1)
        # cv.imwrite(opt_pth, prev)

        while True:
            ret, frame = cap.read()
            cnt += 1
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Calculate optical flow - p1 st err - rate of change of corner pixels(err)
            p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

            if st.max() != 0:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                # Calculate the magnitude of optical flow vectors
                flow_magnitudes = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
                mean_flw_magn = np.mean(flow_magnitudes)

                # print("FM",flow_magnitudes,"Mean",mean_flw_magn)
                # Check if the average flow magnitude exceeds the threshold
                if mean_flw_magn > flow_threshold:
                    # Draw optical flow tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                    img = cv.add(frame, mask)
                    # Perform object detection here on the 'img' frame
                    # cv.imshow('Optical Flow and Object Detection', img)
                    self.summ_vect.append(1)
                    # cv.imwrite(opt_frm+f'{cnt}.jpg',img)#img has the mask with track points
                    # print(f'frame{cnt}.jpg')
                else:
                    missing_frames.append(f'frame{cnt}.jpg:{mean_flw_magn}')
                    self.summ_vect.append(0)

            else:
                self.summ_vect.append(0)
                # p0 = good_new.reshape(-1, 1, 2)

            # Now update the previous frame and previous points
            prev_gray = gray.copy()
            p0 = p1
        return len(missing_frames)


    def ssim_frames(self,inp_vid,opt_frms):
        min_adpt = max_adpt = self.MAX_THRES
        len_thrshold = self.nFPS * self.every
        ssim_score_aary = np.zeros(len_thrshold)
        indx = 0
        cap = ffmpegcv.VideoCapture(inp_vid, pix_fmt='gray')
        tot_cnt = miss_cnt = 0
        res, frame = cap.read()
        prev = np.squeeze(frame)

        res, frame = cap.read()
        curr = np.squeeze(frame)
        tot_cnt += 1
        self.summ_vect.append(1)

        # sim_score = ssim(prev, curr)
        sim_score = 0.9
        sim_score = ssim_numba(prev, curr)
        adapt_threshold = sim_score
        prev = curr
        self.summ_vect.append(0)

        ssim_score_aary[indx] = sim_score
        indx += 1
        self.adapt_thrs_arry.append(np.clip(adapt_threshold, self.MIN_THRES, self.MAX_THRES))  # round 2 decimals
        self.adapt_thrs_ar_x.append(tot_cnt)
        res, frame = cap.read()  # 3 frame for loop continuity

        while res is True:
            tot_cnt += 1
            curr = np.squeeze(frame)
            # sim_score = ssim(prev, curr)
            sim_score = ssim_numba(prev, curr)
            ssim_score_aary[indx] = sim_score
            indx += 1

            if tot_cnt%len_thrshold == 0 or tot_cnt == (self.nFrames-1):
                # mean_threshold = np.mean(ssim_score_aary)
                # max_thres = np.max(ssim_score_aary)

                self.adapt_thrs_ar_x.append(tot_cnt+1)
                adapt_threshold = func_adapt_thres(ssim_score_aary)
                prev = curr #enhancement at window frame
                self.adapt_thrs_arry.append(np.round(adapt_threshold, 2))
                indx = 0
                ssim_score_aary = np.zeros(len_thrshold)
            if sim_score<=adapt_threshold:
                prev = curr
                self.summ_vect.append(1)
            else:
                miss_cnt += 1
                self.summ_vect.append(0)
            res, frame = cap.read()
        cap.release()
        return miss_cnt

    def read_all_frames(self,inpv):
        cap = ffmpegcv.VideoCapture(inpv, pix_fmt='gray')
        idx = 0

        res, frame = cap.read()
        frm_arr = []

        while res is True:
            curr = np.squeeze(frame)
            frm_arr.append(curr)
            res, frame = cap.read()
            idx += 1
        cap.release()
        return frm_arr

    def video_to_frames(self,inp_vid):
        miss_frm_cnt = 0
        options_lst = ["","ssim","orig"]
        opt_pth = os.path.join(self.opt_path, options_lst[self.algor] + inp_file[:-4])
        print(opt_pth)
        if self.TEST:
            frame_array = self.read_all_frames(inp_vid)
            val = self.every*self.nFPS
            # print("call compute")
            scores = compute_ssim_batch(frame_array)
            # print("before adaptive")
            miss_frm_cnt,self.adapt_thrs_ar_x,self.adapt_thrs_arry,self.summ_vect = adaptive_threshold(scores,val)
        else:
            miss_frm_cnt = self.ssim_frames(inp_vid, opt_pth)
        return miss_frm_cnt, opt_pth


    # @nb.jit(cache=True)
    def record_log_data(self,csv_path,inp_vid,actual_frm_cnt,missing_frame_cnt,acc, accM, gt_acc, mn_fscr,mx_fscr,ptime):
      with open(csv_path, 'a', newline='') as file:
          writer = csv.writer(file)
          mn_fscr = float(f"{mn_fscr:.6f}")
          mx_fscr = float(f"{mx_fscr:.6f}")
          acc = float(f"{acc:.6f}")
          gt_acc = float(f"{gt_acc:.6f}")
          writer.writerow([inp_vid,actual_frm_cnt,missing_frame_cnt, acc, accM, gt_acc, mn_fscr, mx_fscr,ptime])

    def plot_adpt_thrs(self,video_name,ax1):
        global bClrFlg

        x = self.adapt_thrs_ar_x
        y = self.adapt_thrs_arry

        # plt.xlabel('Number of Frames')
        # plt.ylabel('Adapt Threshold')
        # fig, ax = plt.subplots()
        ax1.plot(x, y,label=video_name,marker='.')

        # 1. Set the interval to 0.1
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

        # 2. Format labels to 2 decimal places
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax1.set_xlabel('Number of Frames')
        ax1.set_ylabel('Threshold')
        ax1.set_title('Adapt Threshold Plot')
        ax1.legend()
        # plt.plot(x,y,label=video_name)#, marker='o')
        # plt.legend()
        # plt.show()
        # plt.savefig(video_name+'_adapt2.png')

    def evaluate_summary(self, usr_scr,grnd_scr):
        '''Evaluates a summary for video videoName (where HOMEDATA points to the ground truth file)
        f_measure is the mean pairwise f-measure used in Gygli et al. ECCV 2013
        NOTE: This is only a minimal version of the matlab script'''
        n_frames = usr_scr.shape[1]
        nb_users = usr_scr.shape[0]
        vect_len = self.nFrames
        user_acc = np.zeros((nb_users, 1));
        prec_arr = prec_arr1 = np.zeros((nb_users, 1));
        rec_arr = rec_arr1 = np.zeros((nb_users, 1));
        acc_arr = acc_arr1 = np.zeros((nb_users, 1));
        f1sr_arr = f1sr_arr1 = np.zeros((nb_users, 1));
        conf_mat = []
        pref_mat = []
        if len(self.summ_vect) < vect_len:
            self.summ_vect.extend(np.zeros(vect_len - len(self.summ_vect)))

        elif len(self.summ_vect) > self.nFrames:
            self.summ_vect = self.summ_vect[0:vect_len]

        summary_selection = self.summ_vect[0:vect_len]
        sum_ind = summary_selection.copy()
        acc_mat = 0.0
        user_acc = np.zeros((nb_users, 1));
        for userIdx in range(0, nb_users):
            gt_indicator = np.array(usr_scr[userIdx])
            gt_ind = np.clip(gt_indicator, 0, 1)
            gt_ind = (gt_indicator > 0).astype(int)

            conf_mat = metrics.confusion_matrix(gt_ind, sum_ind)
            pref_mat = metrics.precision_recall_fscore_support(gt_ind, sum_ind, average=None)
            acc_mat = metrics.accuracy_score(gt_ind, sum_ind)
          ##vect_len
            n_tp_matches = [True if sum_ind[id] == 1 and sum_ind[id] == gt_ind[id] else False for id in range(vect_len)]
            n_tp = n_tp_matches.count(True)
            n_fp_matches = [True if sum_ind[id] == 1 and gt_ind[id] == 0 else False for id in
                            range(vect_len)]  # (gt_ind == 0 and sum_ind == 1)
            n_fp = n_fp_matches.count(True)
            n_fn_matches = [True if sum_ind[id] == 0 and gt_ind[id] == 1 else False for id in
                            range(vect_len)]  # (gt_ind == 1 and sum_ind == 0)
            n_fn = n_fn_matches.count(True)
            n_tn_matches = [True if sum_ind[id] == 0 and sum_ind[id] == gt_ind[id] else False for id in range(vect_len)]
            n_tn = n_tn_matches.count(True)
            acc = prec = rec = f1_scr = 0
            if (n_tp + n_tn + n_fn + n_fp) > 0:
                acc = (n_tp+n_tn) / (n_tp + n_tn + n_fn + n_fp)
            if n_tp + n_fp > 0:
                prec = n_tp / (n_tp + n_fp)
            if n_fp + n_fn > 0:
                rec = n_tp / (n_tp + n_fn)
            if prec + rec > 0:
                f1_scr = 2 * prec * rec / (prec + rec)
            acc_arr[userIdx] = acc
            prec_arr[userIdx] = prec
            rec_arr[userIdx] = rec
            f1sr_arr[userIdx] = f1_scr
            matches = (gt_ind==sum_ind)
            user_acc[userIdx] = np.mean(matches)
            acc_arr1[userIdx] = acc_mat  # [1][0]
            prec_arr1[userIdx] = pref_mat[0][1]
            rec_arr1[userIdx] = pref_mat[1][1]
            f1sr_arr1[userIdx] = pref_mat[2][1]
            ##
        acc = np.max(user_acc)
        acc_mn = np.mean(user_acc)
        ground = (grnd_scr > 0).astype(int)
        ground_matches = (ground == sum_ind)
        ground_acc = np.mean(ground_matches)
        gacc = metrics.accuracy_score(ground, sum_ind)
        print("Ground Accuracy: ", gacc)
        return acc,acc_m,ground_acc,np.mean(f1sr_arr1), np.max(f1sr_arr1)

if __name__ == '__main__':
    lst_indx = 1
    cdobj = ChangeDetection()
    # base_fldr = 'D:/Project/DataSets/SumMe/'
    base_fldr = 'D:\Project\Datasets\VideoSummaries\\tvsumm'
    # Load GroundTruth file
    VIDEOPATH = os.path.join(base_fldr, 'videos/')
    GTPATH = ''
    gt_data = {}
    if base_fldr.endswith('SumMe'):
        GTPATH = os.path.join(base_fldr, 'GT/')
    elif base_fldr.endswith('tvsumm'):
        GTPATH = os.path.join(base_fldr, 'matlab/json_files')

    inp_fldr = VIDEOPATH
    # inp_fldr = 'D:/Projects/DataSets/SumMe/videos/'
    # inp_fldr = 'D:\\Project\\Datasets\\tvsumm'
    inp_fldr = inp_fldr.replace('\\', '/')
    op_fldr = os.path.join(inp_fldr,'output_graphs')
    cdobj.algor = 1
    cdobj.opt_path = op_fldr
    lst_mp4_files = glob.glob(f'{inp_fldr}/*.mp4')
    # csv_path = os.path.join(inp_fldr,'record_summe_2sec_FstMth_f1_acc.csv')
    csv_path = os.path.join(inp_fldr,'record_tvsumm_2sec_FstMth_f1_acc.csv')

    with open(csv_path, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["VideoName", "FrameCount", "MissCount", "Accuracy", "AccuracyM", "GTAcc", "F1_score","Time"])
    ln_cnt = 4
    # fig, ax = plt.subplots()
    user_score = np.array(1)
    ground_data = np.array(1)

    for vid_file in lst_mp4_files:#[-6:-3]:
      # inp_pth = os.path.join(inp_fldr,inp_vid)
      print(vid_file)
    # print("Folders\n",inp_fldr,opt_fldr)
      inp_file = os.path.basename(vid_file)
      print(inp_file)
      vid_nm = inp_file.split('.')[0]
      if base_fldr.endswith('SumMe'):
        gt_file = GTPATH + '/' + vid_nm + '.mat'
        gt_data = scio.loadmat(gt_file)
        user_score=gt_data.get('user_score')
        user_score = user_score.T
        ground_data=gt_data.get('gt_score')

      elif base_fldr.endswith('tvsumm'):
        gt_file = GTPATH + '/' + vid_nm + '.json'
        gt_data = json.load(open(gt_file))
        user_score = np.array(gt_data.get('user_score'))
        ground_data = np.array(gt_data.get('gt_score'))

      cdobj.nFrames = ground_data.shape[0]
      nFPS = gt_data.get('FPS')
      if isinstance(nFPS,list):
          cdobj.nFPS = int(gt_data.get('FPS')[0][0])
      else:
          cdobj.nFPS = nFPS
      start_time = datetime.now()
      # print("StartTime",start_time)
      ms_frame_cnt,op_pth = cdobj.video_to_frames(vid_file)

      end_time = datetime.now()
      # print("EndTime",end_time)
      # print("Total frame count",frame_cnt)
      process_time = end_time-start_time
      # print("processing time:",process_time)
      # if os.path.exists(GTPATH) is True:
      accuracy, acc_ma, grnd_trt_acc, mn_f1_score, mx_f1_score = cdobj.evaluate_summary(user_score,ground_data)
      cdobj.record_log_data(csv_path,vid_nm, cdobj.nFrames, ms_frame_cnt, accuracy, acc_ma, grnd_trt_acc, mn_f1_score, mx_f1_score,process_time)
      print(vid_nm,accuracy,"GTACC",grnd_trt_acc,mx_f1_score)
      # cdobj.plot_adpt_thrs(vid_nm,ax)
      #
      # if (lst_indx % ln_cnt) == 0:
      #   # bClrFlg = True
      #   plt.savefig(vid_nm+'_plot.png')
      #   #
      #   fig.savefig(vid_nm+'_fig.png')
      #   # fig.clf()
      #   fig.clear()
      #   plt.clf()
      #
      #   fig,ax = plt.subplots()

      cdobj.summ_vect.clear()
      cdobj.adapt_thrs_ar_x.clear()
      cdobj.adapt_thrs_arry.clear()
      lst_indx += 1