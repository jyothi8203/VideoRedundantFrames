import csv
import os,shutil
# from summe import *
import numpy as np
import random
import scipy.io as scio #v7.2
import json
import h5py #v7.3
import glob

''' PATHS '''
HOMEDATA = '../GT/';
HOMEVIDEOS = '../videos/';
video_id_lst = []
nframes_lst = []
user_ann_lst = []
gt_scr_lst = []
ctgry_lst = []
vid_len_lst = []
nrml_gt_scr_lst = []
nrml_usr_scr_lst = []


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
            new_lst = ndarry_lst
            if key == 'nFrames' or key == 'video_duration' or key == 'FPS':
                if len(ndarry_lst) == 1:  # obj.shape[0] == 1
                    # if isinstance(ndarry_lst[0], int) or isinstance(ndarry_lst[0], float) or isinstance(ndarry_lst[0], str): # isinstance float
                    new_lst = ndarry_lst[0]
            elif key == 'gt_score' or key == 'polygt' or key == 'rectgt':
                if len(ndarry_lst) > 1:  # obj.shape[1] == 1
                    if isinstance(ndarry_lst[0], np.ndarray):
                        new_lst = [ele.tolist() for ele in ndarry_lst if isinstance(ele, np.ndarray)]
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

def process_dict(fl_pth):
    vid_data_dict = {}
    for i in range(50):
        vid_data_dict['video_id'] = video_id_lst[i]
        vid_data_dict['category'] = ctgry_lst[i]
        vid_data_dict['frames'] = int(nframes_lst[i])
        vid_data_dict['length'] = vid_len_lst[i]
        vid_data_dict['FPS'] = int(nframes_lst[i]/vid_len_lst[i])
        vid_data_dict['user_score'] = nrml_usr_scr_lst[i]
        vid_data_dict['gt_score'] = nrml_gt_scr_lst[i]
        file_name = os.path.join(fl_pth, video_id_lst[i] + '.json')
        open(file_name, 'w').write(json.dumps(vid_data_dict,indent=None,separators=(',',':')))
        vid_data_dict.clear()

def process_hdf5_ref(ref_data,key):
    if ref_data.shape[1] == 1:
        ref_data = ref_data[:,-1]
    elif ref_data.shape[0] == 1 and ref_data.shape[1] > 1:
        ref_data = ref_data.ravel()

    res_lst = list(ref_data)
    if key == 'video':
        val = bytearray(res_lst)
        video_id_lst.append(val.decode('utf-8'))
    elif key == 'gt_score':
        gt1_l5nrm = (ref_data >= 2.5).astype(np.int8)
        nrml_gt_scr_lst.append(gt1_l5nrm.tolist())
    elif key == 'user_anno':
        if len(res_lst) >= 2 and isinstance(res_lst[0], np.ndarray) and isinstance(res_lst[1], np.ndarray):
            nrm_arr = (ref_data >= 2.5).astype(np.int8)
            nrml_usr_scr_lst.append(nrm_arr.tolist())

    elif key == 'nframes':
        nframes_lst.append(res_lst[0])
    elif key == 'category':
        val = bytearray(res_lst)
        ctgry_lst.append(val.decode('utf-8'))
    elif key == 'length':
        vid_len_lst.append(res_lst[0])


def load_matlab_video_files(vid_name,json_pth):
    ext = vid_name.split('.')[1]
    if ext == 'mat':
        gt_file = vid_name
    gt_data = h5py.File(gt_file, 'r')
    print(gt_data.keys())
    json_fl = open(os.path.join(json_pth, 'refs.json'), 'w', encoding='utf-8')
    # print(gt_data['#refs#'])
    for sup_key in gt_data:
        json_fl.write(f'{sup_key}\n')
        for ky in gt_data[sup_key]:
            vl = gt_data[sup_key][ky]
            dt = np.array(vl).tolist()
            strd = f"'{ky}','{dt}'\n"
            # print(strd)
            if sup_key == '#refs#':
                json_fl.write(strd)
            if sup_key == 'tvsum50':
                ln_val = len(vl)
                for id in range(ln_val):
                    ele = vl[id]
                    # print(ele,type(ele))
                    if isinstance(ele, np.ndarray) and ele.size == 1:
                        ref = gt_data[sup_key][ky][id][0]
                        res = h5py.h5r.get_name(ref,gt_data.id)
                        nv_val = np.array(gt_data[res])
                        process_hdf5_ref(nv_val,ky)

        if sup_key == 'tvsum50':
            json_fl.close()
            process_dict(json_pth)


def load_summe_matlab_files(vid_name,wrtr):
    # Convert any non-JSON-serializable types (e.g., NumPy arrays)
    gt_file = vid_name
    gt_data = scio.loadmat(gt_file)
    json_compatible_data = convert_numpy_to_list(gt_data)

    with open('./matlab_data_files/'+vid_name + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_compatible_data, json_file)
    no_users = gt_data['user_score'].shape[1]
    wrtr.writerow([vid_name+'.mp4',json_compatible_data['nFrames'],json_compatible_data['segments'],json_compatible_data['all_userIDs'],json_compatible_data['video_duration'],json_compatible_data['FPS'],no_users])


def process_summe_mat(inp_fl):
    # Take a random video and create a random summary for it
    included_extenstions = ['mp4']

    videoList = [fn for fn in os.listdir(HOMEVIDEOS) if any([fn.endswith(ext) for ext in included_extenstions])]
    fldr_pth = './matlab_data_files'
    if os.path.exists(fldr_pth):
        print("exists")
    else:
        os.mkdir(fldr_pth)
    csv_path = fldr_pth+'/summe_videos_matlab_data.csv'
    csv_file = open(csv_path,'w',newline='')

    csv_writer = csv.writer(csv_file,dialect=csv.excel)
    csv_writer.writerow(['VideoName','Frames', 'Segments', 'All UserIDs','Video Duration', 'FPS', 'No Users'])

    load_summe_matlab_files(inp_fl,csv_writer)


def process_tvsumm_mat():
    root_pth = 'D:\\Project\\Datasets\\VideoSummaries\\tvsumm\\matlab'
    inp_mat = os.path.join(root_pth,'ydata-tvsum50.mat')
    json_fldr_pth = os.path.join(root_pth,'json_files')
    if os.path.exists(json_fldr_pth) is True:
        shutil.rmtree(json_fldr_pth)  # Delete folder and contents
    os.makedirs(json_fldr_pth)
    load_matlab_video_files(inp_mat, json_fldr_pth)

def write_tot_txt_json_file(g_fl,mt_fdr,jsn_data,wrtr):
    base_name = os.path.basename(g_fl)
    img_name = base_name.split('.')[0]
    json_nm = img_name+'.json'
    # print(os.path.join(mat_fldr,json_nm))
    with open(os.path.join(mt_fdr,json_nm), 'w', encoding='utf-8') as json_file:
        json.dump(jsn_data, json_file)
    if g_fl.find("Rectangular") != -1:
        wrtr.writerow([base_name,jsn_data['rectgt']])
    elif g_fl.find("Polygon") != -1:
        wrtr.writerow([base_name,jsn_data['polygt']])

def write_one_gt_json(g_fl, jsn_data):
    base_name = os.path.basename(g_fl)
    img_name = base_name.split('.')[0]
    json_nm = img_name + '.json'

    if g_fl.find("Polygon") != -1:
        # wrtr_mat.writerow([base_name, jsn_data['polygt']])
        poly_data = jsn_data['polygt']
        ornt = 'c'

    # Assuming format: xmin ymin xmax ymax text...
        slc_len = 6
        len_poly_data = len(poly_data)
        ent_lst = []
        for indx in range(0,len_poly_data,slc_len):
            bboxes = []
            ent_dict = {}
            data = poly_data[indx:indx+slc_len]
            if len(data) == 6:
                bboxes = [[data[1][0][i],data[3][0][i]] for i in range(len(data[1][0]))]
                texts = data[4]
                ornt = data[5]
                ent_dict["n_pts"] = len(bboxes)
                ent_dict["bbox"] = bboxes
                ent_dict["text"] = texts
                ent_dict["ornt"] = ornt
                ent_lst.append(ent_dict)
        tot_txt_dict[img_name] = {
            "num_entities": len(ent_lst),
            "entities": ent_lst
        }
    # Add to dictionary
    print(img_name)




def process_mat(gt_fl,mat_fldr,wrtr_mat):
    # gt_file = vid_name
    gt_data = scio.loadmat(gt_fl)
    # if gt_fl.find("Rectangular") != -1:
    #     print("Rectangular")
    json_compatible_data = convert_numpy_to_list(gt_data)

    # write_tot_txt_json_file(gt_fl,mat_fldr,json_compatible_data,wrtr_mat)
    write_one_gt_json(gt_fl,json_compatible_data)




def process_folder(pth,wrtr,jfile):
    for r, d, f in os.walk(pth):
        for folder in d:
            fl_pth = os.path.join(r, folder)
            fls_lst = glob.glob(f'{fl_pth}/*.mat')
            print(fls_lst)
            if len(fls_lst) > 0:
                mat_pth = fl_pth +'/matlab_files'
                if not os.path.exists(mat_pth):
                    os.mkdir(mat_pth)
                for mat_fl in fls_lst:
                    process_mat(mat_fl,mat_pth,wrtr)
                if len(tot_txt_dict.keys()) == len(fls_lst):
                    fl = open(os.path.join(fl_pth,jfile), 'w')
                    json.dump(tot_txt_dict, fl)
                    fl.close()
                    tot_txt_dict.clear()
            process_folder(fl_pth,wrtr,jfile)



def process_ground_truth_total_text():
    rt_pth = 'D:/Project/Datasets/scene_text/totaltext/GT/groundtruth_text/Groundtruth'
    # fls_lst = []
    # folder_list = []
    csv_path = rt_pth+'/tot_text_matlab_data.csv'
    csv_file = open(csv_path,'w',newline='')

    csv_writer = csv.writer(csv_file,dialect=csv.excel)
    csv_writer.writerow(['ImgName','polygon','BBOX','x','y'])
    json_file = 'tot_txt_poly_test_one_file.json'

    process_folder(rt_pth,csv_writer,json_file)



if __name__ == "__main__":
#     # process_summe_mat()
#     # process_tvsumm_mat()
    tot_txt_dict = {}
    process_ground_truth_total_text()
