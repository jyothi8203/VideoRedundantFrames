import scipy.io as scio
import warnings
import numpy as np
import matplotlib.pyplot as plt


def evaluateSummary(summary_selection, videoName, HOMEDATA, summ_lst, user_score):
    '''Evaluates a summary for video videoName (where HOMEDATA points to the ground truth file)
    f_measure is the mean pairwise f-measure used in Gygli et al. ECCV 2013
    NOTE: This is only a minimal version of the matlab script'''
    # Load GT file
    # gt_file=HOMEDATA+'/'+videoName+'.mat'
    # gt_data = scio.loadmat(gt_file)

    # user_score=gt_data.get('user_score')
    nFrames = user_score.shape[0];
    nbOfUsers = user_score.shape[1];

    # Check inputs
    if len(summary_selection) < nFrames:
        warnings.warn('Pad selection with %d zeros!' % (nFrames - len(summary_selection)))
        summary_selection.extend(np.zeros(nFrames - len(summary_selection)))

    elif len(summary_selection) > nFrames:
        warnings.warn('Crop selection (%d frames) to GT length' % (len(summary_selection) - nFrames))
        summary_selection = summary_selection[0:nFrames];

    summary_selection = summary_selection[0:nFrames];

    # Compute pairwise f-measure, summary length and recall
    summary_indicator = np.array(map(lambda x: (1 if x > 0 else 0), summary_selection));
    user_intersection = np.zeros((nbOfUsers, 1));
    user_union = np.zeros((nbOfUsers, 1));
    user_length = np.zeros((nbOfUsers, 1));
    for userIdx in range(0, nbOfUsers):
        gt_indicator = np.array(map(lambda x: (1 if x > 0 else 0), user_score[:, userIdx]))

        user_intersection[userIdx] = np.sum(gt_indicator * summary_indicator);
        user_union[userIdx] = sum(np.array(map(lambda x: (1 if x > 0 else 0), gt_indicator + summary_indicator)));

        user_length[userIdx] = sum(gt_indicator)

    recall = user_intersection / user_length;
    p = user_intersection / np.sum(summary_indicator);

    f_measure = []
    for idx in range(0, len(p)):
        if p[idx] > 0 or recall[idx] > 0:
            f_measure.append(2 * recall[idx] * p[idx] / (recall[idx] + p[idx]))
        else:
            f_measure.append(0)
    nn_f_meas = np.max(f_measure);
    f_measure = np.mean(f_measure);

    nnz_idx = np.nonzero(summary_selection)
    nbNNZ = len(nnz_idx[0])

    summary_length = float(nbNNZ) / float(len(summary_selection));

    recall = np.mean(recall);
    p = np.mean(p);

    return f_measure, summary_length

gt_score = [2.9,1.7,1.75,1.3,1.35,1.2,1.25,1.45,2.5,1.6,1.85,1.5,1.15,]
gt1 = {1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.8, 2.9, 2.95, 3.0, 3.05, 3.1, 3.2, 3.45, 3.75, 3.8}
gt1_l = list(gt1)
gt1_l5 = [vl/5 for vl in gt1_l]
gt1_nrm = [0 if 0<vl<0.5 else 1 for vl in gt1_l5]
gt1_l5nrm = [0 if 0<(vl/5)<0.5 else 1 for vl in gt1_l]

print(gt1_nrm)
print(gt1_l5nrm)