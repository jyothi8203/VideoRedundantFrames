import numpy as np
import warnings


def evaluateSummary(summary_selection, videoName, HOMEDATA, summ_lst, user_score):
    '''Evaluates a summary for video videoName (where HOMEDATA points to the ground truth file)
    f_measure is the mean pairwise f-measure used in Gygli et al. ECCV 2013
    NOTE: This is only a minimal version of the matlab script'''
    # Load GT file
    # gt_file=HOMEDATA+'/'+videoName+'.mat'
    # gt_data = scio.loadmat(gt_file)

    # user_score=gt_data.get('user_score')
    nFrames = user_score.shape[0]
    nbOfUsers = user_score.shape[1]

    # Check inputs
    if len(summary_selection) < nFrames:
        warnings.warn('Pad selection with %d zeros!' % (nFrames - len(summary_selection)))
        summary_selection.extend(np.zeros(nFrames - len(summary_selection)))

    elif len(summary_selection) > nFrames:
        warnings.warn('Crop selection (%d frames) to GT length' % (len(summary_selection) - nFrames))
        summary_selection = summary_selection[0:nFrames]

    summary_selection = summary_selection[0:nFrames]

    # Compute pairwise f-measure, summary length and recall
    summary_indicator = np.array([1 if x > 0 else 0 for x in summary_selection])
    user_intersection = np.zeros((nbOfUsers, 1))
    user_union = np.zeros((nbOfUsers, 1))
    user_length = np.zeros((nbOfUsers, 1))

    for userIdx in range(0, nbOfUsers):
        gt_indicator = np.array([1 if x > 0 else 0 for x in user_score[:, userIdx]])

        user_intersection[userIdx] = np.sum(gt_indicator * summary_indicator)
        user_union[userIdx] = sum([1 if x > 0 else 0 for x in (gt_indicator + summary_indicator)])

        user_length[userIdx] = sum(gt_indicator)

    recall = user_intersection / user_length
    p = user_intersection / np.sum(summary_indicator)

    f_measure = []
    for idx in range(0, len(p)):
        if p[idx] > 0 or recall[idx] > 0:
            f_measure.append(2 * recall[idx] * p[idx] / (recall[idx] + p[idx]))

    return f_measure  # Assuming f_measure is the intended return value