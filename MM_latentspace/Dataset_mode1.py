import scipy.io
import numpy as np
import h5py


if __name__ == "__main__":  #Get mode-1 for multi-view data and the weight tensor used for the completion of tensors
    data = scipy.io.loadmat(r'../Datasets/all_mat/Emotions/Emotions_10_fold_origin_2_views.mat')
    # data = h5py.File(r'../Datasets/all_mat/Pascal/Pascal_10_fold_origin_5_views.mat','r')
    # feature = data['data'][:]
    # feature_v1 = (data[feature[0][0]].value).T
    # feature_v2 = (data[feature[0][4]].value).T
    #feature_v3 = (data[feature[0][4]].value).T
    #feature_v4 = (data[feature[0][4]].value).T
    feature = data['data']
    feature_v1 = np.ones_like(feature[0][0])
    feature_v2 = np.ones_like(feature[1][0])
    # feature_v1 = feature[0][0]
    # feature_v2 = feature[1][0]
    # feature_v3 = feature[2][0]
    # feature_v4 = feature[3][0]
    # feature_v5 = feature[4][0]
    d_1 = len(feature_v1[0])
    d_2 = len(feature_v2[0])
    #d_3 = len(feature_v3[0])
    d_sum = d_1
    n_sample = len(feature_v1)
    f_1_behind = np.zeros((n_sample, d_2), dtype=float)
    feature_v1 = (np.hstack((feature_v1, f_1_behind))).T
    feature_v1 = feature_v1.reshape((1, -1))
    f_2_front = np.zeros((n_sample, d_1), dtype=float)
    # f_2_behind = np.zeros((n_sample, d_3), dtype=float)
    feature_v2 = (np.hstack((f_2_front, feature_v2))).T
    feature_v2 = feature_v2.reshape((1, -1))
    #f_3_front = np.zeros((n_sample, d_1 + d_2), dtype=float)
    #f_3_behind = np.zeros((n_sample, d_4), dtype=float)
    #feature_v3 = (np.hstack((f_3_front, feature_v3))).T
    #feature_v3 = feature_v3.reshape((1, -1))
    # feature_mode = np.hstack((feature_v1, feature_v2))
    feature_mode = np.vstack((feature_v1, feature_v2))
    mat_path = r'../Datasets/all_mat/Emotions/W_mode3.mat'
    scipy.io.savemat(mat_path, {'W': feature_mode})

