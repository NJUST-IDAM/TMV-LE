import scipy.io as sio
from scipy.optimize import fmin_l_bfgs_b
from MM_latentspace.Gradient import *
from MM_latentspace.lossfunction import *
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac


def read_mat(url):
    data = sio.loadmat(url)
    return data


if __name__ == "__main__":
    data = sio.loadmat(r'../Datasets/all_mat/Emotions/Emotions_10_fold_origin_2_views.mat')
    feature = data['data']
    X_t = []
    X_t.append(feature[0][0])
    X_t.append(feature[1][0])
    data1 = read_mat(r"../Datasets/all_mat/Emotions/Emotions_mode1.mat")
    data2 = read_mat(r"../Datasets/all_mat/Emotions/Emotions_mode2.mat")
    data3 = read_mat(r"../Datasets/all_mat/Emotions/Emotions_mode3.mat")
    W_mode1 = (read_mat(r"../Datasets/all_mat/Emotions/W_mode1.mat"))["W"]
    W_mode2 = (read_mat(r"../Datasets/all_mat/Emotions/W_mode2.mat"))["W"]
    W_mode3 = (read_mat(r"../Datasets/all_mat/Emotions/W_mode3.mat"))["W"]
    X_1 = data1["feature"]
    X_2 = data2["feature"]
    X_3 = data3["feature"]
    r = 60 # Emotions {30,40,50,60}, Yeast {30,40,50,60},  Corel5k {100,300,500,700,900},  PASCAL{100,300,500,700,900}
    sample_num = len(X_1)
    feature_D = len(X_2)
    view_num = len(X_3)
    X_v = np.zeros((view_num, sample_num, feature_D), dtype=float)
    P = np.random.rand(sample_num, r)
    z = P.copy()
    B_s = np.random.rand(feature_D, r)
    H = np.random.rand(view_num, r)
    P = P.reshape(1, sample_num * r)
    B_s = B_s.reshape(1, feature_D * r)
    H = H.reshape(1, view_num * r)
    Lambda = np.zeros((sample_num, r), dtype=float)
    theta_v = []
    for i in range(view_num):
        theta_v.append(1/view_num)
        X_v[i] = (X_1.T[np.arange(i*feature_D, (i+1)*feature_D)]).T
    B_v = []
    for i in range(view_num):
        B_v.append(np.random.random((r, len(X_t[i][0]))))
    lam1 = 10 ** -6  # {10**-1,10**-2,10**-3,10**-6}
    lam2 = 10 ** -6  # {10**-1,10**-2,10**-3,10**-6}
    lam3 = 10 ** -6  # {10**-1,10**-2,10**-3,10**-6}
    lam4 = 10 ** -6  # {10**-1,10**-2,10**-3,10**-6}
    rho = 10 ** -6
    rho_max = 10 ** 6
    beta = 1.1
    loss = loss_function(X_t, theta_v, X_v, B_v, P, B_s, H, X_1, lam1, lam2, lam3, lam4, sample_num, feature_D, view_num, r, W_mode1)
    print(loss)
    for i in range(80):
        print(i, "-" * 20)
        result_P = fmin_l_bfgs_b(lossfunction_P, P, gradient_P, args=(theta_v, X_t, B_v, X_1, B_s, H, lam1, z, rho, Lambda, sample_num, feature_D, view_num, r, W_mode1),
                                   pgtol=0.001, maxiter=20)
        P = result_P[0]
        result_B_s = fmin_l_bfgs_b(lossfunction_B_s, B_s, gradient_B_s, args=(theta_v, B_v, X_2, P, H, lam1, lam3, sample_num, feature_D, view_num, r, W_mode2),
                                 pgtol=0.001, maxiter=20)
        B_s = result_B_s[0]
        for i in range(view_num):
            feature_Di = len(X_t[i][0])
            B_vi = B_v[i].reshape(1, -1)
            result_B_v = fmin_l_bfgs_b(lossfunction_B_v, B_vi, gradient_B_v,
                                       args=(theta_v[i], X_t[i], B_s, P, lam3, sample_num, feature_D, feature_Di, r),
                                       pgtol=0.001, maxiter=10)
            B_v[i] = result_B_v[0].reshape(r, feature_Di)
        result_H = fmin_l_bfgs_b(lossfunction_H, H, gradient_H, args=(X_3, P, B_s, lam1, lam4, sample_num, feature_D, view_num, r, W_mode3),
                                 pgtol=0.001, maxiter=10)
        H = result_H[0]
        z = update_z(lam4, z, Lambda, P, rho, sample_num, r)
        Lambda = update_Lambda(z, P, Lambda, rho, sample_num, r)
        loss_new = loss_function(X_t, theta_v, X_v, B_v, P, B_s, H, X_1, lam1, lam2, lam3, lam4, sample_num, feature_D, view_num, r, W_mode1)
        if abs(loss - loss_new) < 10 ** -4 or loss_new > loss:
            break
        rho = np.min([rho * beta, rho_max])
        loss = loss_new
        print(loss)
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    H = H.reshape(view_num, r)
    mat_path_P = r'../Datasets/all_mat/Emotions/2/Emotions_P.mat'
    # mat_path_B = r'../Datasets/all_mat/Pascal/1/Pascal_B.mat'
    # mat_path_H = r'../Datasets/all_mat/Pascal/1/Pascal_H.mat'
    sio.savemat(mat_path_P, {'P': P})
    # sio.savemat(mat_path_B, {'B': B})
    # sio.savemat(mat_path_H, {'H': H})
