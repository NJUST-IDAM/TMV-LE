import scipy.io as sio
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.cluster import KMeans
from sklearn import neighbors


def read_mat(url):
    data = sio.loadmat(url)
    return data


def M_contrust(indices, k):
    M = np.zeros((len(indices), k))
    for i in range(len(indices)):
        for j in range(k):
            M[i][indices[i][j]] = 1/len(indices[0])
    return M


def lossfunction_m(M_i, feature_m, x_i, rho_m):
    loss1 = np.linalg.norm(x_i - np.dot(M_i, feature_m))**2
    one_m = np.ones((len(M_i), 1))
    loss2 = rho_m * ((np.dot(M_i, one_m) - 1) ** 2)
    return loss1 + loss2


def gradient_m(M_i, feature_m, x_i, rho_m):
    gd_m1 = 2 * (-np.dot(x_i, feature_m.T) + np.dot(np.dot(M_i, feature_m), feature_m.T))
    gd_m2 = 2 * rho_m * M_i
    return gd_m1 + gd_m2


if __name__ == "__main__":
    data_x = read_mat(r"../Datasets/all_mat/Emotions/2/Emotions_P.mat")
    feature_p = data_x["P"]
    k = 20
    neighx = neighbors.NearestNeighbors(n_neighbors=k)
    neighx.fit(feature_p)
    distances, indices = neighx.kneighbors(feature_p)
    M_neik = indices[:, 1:]
    samples_num = len(feature_p)
    rho_m = 10
    M_index = np.zeros((samples_num, k-1), dtype=float)
    for i in range(samples_num):
        feature_m = np.zeros((k-1, len(feature_p[0])), dtype=float)
        M_i = np.zeros((1, k-1), dtype=float)
        x_i = feature_p[i]
        for j in range(len(M_neik[0])):
            feature_m[j] = feature_p[M_neik[i][j]]
        result_m = fmin_l_bfgs_b(lossfunction_m, M_i, gradient_m,
                                 args=(feature_m, x_i, rho_m),
                                 pgtol=0.01, maxiter=10)
        M_i = result_m[0]
        M_index[i] = M_i
    M = np.zeros((samples_num, samples_num), dtype=float)
    for i in range(samples_num):
        for j in range(len(M_index[0])):
            M[i][M_neik[i][j]] = M_index[i][j]
    print(M)
    mat_path = r'../Datasets/all_mat/Emotions/2/Emotions_M.mat'
    sio.savemat(mat_path, {'M': M})
