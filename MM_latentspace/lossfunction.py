import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac   #CPdecomposition


#The total loss function and the loss function for the subproblem
def loss_function(X_t, theta_v, X_v, B_v, P, B_s, H, X_1, lam1, lam2, lam3, lam4, sample_num, feature_D, view_num, r, W_mode1):
    H = H.reshape(view_num, r)
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    # for i in range(view_num):
    #     B_v[i] = B_v[i].reshape(r, feature_D)
    loss0 = 0
    for i in range(view_num):
        loss0 = loss0 + theta_v[i] * (np.linalg.norm(X_t[i] - np.dot(P, B_v[i])) ** 2)
    kron = np.zeros((len(B_s) * len(H), len(B_s[0])), dtype=float)
    for i in range(len(B_s[0])):
        kron[:, i] = np.kron(H[:, i], B_s[:, i])
    kha = np.dot(P, kron.T)
    loss1 = lam1 * (np.linalg.norm(W_mode1*(X_1 - kha))**2)
    loss2 = lam2 * np.linalg.norm(P, ord='nuc')
    loss3 = 0
    for i in range(len(H)):
        loss3 = loss3 + np.linalg.norm(H[i])
    loss3 = lam4 * loss3
    return loss0 + loss1 + loss2 + loss3


def lossfunction_H(H, X_3, P, B_s, lam1, lam4, sample_num, feature_D, view_num, r, W_mode3):
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    H = H.reshape(view_num, r)
    matrix_F = np.zeros((len(B_s) * len(P), len(P[0])), dtype=float)
    for i in range(len(P[0])):
        matrix_F[:, i] = np.kron(B_s[:, i], P[:, i])
    loss1 = lam1 * (np.linalg.norm(W_mode3*(X_3 - np.dot(H, matrix_F.T)))**2)
    loss2 = 0
    for i in range(len(H)):
        loss2 = loss2 + np.linalg.norm(H[i])
    loss2 = lam4 * loss2
    return loss1 + loss2


def lossfunction_P(P, theta_v, X_t, B_v, X_1, B_s, H, lam1, z, rho, Lambda, sample_num, feature_D, view_num, r, W_mode1):
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    H = H.reshape(view_num, r)
    # for i in range(view_num):
    #     B_v[i] = B_v[i].reshape(r, feature_D)
    loss0 = 0
    for i in range(view_num):
        loss0 = loss0 + theta_v[i] * (np.linalg.norm(X_t[i] - np.dot(P, B_v[i])) ** 2)
    matrix_D = np.zeros((len(B_s) * len(H), len(B_s[0])), dtype=float)
    for i in range(len(B_s[0])):
        matrix_D[:, i] = np.kron(H[:, i], B_s[:, i])
    loss1 = lam1 * (np.linalg.norm(W_mode1*(X_1 - np.dot(P, matrix_D.T))) ** 2)
    loss2 = np.sum(Lambda * (P - z))
    loss3 = 0.5 * rho * (np.linalg.norm(P - z) ** 2)
    return loss0 + loss1 + loss2 + loss3


def lossfunction_B_s(B_s, theta_v, B_v, X_2, P, H, lam1, lam3, sample_num, feature_D, view_num, r, W_mode2):
    # for i in range(view_num):
    #     B_v[i] = B_v[i].reshape(r, feature_D)
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    H = H.reshape(view_num, r)
    matrix_E = np.zeros((len(H) * len(P), len(P[0])), dtype=float)
    for i in range(len(P[0])):
        matrix_E[:, i] = np.kron(H[:, i], P[:, i])
    loss1 = lam1 * (np.linalg.norm(W_mode2*(X_2 - np.dot(B_s, matrix_E.T))) ** 2)
    loss2 = 0
    # for i in range(view_num):
    #     loss2 = loss2 + theta_v[i] * (np.linalg.norm(B_s - B_v[i].T) ** 2)
    loss2 = lam3 * loss2
    return loss1 + loss2


def lossfunction_B_v(B_vi, theta_vi, X_ti, B_s, P, lam3, sample_num, feature_D, feature_Di, r):
    B_vi = B_vi.reshape(r, feature_Di)
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    loss0 = theta_vi * (np.linalg.norm(X_ti - np.dot(P, B_vi)) ** 2)
    return loss0






