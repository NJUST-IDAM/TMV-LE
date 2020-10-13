import numpy as np


def update_z(lam4, z, Lambda, P, rho, sample_num, r):  #Derivatives of variables in subproblems
    P = P.reshape(sample_num, r)
    u, sigma, vt = np.linalg.svd(P + Lambda / rho)
    sigma_new = [s if s-(lam4/rho) > 0 else 0 for s in sigma]
    temp = np.diag(sigma_new)
    height, width = z.shape
    if len(sigma) < width:
        temp = np.c_[temp, np.zeros([len(sigma), width-len(sigma)])]
    if len(sigma) < height:
        temp = np.r_[temp, np.zeros([height-len(sigma), width])]
    z_new = np.dot(np.dot(u, temp), vt)
    return z_new


def update_Lambda(z, P, Lambda, rho, sample_num, r):
    P = P.reshape(sample_num, r)
    Lambda_new = Lambda + rho * (P - z)
    return Lambda_new


def gradient_H(H, X_3, P, B_s, lam1, lam4, sample_num, feature_D, view_num, r, W_mode3):
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    H = H.reshape(view_num, r)
    matrix_F = np.zeros((len(B_s) * len(P), len(P[0])), dtype=float)
    for i in range(len(P[0])):
        matrix_F[:, i] = np.kron(B_s[:, i], P[:, i])
    gd_H = 2 * lam1 * (-np.dot(W_mode3*X_3, matrix_F) + np.dot(np.dot(H, matrix_F.T)*W_mode3, matrix_F))
    sig_H = np.zeros((len(H), len(H)), dtype=float)
    for i in range(len(H)):
        sig_H[i][i] = 1/(np.linalg.norm(H[i]))
    gd_H = gd_H + lam4 * np.dot(sig_H, H)
    gd_H = gd_H.reshape(1, view_num * r)
    return gd_H


def gradient_P(P, theta_v, X_t, B_v, X_1, B_s, H, lam1, z, rho, Lambda, sample_num, feature_D, view_num, r, W_mode1):
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    H = H.reshape(view_num, r)
    # for i in range(view_num):
    #     B_v[i] = B_v[i].reshape(r, feature_D)
    gd_P = np.zeros((sample_num, r), dtype=float)
    for i in range(view_num):
        gd_P = gd_P + 2 * theta_v[i] * (np.dot(-X_t[i], B_v[i].T) + np.dot(np.dot(P, B_v[i]), B_v[i].T))
    matrix_D = np.zeros((len(B_s) * len(H), len(B_s[0])), dtype=float)
    for i in range(len(B_s[0])):
        matrix_D[:, i] = np.kron(H[:, i], B_s[:, i])
    gd_P = gd_P + 2 * lam1 * (-np.dot(X_1*W_mode1, matrix_D) + np.dot(np.dot(P, matrix_D.T)*W_mode1, matrix_D))
    gd_P = gd_P + Lambda + rho * (P - z)
    gd_P = gd_P.reshape(1, sample_num * r)
    return gd_P


def gradient_B_s(B_s, theta_v, B_v, X_2, P, H, lam1, lam3, sample_num, feature_D, view_num, r, W_mode2):
    # for i in range(view_num):
    #     B_v[i] = B_v[i].reshape(r, feature_D)
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    H = H.reshape(view_num, r)
    matrix_E = np.zeros((len(H) * len(P), len(P[0])), dtype=float)
    for i in range(len(P[0])):
        matrix_E[:, i] = np.kron(H[:, i], P[:, i])
    gd_B = 2 * lam1 * (-np.dot(W_mode2*X_2, matrix_E) + np.dot(np.dot(B_s, matrix_E.T)*W_mode2, matrix_E))
    # for i in range(view_num):
    #     gd_B = gd_B + 2 * lam3 * theta_v[i] * (B_s - B_v[i].T)
    gd_B = gd_B.reshape(1, feature_D * r)
    return gd_B


def gradient_B_v(B_vi, theta_vi, X_ti, B_s, P, lam3, sample_num, feature_D, feature_Di, r):
    B_vi = B_vi.reshape(r, feature_Di)
    P = P.reshape(sample_num, r)
    B_s = B_s.reshape(feature_D, r)
    gd_Bv = 2 * theta_vi * (-np.dot(P.T, X_ti) + np.dot(np.dot(P.T, P), B_vi))
    # gd_Bv = gd_Bv + lam3 * 2 * theta_vi * (B_vi.T - B_s).T
    gd_Bv = gd_Bv.reshape(1, feature_Di * r)
    return gd_Bv
