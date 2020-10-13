import numpy as np


def predict_func(x, w, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    result = np.dot(x, w)
    # result = np.exp(result)
    # for i in range(len(result)):
    #     result[i] = result[i]/np.sum(result[i])
    return result


def lossfunction_LE(w, x_original, y_logical, M, z, k, f_dim, l_dim, lam1, lam2):
    w = w.reshape(f_dim, l_dim)
    d_predict = predict_func(x_original, w, f_dim, l_dim)
    loss1 = np.linalg.norm(d_predict - y_logical)**2
    # loss2 = lam1 * (np.linalg.norm(d_predict - np.dot(M, d_predict))**2)
    loss3 = 0.
    for i in range(k):
        loss3 += np.linalg.norm(z[i], ord='nuc')
    loss3 = lam2 * loss3
    # print("D-D^:", loss1)
    # print("lam1 D-MD:", loss2)
    # print("lam2 D_tr:", loss3)
    return loss1 + loss3


def lossfunction_w(w, x_original, x_cluster, y_logical, M, z, Lambda, rho, k, f_dim, l_dim, lam1):
    w = w.reshape(f_dim, l_dim)
    d_predict = predict_func(x_original, w, f_dim, l_dim)
    loss1 = np.linalg.norm(d_predict - y_logical) ** 2
    # loss2 = lam1 * (np.linalg.norm(d_predict - np.dot(M, d_predict)) ** 2)
    loss3 = 0.
    loss4 = 0.
    for i in range(k):
        loss3 += np.sum(Lambda[i] * (predict_func(x_cluster[i], w, f_dim, l_dim) - z[i]))
        loss4 += (rho[i] / 2) * np.sum((predict_func(x_cluster[i], w, f_dim, l_dim) - z[i]) ** 2)
    return loss1 + loss3 + loss4


def update_z(w, x_cluster, z, Lambda, rho, k, f_dim, l_dim, lam2):
    w = w.reshape(f_dim, l_dim)
    z_new = []
    for i in range(k):
        u, sigma, vt = np.linalg.svd(predict_func(x_cluster[i], w, f_dim, l_dim) + Lambda[i] / rho[i])
        sigma_new = [s if s - (lam2 / rho[i]) > 0 else 0 for s in sigma]
        temp = np.diag(sigma_new)
        height, width = z[i].shape
        if len(sigma) < width:
            temp = np.c_[temp, np.zeros([len(sigma), width - len(sigma)])]
        if len(sigma) < height:
            temp = np.r_[temp, np.zeros([height - len(sigma), width])]
        z_new.append(np.dot(np.dot(u, temp), vt))
    return z_new


def update_Lambda(w, x_cluster, z, Lambda, rho, k, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    Lambda_new = []
    for i in range(k):
        Lambda_new.append(Lambda[i] + rho[i]*(predict_func(x_cluster[i], w, f_dim, l_dim)-z[i]))
    return Lambda_new


def gradient_w(w, x_original, x_cluster, fake_LD, M, z, Lambda, rho, k, f_dim, l_dim, lam1):
    w = w.reshape(f_dim, l_dim)
    gd_w1 = 2 * (-np.dot(x_original.T, fake_LD) + np.dot(np.dot(x_original.T, x_original), w))
    # term = np.dot(M, x_original) - x_original
    # gd_w2 = lam1 * np.dot(np.dot(term.T, term), w)
    gd_w3 = np.zeros((f_dim, l_dim), dtype=float)
    gd_w4 = np.zeros((f_dim, l_dim), dtype=float)
    for i in range(k):
        gd_w3 = gd_w3 + np.dot(list(map(list, zip(*(x_cluster[i])))), Lambda[i])
        gd_w4 = gd_w4 + rho[i] * (-np.dot(list(map(list, zip(*(x_cluster[i])))), z[i]) + np.dot(np.dot(list(map(list, zip(*(x_cluster[i])))), x_cluster[i]), w))
    gd_w1 = gd_w1.reshape(1, -1)
    # gd_w2 = gd_w2.reshape(1, -1)
    gd_w3 = gd_w3.reshape(1, -1)
    gd_w4 = gd_w4.reshape(1, -1)
    return gd_w1 + gd_w3 + gd_w4
