import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.optimize import fmin_l_bfgs_b
from LE.lossfunction_LE import *
from LE.evaluation_metrics import *
from sklearn.model_selection import train_test_split
import h5py



def read_mat(url):
    data = sio.loadmat(url)
    return data


if __name__ == "__main__":
    lam1 = 10 ** -6
    lam2 = 10 ** -6
    k = 5
    data1 = read_mat(r"../Datasets/all_mat/Emotions/2/Emotions_P.mat")
    x_original = data1['P']
    ones = 0.5 * np.ones((len(x_original), 1), dtype=float)

    data3 = read_mat(r"../Datasets/all_mat/Emotions/Emotions_10_fold_origin_2_views.mat")
    y_logical_t = data3['target']
    # data3 = h5py.File(r'../Datasets/all_mat/Pascal/Pascal_10_fold_origin_5_views.mat', 'r')
    # y_logical_t = ((data3['target'].value).T)
    # data3 = read_mat(r"../Datasets/all_mat/Emotions/2/Emotions_labels.mat")
    # y_logical_t = data3['labels']


    data4 = read_mat(r"../Datasets/all_mat/Emotions/2/Emotions_M.mat")
    M = data4['M']
    # threshold = 0.5 * np.ones((len(y_logical_t), 1), dtype=float)
    # y_logical = np.c_[y_logical_t, threshold]
    f_dim = len(x_original[0]) 
    l_dim = len(y_logical_t[0])

    result1 = []
    result2 = []
    result3 = []
    result4 = []

    for t in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x_original, y_logical_t, test_size=0.1, random_state=2)
        # initialize
        w = np.random.rand(f_dim, l_dim) # Three ways to initialize
        # w = np.zeros((f_dim, l_dim), dtype=float)
        # w = np.ones((f_dim, l_dim), dtype=float)
        kmeans = KMeans(n_clusters=k).fit(x_train)
        kmeans_result = kmeans.predict(x_train)
        x_cluster = []
        y_cluster = []
        for i in range(k):
            x_cluster.append([])
            y_cluster.append([])
        for i in range(len(x_train)):  # 后期len(features)改为训练集大小
            x_cluster[kmeans_result[i]].append(list(x_train[i]))
            y_cluster[kmeans_result[i]].append(list(y_train[i]))
        z = []  # update
        for i in range(k):
            # z.append(np.ones_like(d_result[i]))
            z.append(np.ones_like(y_cluster[i]))
            # z.append(np.ones_like(d_result[i]) / labels_dim)
        Lambda = []  # update
        for i in range(k):
            Lambda.append(np.zeros_like(y_cluster[i]))
        rho = np.ones(k) * (10 ** -6)  # parameter
        rho_max = 10 ** 6
        beta = 1.1  # increase factor
        # update step
        loss = lossfunction_LE(w, x_train, y_train, M, z, k, f_dim, l_dim, lam1, lam2)
        print(loss)
        for i in range(100):
            print(t, i, "-" * 20)
            #Because of the fmin_l_bfgs_b function, the objective function may not converge for some data sets and initialization parameters, at which point another optimizer can be replaced.
            result = fmin_l_bfgs_b(lossfunction_w, w, gradient_w, args=(x_train, x_cluster, y_train, M, z, Lambda, rho, k, f_dim, l_dim, lam1),
                                   pgtol=0.00001, maxiter=20)
            w = result[0]
            if i <= 10:
                z = update_z(w, x_cluster, z, Lambda, rho, k, f_dim, l_dim, lam2)
                Lambda = update_Lambda(w, x_cluster, z, Lambda, rho, k, f_dim, l_dim)
            loss_new = lossfunction_LE(w, x_train, y_train, M, z, k, f_dim, l_dim, lam1, lam2)
            if abs(loss - loss_new) < 10 ** -4 or loss_new > loss:
                break
            rho = np.min([rho[0]*beta, rho_max]) * np.ones(k)
            loss = loss_new
            print(loss)


        # predict the label distributions of test set
        pre_test = predict_func(x_test, w, f_dim, l_dim)
        pre_test = np.exp(pre_test)
        for i in range(len(pre_test)):
            pre_test[i] = pre_test[i]/np.sum(pre_test[i])
        for i in range(len(pre_test)):
            s = 1.05
            # threshold_i = s * ((np.max(pre_test[i]) + np.min(pre_test[i]))/2)
            threshold_i = s * np.average(pre_test[i]) # Three kinds of threshold, adjustable parameter s
            # threshold_i = pre_test[i][-1]
            for j in range(len(pre_test[0])):
                if pre_test[i][j] >= threshold_i:
                    pre_test[i][j] = 1
                else:
                    pre_test[i][j] = 0

        result1.append(hamming_loss(y_test, pre_test))
        result2.append(coverage(y_test, pre_test))
        result3.append(average_precision(y_test, pre_test))
        result4.append(Mic_F1(y_test, pre_test))


    print("hamming_loss:", np.mean(result1), "+", np.std(result1))
    print("coverage:", np.mean(result2), "+", np.std(result2))
    print("average_precision:", np.mean(result3), "+", np.std(result3))
    print("Mic_F1:", np.mean(result4), "+", np.std(result4))

