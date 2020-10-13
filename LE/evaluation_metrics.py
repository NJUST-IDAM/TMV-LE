import numpy as np


def hamming_loss(true_lab, pre_lab):
    num_instance = true_lab.shape[0]
    num_class = true_lab.shape[1]
    ham = true_lab != pre_lab
    return ham.sum() / (num_class * num_instance)


def one_error(true_lab, pre_value):
    true_lab, pre_value = data_clean(true_lab, pre_value)
    num_instance, num_class = true_lab.shape
    error_amount = 0
    for i in range(num_instance):
        temp = pre_value[i, :]
        max_index = np.argmax(temp)
        if true_lab[i, max_index] != 1:
            error_amount += 1
    return error_amount / num_instance


def coverage(test_target, pre_value):
    # test_target, pre_value = data_clean(test_target, pre_value)
    num_instance, num_class = test_target.shape
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(num_instance):
        labels_size.append(sum(test_target[i] == 1))
        index1, index2 = find(test_target[i], 1, 0)
        labels_index.append(index1)
        not_labels_index.append(index2)

    cover = 0
    max_index = num_class - 1
    for i in range(num_instance):
        tempvalue, index = sort(pre_value[i])
        temp_min = max_index + 1
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            if loc < temp_min:
                temp_min = loc
        cover = cover + (max_index - temp_min + 1)
    return (cover / num_instance - 1) / num_class


def average_precision(test_target, pre_value):
    test_target, pre_value = data_clean(test_target, pre_value)
    num_instance, num_class = test_target.shape
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(num_instance):
        labels_size.append(sum(test_target[i] == 1))
        index1, index2 = find(test_target[i], 1, 0)
        labels_index.append(index1)
        not_labels_index.append(index2)
    aveprec = 0
    for i in range(num_instance):
        tempvalue, index = sort(pre_value[i])
        indicator = np.zeros((num_class,))
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            # print(loc)
            summary = summary + sum(indicator[loc:num_class]) / (num_class - loc);
        aveprec = aveprec + summary / labels_size[i]
    return  aveprec / num_instance


def SubsetAccuracy(test_target, pre_labels):
    test_data_num, class_num = pre_labels.shape
    correct_num = 0
    for i in range(test_data_num):
        for j in range(class_num):
            if pre_labels[i][j] != test_target[i][j]:
                break
        if j == class_num - 1:
            correct_num = correct_num + 1

    return correct_num / test_data_num


def MacroAveragingAUC(test_target, pre_value):
    num_instance, num_class = pre_value.shape
    P = []
    N = []
    labels_size = []
    not_labels_size = []
    AUC = 0
    for i in range(num_class):
        P.append([])
        N.append([])

    for i in range(num_instance):  # 得到Pk和Nk
        for j in range(num_class):
            if test_target[i][j] == 1:
                P[j].append(i)
            else:
                N[j].append(i)

    for i in range(num_class):
        labels_size.append(len(P[i]))
        not_labels_size.append(len(N[i]))

    for i in range(num_class):
        auc = 0
        for j in range(labels_size[i]):
            for k in range(not_labels_size[i]):
                pos = pre_value[P[i][j]][i]
                neg = pre_value[N[i][k]][i]
                if pos > neg:
                    auc = auc + 1
        AUC = AUC + auc / (labels_size[i] * not_labels_size[i])
    return AUC / num_class


def Mic_F1(test_target, pre_labels):
    num_instance, num_class = test_target.shape
    num_pos_instance = np.sum(test_target, axis=0)
    num_neg_instance = num_instance - num_pos_instance
    pre_labels[pre_labels == 0] = 2  # 暂时将预测结果中的0改为2方便下面统计真正例的数量
    true_pos = np.sum(test_target == pre_labels, axis=0)  # TP
    false_neg = num_pos_instance - true_pos  # FN
    pre_labels[pre_labels == 2] = 0  # 更改为原来的真是结果
    pre_labels[pre_labels == 1] = 2  # 原因同上
    true_neg = np.sum(test_target == pre_labels, axis=0)  # TN
    false_pos = num_neg_instance - true_neg  # FP
    pre_labels[pre_labels == 2] = 1
    sum_tp = np.sum(true_pos)
    sum_fn = np.sum(false_neg)
    sum_tn = np.sum(true_neg)
    sum_fp = np.sum(false_pos)
    mircro_f1 = 2 * sum_tp / (2 * sum_tp + sum_fn + sum_fp)
    return mircro_f1


# 去除全1和全0样本
def data_clean(test_target, pre_value):
    num_instance, num_class = test_target.shape
    temp_true_lab = []
    temp_pre_value = []
    for i in range(num_instance):
        temp = test_target[i, :]
        if temp.sum() != num_class and temp.sum() != 0:
            temp_true_lab.append(temp)
            temp_pre_value.append(pre_value[i, :])
    test_target = np.array(temp_true_lab)
    pre_value = np.array(temp_pre_value)
    return test_target, pre_value


def find(instance, label1, label2):
    index1 = []
    index2 = []
    for i in range(instance.shape[0]):
        if instance[i] == label1:
            index1.append(i)
        if instance[i] == label2:
            index2.append(i)
    return index1, index2


def sort(x):
    temp = np.array(x)
    length = temp.shape[0]
    index = []
    sortX = []
    for i in range(length):
        Min = float("inf")
        Min_j = i
        for j in range(length):
            if temp[j] < Min:
                Min = temp[j]
                Min_j = j
        sortX.append(Min)
        index.append(Min_j)
        temp[Min_j] = float("inf")
    return temp,index


def findIndex(a, b):
    for i in range(len(b)):
        if a == b[i]:
            return i