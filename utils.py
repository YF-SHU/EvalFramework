import numpy as np


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def early_stop(flag1, flag2, flag):
    if flag < flag2 < flag1:
        return flag2, flag, True
    else:
        return flag2, flag, False


def get_statistics(pair_set):
    id_num_dic = dict()
    num = len(pair_set)
    for i in range(num):
        id_num_dic[i] = 0
    for _, id in pair_set:
        id_num_dic[id] += 1
    id_num = np.array(list(id_num_dic.values()))
    num_0 = round(len(np.where(id_num == 0)[0]) / num, 2)
    num_1 = round(len(np.where(id_num == 1)[0]) / num, 2)
    num_x = 1.0 - num_0 - num_1
    num_prop = [num_0, num_1, num_x]
    print(num_prop)





