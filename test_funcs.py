# Adapted from OpenEA
import gc
import multiprocessing
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances

from utils import task_divide, merge_dic


def greedy_alignment(embeds1, embeds2, top_k, nums_threads, metric, normalize, csls_k, accurate):
    """
    Search align with greedy strategy.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    top_k : list of integers
        Hits@k metrics for evaluating results.
    nums_threads : int
        The number of threads used to search align.
    metric : string
        The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
    normalize : bool, true or false.
        Whether to normalize the input embeddings.
    csls_k : int
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    alignment_rest :  list, pairs of aligned entities
    hits1 : float, hits@1 values for align results
    mr : float, MR values for align results
    mrr : float, MRR values for align results
    """
    t = time.time()
    sim_mat = sim(embeds1, embeds2, metric=metric, normalize=normalize, csls_k=csls_k)
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        search_tasks = task_divide(np.array(range(num)), nums_threads)
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, accurate, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    assert len(alignment_rest) == num
    hits = np.array(hits) / num * 100
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    cost = time.time() - t
    if accurate:
        if csls_k > 0:
            print("accurate results with csls, hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
        else:
            print("accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
    else:
        if csls_k > 0:
            print("quick results with csls, hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
        else:
            print("quick results: hits@{} = {}%, time = {:.3f} s \n".format(top_k, hits, cost))
    hits1 = hits[0]
    del sim_mat
    gc.collect()
    return alignment_rest, hits1, mr, mrr


def stable_alignment(embeds1, embeds2, metric, normalize, csls_k, nums_threads, cut=100, sim_mat=None):
    t = time.time()
    if sim_mat is None:
        sim_mat = sim(embeds1, embeds2, metric=metric, normalize=normalize, csls_k=csls_k)

    kg1_candidates, kg2_candidates = dict(), dict()

    num = sim_mat.shape[0]
    x_tasks = task_divide(np.array(range(num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(x_tasks))
    rests = list()
    total = 0
    for task in x_tasks:
        total += len(task)
        mat = sim_mat[task, :]
        rests.append(pool.apply_async(arg_sort, (task, mat, 'x_', 'y_')))
    assert total == num
    pool.close()
    pool.join()
    for rest in rests:
        kg1_candidates = merge_dic(kg1_candidates, rest.get())

    sim_mat = sim_mat.T
    num = sim_mat.shape[0]
    y_tasks = task_divide(np.array(range(num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(y_tasks))
    rests = list()
    for task in y_tasks:
        mat = sim_mat[task, :]
        rests.append(pool.apply_async(arg_sort, (task, mat, 'y_', 'x_')))
    pool.close()
    pool.join()
    for rest in rests:
        kg2_candidates = merge_dic(kg2_candidates, rest.get())

    # print("kg1_candidates", len(kg1_candidates))
    # print("kg2_candidates", len(kg2_candidates))

    # print("generating candidate lists costs time {:.3f} s ".format(time.time() - t))
    t = time.time()
    matching = galeshapley(kg1_candidates, kg2_candidates, cut)
    n = 0
    for i, j in matching.items():
        if int(i.split('_')[-1]) == int(j.split('_')[-1]):
            n += 1
    cost = time.time() - t
    print("stable alignment precision = {:.3f}%, time = {:.3f} s ".format(n / len(matching) * 100, cost))


def arg_sort(idx, sim_mat, prefix1, prefix2):
    candidates = dict()
    for i in range(len(idx)):
        x_i = prefix1 + str(idx[i])
        rank = (-sim_mat[i, :]).argsort()
        y_j = [prefix2 + str(r) for r in rank]
        candidates[x_i] = y_j
    return candidates


def galeshapley(suitor_pref_dict, reviewer_pref_dict, max_iteration):
    """ The Gale-Shapley algorithm. This is known to provide a unique, stable
    suitor-optimal matching. The algorithm is as follows:

    (1) Assign all suitors and reviewers to be unmatched.

    (2) Take any unmatched suitor, s, and their most preferred reviewer, r.
            - If r is unmatched, match s to r.
            - Else, if r is matched, consider their current partner, r_partner.
                - If r prefers s to r_partner, unmatch r_partner from r and
                  match s to r.
                - Else, leave s unmatched and remove r from their preference
                  list.
    (3) Go to (2) until all suitors are matched, then end.

    Parameters
    ----------
    suitor_pref_dict : dict
        A dictionary with suitors as keys and their respective preference lists
        as values
    reviewer_pref_dict : dict
        A dictionary with reviewers as keys and their respective preference
        lists as values
    max_iteration : int
        An integer as the maximum iterations

    Returns
    -------
    matching : dict
        The suitor-optimal (stable) matching with suitors as keys and the
        reviewer they are matched with as values
    """
    suitors = list(suitor_pref_dict.keys())
    matching = dict()
    rev_matching = dict()

    for i in range(max_iteration):
        if len(suitors) <= 0:
            break
        for s in suitors:
            r = suitor_pref_dict[s][0]
            if r not in matching.values():
                matching[s] = r
                rev_matching[r] = s
            else:
                r_partner = rev_matching.get(r)
                if reviewer_pref_dict[r].index(s) < reviewer_pref_dict[r].index(r_partner):
                    del matching[r_partner]
                    matching[s] = r
                    rev_matching[r] = s
                else:
                    suitor_pref_dict[s].remove(r)
        suitors = list(set(suitor_pref_dict.keys()) - set(matching.keys()))
    return matching


def sim(embeds1, embeds2, metric='cosine', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    # if normalize:
    #     embeds1 = preprocessing.normalize(embeds1)
    #     embeds2 = preprocessing.normalize(embeds2)
    # if metric == 'inner':
    #     sim_mat = np.matmul(embeds1, embeds2.T)  # numpy.ndarray, float32
    # elif metric == 'cosine' and normalize:
    #     sim_mat = np.matmul(embeds1, embeds2.T)  # numpy.ndarray, float32
    if metric =='cosine':
        # embeds1 = embeds1 / np.linalg.norm(embeds1, axis=-1, keepdims=True)
        # embeds2 = embeds2 / np.linalg.norm(embeds2, axis=-1, keepdims=True)
        embeds1 = embeds1 / np.maximum(np.linalg.norm(embeds1, axis=-1, keepdims=True), 1e-7)
        embeds2 = embeds2 / np.maximum(np.linalg.norm(embeds2, axis=-1, keepdims=True), 1e-7)
        sim_mat = np.matmul(embeds1, embeds2.T)
    elif metric == 'euclidean':
        sim_mat = 1 - euclidean_distances(embeds1, embeds2)
        sim_mat = sim_mat.astype(np.float32)
    # elif metric == 'cosine':
    #     sim_mat = 1 - cdist(embeds1, embeds2, metric='cosine')  # numpy.ndarray, float64
    #     sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embeds1, embeds2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embeds1, embeds2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        hits1_rest.add((gold, rank[0]))
        assert gold in rank
        rank_index = np.where(rank == gold)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)

