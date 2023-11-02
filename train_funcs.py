import numpy as np
import tensorflow as tf

from test_funcs import sim


def uniform_sampling(ent_num, t, k):
    neg_right = np.random.choice(ent_num, (k, t))
    neg_left = np.random.choice(ent_num, (k, t))
    return neg_right, neg_left


# nearest neighbors are chosen from entities in train_links
def nearest_sampling(embeds, train_links, k, metric='cosine'):
    left = train_links[:, 0]
    right = train_links[:, 1]
    embs_left = tf.gather(embeds, left).numpy()
    embs_right = tf.gather(embeds, right).numpy()

    t = left.shape[0]
    neg_right = []
    sim_mat = sim(embs_right, embs_right, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k+1)
        neg_right.append(right[indices[1: k+1]])
    neg_right = np.array(neg_right)
    neg_right = neg_right.T  # shape: (k, t)

    neg_left = []
    sim_mat = sim(embs_left, embs_left, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k + 1)
        neg_left.append(left[indices[1: k + 1]])
    neg_left = np.array(neg_left)
    neg_left = neg_left.T  # shape: (k, t)

    return neg_right, neg_left


# nearest neighbors are chosen from all entities
def nearest_sampling_1(embeds, train_links, k, metric='cosine'):
    left = train_links[:, 0]
    right = train_links[:, 1]
    embs_left = tf.gather(embeds, left).numpy()
    embs_right = tf.gather(embeds, right).numpy()
    embeds = embeds.numpy()

    t = left.shape[0]
    neg_right = []
    sim_mat = sim(embs_left, embeds, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k+1)
        neg_right.append(indices[1: k+1])
    neg_right = np.array(neg_right)
    neg_right = neg_right.T  # shape: (k, t)

    neg_left = []
    sim_mat = sim(embs_right, embeds, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k + 1)
        neg_left.append(indices[1: k + 1])
    neg_left = np.array(neg_left)
    neg_left = neg_left.T  # shape: (k, t)

    return neg_right, neg_left


# nearest neighbors are chosen from KG1 and KG2 respectively
def nearest_sampling_2(embeds, train_links, ents1, ents2, k, metric='cosine'):
    left = train_links[:, 0]
    right = train_links[:, 1]
    embs_left = tf.gather(embeds, left).numpy()
    embs_right = tf.gather(embeds, right).numpy()
    embs_ents1 = tf.gather(embeds, ents1).numpy()
    embs_ents2 = tf.gather(embeds, ents1).numpy()

    t = left.shape[0]
    neg_right = []
    sim_mat = sim(embs_left, embs_ents1, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k+1)
        neg_right.append(ents1[indices[1: k+1]])
    sim_mat = sim(embs_left, embs_ents2, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k + 1)
        neg_right.append(ents2[indices[1: k + 1]])
    neg_right = np.array(neg_right)
    neg_right = neg_right.T  # shape: (2*k, t)

    neg_left = []
    sim_mat = sim(embs_right, embs_ents2, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k + 1)
        neg_left.append(ents2[indices[1: k + 1]])
    sim_mat = sim(embs_right, embs_ents1, metric)
    for i in range(t):
        indices = np.argpartition(-sim_mat[i, :], k + 1)
        neg_left.append(ents1[indices[1: k + 1]])
    neg_left = np.array(neg_left)
    neg_left = neg_left.T  # shape: (2*k, t)

    return neg_right, neg_left

