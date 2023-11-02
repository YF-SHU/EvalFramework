import math
import time
import gc
import numpy as np
import tensorflow as tf

from inputs import read_kgs, read_embeds
from utils import early_stop, get_statistics
from train_funcs import uniform_sampling, nearest_sampling, nearest_sampling_1, nearest_sampling_2
from test_funcs import greedy_alignment, stable_alignment, sim
from bootstrap import find_potential_alignment_mwgm, update_labeled_alignment_x, update_labeled_alignment_y


class BasicModel:
    def __init__(self):
        self.args = None

        self.links = None
        self.triples = None
        self.ents1 = None
        self.ents2 = None
        self.ent_num = None
        self.rel_num = None
        self.train_links = None
        self.test_links = None
        self.pretrained_embeds = None
        self.new_alignments = None
        self.new_alignment_indices = None

        self.all_layers = None
        self.model = None
        self.inputs = None

    def init(self):
        pass

    def set_args(self, args):
        self.args = args

    def set_kgs(self):
        self.train_links, self.test_links, self.triples, self.ents1, self.ents2, self.ent_num, self.rel_num = \
            read_kgs(self.args.input, self.args.train_ratio)
        if self.args.ent_name_init:
            self.pretrained_embeds = read_embeds(self.args.input, self.args.pretrained_model)
        self.new_alignments = []
        self.new_alignment_indices = set()

    def get_optimizer(self, name, learning_rate):
        optimizer = tf.optimizers.get(name)
        config = optimizer.get_config()
        config["learning_rate"] = learning_rate
        return optimizer.from_config(config)

    def concatenate(self, outputs, normalize=True):
        embeds_list = list()
        if normalize:
            for embeds in outputs:
                # embeds = tf.nn.l2_normalize(embeds, axis=-1)
                embeds_list.append(embeds)
        else:
            embeds_list = outputs
        embeds = tf.concat(embeds_list, axis=-1)
        return embeds

    def valid(self, embeds):
        embeds1 = tf.gather(embeds, self.test_links[:, 0]).numpy()
        embeds2 = tf.gather(embeds, self.test_links[:, 1]).numpy()
        _, hits1_12, _, _ = greedy_alignment(embeds1, embeds2, self.args.hits_k, self.args.eval_threads_num,
                                             self.args.eval_metric, False, 0, False)
        return hits1_12

    def test(self):
        embeds = self.model(self.inputs, training=False)
        if self.all_layers:
            embeds = self.concatenate(embeds)
        embeds1 = tf.gather(embeds, self.test_links[:, 0]).numpy()
        embeds2 = tf.gather(embeds, self.test_links[:, 1]).numpy()

        greedy_alignment(embeds1, embeds2, self.args.hits_k, self.args.eval_threads_num,
                         self.args.eval_metric, False, self.args.eval_csls, True)
        # alignment_result_set, _, _, _ = greedy_alignment(embeds1, embeds2, self.args.hits_k, self.args.eval_threads_num,
        #                                                  self.args.eval_metric, False, 1, True)
        # get_statistics(alignment_result_set)

        if self.args.inference_strategy == 'SM':
            stable_alignment(embeds1, embeds2, self.args.eval_metric, False, self.args.eval_csls,
                             self.args.eval_threads_num)

    def save(self):
        pass

    def generate_neg_samples(self, embeds, train_links):
        if self.args.neg_strategy == 'uniform':
            neg_right, neg_left = uniform_sampling(self.ent_num, train_links.shape[0], self.args.neg_multi)
        else:
            neg_right, neg_left = nearest_sampling(embeds, train_links, self.args.neg_multi)

        neg_samples = np.stack((neg_right, neg_left))
        return neg_samples

    def compute_loss(self, embeds, pos_batch, neg_batch):
        def dist(x1, x2):
            return tf.reduce_sum(tf.abs(x1 - x2), axis=-1)

        if self.all_layers:
            embeds = self.concatenate(embeds)
        pos_left = tf.gather(embeds, pos_batch[:, 0])
        pos_right = tf.gather(embeds, pos_batch[:, 1])
        neg1_left = pos_left
        neg1_right = tf.reshape(tf.gather(embeds, neg_batch[0].flatten()), [-1, len(pos_batch), embeds.shape[1]])
        neg2_left = tf.reshape(tf.gather(embeds, neg_batch[1].flatten()), [-1, len(pos_batch), embeds.shape[1]])
        neg2_right = pos_right
        pos_dist = dist(pos_left, pos_right)
        loss1 = tf.reduce_mean(tf.nn.relu(self.args.neg_margin + pos_dist - dist(neg1_left, neg1_right)))
        loss2 = tf.reduce_mean(tf.nn.relu(self.args.neg_margin + pos_dist - dist(neg2_left, neg2_right)))
        loss = (loss1 + loss2) / 2
        return loss

    def train(self):
        optimizer = self.get_optimizer(self.args.optimizer, self.args.learning_rate)
        flag1, flag2 = 0, 0
        neg_samples = None
        train_links = self.get_train_links()
        train_num = len(train_links)
        for i in range(self.args.max_epoch):
            start = time.time()
            epoch_loss = 0.0
            if i % self.args.neg_update_freq == 0:
                embeds = self.model(self.inputs, training=False)
                if self.all_layers:
                    embeds = embeds[-1]
                    # embeds = self.concatenate(embeds)
                neg_samples = self.generate_neg_samples(embeds, train_links)
            for j in range(self.args.batch_num):
                begin = int(train_num / self.args.batch_num * j)
                if j == self.args.batch_num - 1:
                    end = train_num
                else:
                    end = int(train_num / self.args.batch_num * (j + 1))
                with tf.GradientTape() as tape:
                    embeds = self.model(self.inputs, training=True)
                    batch_loss = self.compute_loss(embeds, train_links[begin:end], neg_samples[:, :, begin:end])
                grads = tape.gradient(batch_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_loss += batch_loss
            # embeds = self.model(self.inputs, training=False)
            # self.valid(embeds)
            print('epoch {}, loss: {:.4f}, cost time: {:.4f}s'.format(i, epoch_loss, time.time() - start))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                embeds = self.model(self.inputs, training=False)
                if self.all_layers:
                    embeds = self.concatenate(embeds)
                if self.args.early_stopping:
                    flag = self.valid(embeds)
                    flag1, flag2, is_stop = early_stop(flag1, flag2, flag)
                    if is_stop:
                        break

    def get_train_links(self):
        if len(self.new_alignments) == 0:
            return self.train_links
        return np.concatenate([self.train_links, np.array(self.new_alignments)], axis=0)

    def generate_new_alignments(self):
        rest_test_1 = [e1 for e1, e2 in self.test_links]
        rest_test_2 = [e2 for e1, e2 in self.test_links]
        train_links = self.get_train_links()
        for e1, e2 in train_links:
            if e1 in rest_test_1:
                rest_test_1.remove(e1)
            if e2 in rest_test_2:
                rest_test_2.remove(e2)

        embeds = self.model(self.inputs, training=False)
        embeds1 = tf.gather(embeds, rest_test_1)
        embeds2 = tf.gather(embeds, rest_test_2)
        embeds1 = embeds1.numpy()
        embeds2 = embeds2.numpy()
        sim_mat = sim(embeds1, embeds2, self.args.eval_metric, False, self.args.eval_csls)
        sim_mat = tf.convert_to_tensor(sim_mat)

        _, indices_x2y = tf.nn.top_k(sim_mat)
        _, indices_y2x = tf.nn.top_k(tf.transpose(sim_mat))
        for i in range(sim_mat.shape[0]):
            j = indices_x2y[i, 0]
            if indices_y2x[j, 0] == i:
                self.new_alignments.append([rest_test_1[i], rest_test_2[j]])

    def bootstrapping(self):
        embeds = self.model(self.inputs, training=False)
        embeds1 = tf.gather(embeds, self.test_links[:, 0])
        embeds2 = tf.gather(embeds, self.test_links[:, 1])
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        embeds2 = tf.nn.l2_normalize(embeds2, 1)
        sim_mat = tf.matmul(embeds1, tf.transpose(embeds2)).numpy()
        curr_labeled_alignment = find_potential_alignment_mwgm(sim_mat, sim_th=0.7, k=10, heuristic=False)
        if curr_labeled_alignment is not None:
            labeled_alignment = self.new_alignment_indices
            labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat)
            self.new_alignment_indices = update_labeled_alignment_y(labeled_alignment, sim_mat)
            del curr_labeled_alignment
        if len(self.new_alignment_indices) != 0:
            self.new_alignments = []
            for (i, j) in self.new_alignment_indices:
                self.new_alignments.append([self.test_links[i, 0], self.test_links[j, 1]])
        del sim_mat
        gc.collect()

    def generate_new_triples(self):
        links = self.get_train_links()
        links1 = dict(zip(links[:, 0], links[:, 1]))
        links2 = dict(zip(links[:, 1], links[:, 0]))
        new_triples = set()
        num = 0
        for h, r, t in self.triples:
            if h in links1 and t in links1:
                triple = (links1[h], r, links1[t])
                if triple not in new_triples:
                    new_triples.add(triple)
                    num += 1
            if h in links2 and t in links2:
                triple = (links2[h], r, links2[t])
                if triple not in new_triples:
                    new_triples.add(triple)
                    num += 1
        print("new triples added: ", num)
        return new_triples

    def eval_new_alignments(self):
        total_new_num = len(self.new_alignments)
        total_true_num = len(self.test_links)
        align_dict = dict(zip(self.test_links[:, 0], self.test_links[:, 1]))
        new_true_num = 0
        for i, j in self.new_alignments:
            if i in align_dict and j == align_dict[i]:
                new_true_num += 1
        precision = new_true_num / total_new_num
        recall = new_true_num / total_true_num
        print("precision: {:.4f}, recall: {:.4f}".format(precision, recall))



