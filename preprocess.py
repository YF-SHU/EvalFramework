import pickle
import random

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from inputs import read_id_features

seed = 1234
random.seed(seed)
np.random.seed(seed)


# BERT
def minus_mask(inputs, input_lens, mask_type='max'):
    # Inputs shape = (batch_size, sent_len, embed_dim)
    # input_lens shape = [batch_size]
    # max_len scalar
    assert inputs.shape[0] == input_lens.shape[0]
    assert len(input_lens.shape) == 1
    assert len(inputs.shape) == 3

    max_len = tf.reduce_max(input_lens)
    mask = tf.reshape(tf.range(max_len), [1, -1])
    mask = tf.greater_equal(mask, tf.reshape(input_lens, [-1, 1]))
    mask = tf.where(mask, 1.0, 0.0)
    mask = tf.reshape(mask, [-1, max_len, 1])
    if mask_type == 'max':
        mask = mask * 1e-30
        inputs = inputs + mask
    elif mask_type == "mean":
        inputs = inputs - mask * inputs
    return inputs


class BERT(object):
    # For entity alignment, the best layer is 1
    def __init__(self, model='bert-base-cased', pool='max'):
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=False)
        self.model = TFBertModel.from_pretrained(model, output_hidden_states=True)
        self.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.cls_token_id = self.tokenizer.encode(self.tokenizer.cls_token)[0]
        self.sep_token_id = self.tokenizer.encode(self.tokenizer.sep_token)[0]
        self.pool = pool

    def pooled_encode_batched(self, sentences, batch_size=512, layer=1):
        # Split the sentences into batches and further encode
        sent_batch = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        outputs = []
        for batch in tqdm(sent_batch):
            out = self.pooled_bert_encode(batch, layer)
            outputs.append(out)
        outputs = tf.concat(outputs, axis=0)
        return outputs.numpy()

    def pooled_bert_encode(self, sentences, layer=1):
        required_layer_hidden_state, sent_lens = self.bert_encode(sentences, layer)
        required_layer_hidden_state = minus_mask(required_layer_hidden_state, sent_lens, self.pool)
        # Max or mean pooling
        if self.pool == 'max':
            required_layer_hidden_state = tf.reduce_max(required_layer_hidden_state, axis=1)
        elif self.pool == 'mean':
            required_layer_hidden_state = tf.reduce_mean(required_layer_hidden_state, axis=1)
        return required_layer_hidden_state

    def bert_encode(self, sentences, layer=1):
        # layer: output the max pooling over the designated layer hidden state

        # Limit batch size to avoid exceed gpu memory limitation
        sent_num = len(sentences)
        assert sent_num <= 512
        # The 382 is to avoid exceed bert's maximum seq_len and to save memory
        sentences = [[self.cls_token_id] + self.tokenizer.encode(sent)[:382] + [self.sep_token_id] for sent in
                     sentences]
        sent_lens = [len(sent) for sent in sentences]
        max_len = max(sent_lens)
        sent_lens = tf.convert_to_tensor(sent_lens)
        sentences = tf.convert_to_tensor([sent + (max_len - len(sent)) * [self.pad_token_id] for sent in sentences])
        result = self.model(sentences, training=False)
        last_hidden_state, all_hidden_state = result["last_hidden_state"], result["hidden_states"]
        if layer is None:
            required_layer_hidden_state = last_hidden_state
        else:
            required_layer_hidden_state = all_hidden_state[layer]
        return required_layer_hidden_state, sent_lens


# fasttext
def get_fasttext_vectors(words, all_vectors_file):
    word2embs = {}
    with open(all_vectors_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            word, embed = line.strip().split(' ', 1)
            word2embs[word] = np.fromstring(embed, sep=' ')
    emb_unknown = np.zeros(300)
    embeds = [word2embs.get(word, emb_unknown) for word in words]
    return np.vstack(embeds)


def reduce(embeds, method, axis=0):
    assert method in {'mean', 'max', 'min'}
    if method == 'mean':
        return np.mean(embeds, axis)
    if method == 'max':
        return np.max(embeds, axis)
    if method == 'min':
        return np.min(embeds, axis)


def fasttext_embedding(names, words_aggr_type, all_vectors_file):
    word_id_map = {}
    ent_words = []
    for name in names:
        words = name.lower().split()
        for word in words:
            if word not in word_id_map:
                word_id_map[word] = len(word_id_map)
        word_ids = [word_id_map[word] for word in words]
        ent_words.append(word_ids)
    all_words = sorted(list(word_id_map.keys()), key=lambda x: word_id_map[x])
    all_embs = get_fasttext_vectors(all_words, all_vectors_file)
    embeds = []
    if words_aggr_type == 'cpm':
        for word_ids in ent_words:
            concat_embeds = []
            for method in ['mean', 'min', 'max']:
                concat_embeds.append(reduce(all_embs[word_ids], method))
            embeds.append(np.concatenate(concat_embeds, axis=0))
    else:
        for word_ids in ent_words:
            embeds.append(reduce(all_embs[word_ids], 'mean'))
    return np.vstack(embeds)


def write_fasttext_vec_file(feature_file, output_file, all_vectors_file):
    id_names = read_id_features(feature_file)
    ids = list(id_names.keys())
    names = list(id_names.values())
    embeds = fasttext_embedding(names, "mean", all_vectors_file)
    id_embeds = dict(zip(ids, embeds))
    with open(output_file + ".pkl", 'wb') as f:
        pickle.dump(id_embeds, f)


if __name__ == '__main__':
    datasets = ["data/DBP15K/zh_en/", "data/DBP15K/ja_en/", "data/DBP15K/fr_en/"]
    all_vectors_file = "data/wiki.en.vec"
    for path in datasets:
        print("write vectors for " + path)
        feature_file_1 = path + "id_features_1"
        output_file_1 = path + "fasttext_mean_1"
        write_fasttext_vec_file(feature_file_1, output_file_1, all_vectors_file)
        feature_file_2 = path + "id_features_2"
        output_file_2 = path + "fasttext_mean_2"
        write_fasttext_vec_file(feature_file_2, output_file_2, all_vectors_file)

    path = "data/SRPRS/en_fr_15k_V1/"
    print("write vectors for " + path)
    feature_file_1 = path + "id_features_1"
    output_file_1 = path + "fasttext_mean_1"
    write_fasttext_vec_file(feature_file_1, output_file_1, "data/wiki.en.align.vec")
    feature_file_2 = path + "id_features_2"
    output_file_2 = path + "fasttext_mean_2"
    write_fasttext_vec_file(feature_file_2, output_file_2, "data/wiki.fr.align.vec")

    path = "data/SRPRS/en_de_15k_V1/"
    print("write vectors for " + path)
    feature_file_1 = path + "id_features_1"
    output_file_1 = path + "fasttext_mean_1"
    write_fasttext_vec_file(feature_file_1, output_file_1, "data/wiki.en.align.vec")
    feature_file_2 = path + "id_features_2"
    output_file_2 = path + "fasttext_mean_2"
    write_fasttext_vec_file(feature_file_2, output_file_2, "data/wiki.de.align.vec")

    datasets = ["data/DBP15K/zh_en/", "data/DBP15K/ja_en/", "data/DBP15K/fr_en/",
                "data/SRPRS/en_fr_15k_V1/", "data/SRPRS/en_de_15k_V1/"]
    for path in datasets:
        print("merge vectors for " + path)
        file = open(path + "fasttext_mean_1.pkl", 'rb')
        id_embeds1 = pickle.load(file)
        print(len(id_embeds1))
        file = open(path + "fasttext_mean_2.pkl", 'rb')
        id_embeds2 = pickle.load(file)
        print(len(id_embeds2))
        ids = list(id_embeds1.keys())
        ids.extend(list(id_embeds2.keys()))
        embedding_list = []
        for i in sorted(ids):
            value = id_embeds1.get(i)
            if value is None:
                value = id_embeds2.get(i)
            embedding_list.append(list(value))
        with open(path + "fasttext_mean_all.pkl", 'wb') as f:
            pickle.dump(embedding_list, f)
        print('fasttext: {} rows, {} columns'.format(len(embedding_list), len(embedding_list[0])))


