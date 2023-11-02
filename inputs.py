import pickle
import string
import numpy as np


class KG:
    def __init__(self, ents, rels, triples):
        self.ents = ents
        self.rels = rels
        self.triples = triples

    def get_ent_degree_dict(self):
        ent_deg_dict = dict()
        for h, r, t in self.triples:
            if h not in ent_deg_dict:
                ent_deg_dict[h] = 1
            else:
                ent_deg_dict[h] += 1
            if t != h:
                if t not in ent_deg_dict:
                    ent_deg_dict[t] = 1
                else:
                    ent_deg_dict[t] += 1
        return ent_deg_dict


def read_triples(file_name):
    triples = []
    ents = set()
    rels = set()
    for line in open(file_name, 'r'):
        h, r, t = [int(item) for item in line.split()]
        ents.add(h)
        ents.add(t)
        rels.add(r)
        triples.append((h, r, t))
    return KG(ents, rels, triples)


def read_links(file_name):
    links = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        links.append((int(e1), int(e2)))
    return links


def read_id_features(file_name):
    id_names = {}
    for line in open(file_name, 'r', encoding='utf-8'):
        info = line.strip().split('\t')
        if len(info) == 2:
            name = ''.join([' ' if i in string.punctuation else i for i in info[1]])
        else:
            name = '<unk>'
        id_names[int(info[0])] = name
    return id_names


def read_embeds(folder, pretrained_model):
    if pretrained_model == "bert":
        file = open(folder + "Bert_max_all.pkl", 'rb')
        embedding_list = pickle.load(file)
    else:
        file = open(folder + "fasttext_mean_all.pkl", 'rb')
        embedding_list = pickle.load(file)
    return embedding_list


def read_kgs(path, train_ratio=0.3):
    kg1 = read_triples(path + 'triples_1')
    kg2 = read_triples(path + 'triples_2')
    triples = kg1.triples + kg2.triples
    num_ents = len(kg1.ents.union(kg2.ents))
    num_rels = len(kg1.rels.union(kg2.rels))
    print('ents, rels: %d, %d' % (num_ents, num_rels))
    if "_en" in path:
        links = read_links(path + 'ref_ent_ids')
        np.random.shuffle(links)
        train_links, test_links = links[0:int(len(links) * train_ratio)], links[int(len(links) * train_ratio):]
    else:
        train_links = read_links(path + 'sup_ent_ids')
        test_links = read_links(path + 'ref_ent_ids')
    return np.array(train_links), np.array(test_links), triples, list(kg1.ents), list(kg2.ents), num_ents, num_rels
