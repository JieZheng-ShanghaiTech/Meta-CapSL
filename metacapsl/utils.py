import torch
import numpy as np
import pandas as pd
import random
import os
import yaml
import scipy.sparse as sp
import time
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold


def feature_move_batch_to_device_cpu(batch):
    train, C1, C2, C3, cancer_type = batch

    train_sl_features_ = [] 
    train_ids_ = []
    train_labels_ = []
    for i in train:
        train_sl_features_.append(i[:, :-3].numpy())
        train_ids_.append(i[:, -3:-1].numpy())
        train_labels_.append(i[:, -1].numpy())

    C1_sl_features_ = [] 
    C1_ids_ = []
    C1_labels_ = []
    for i in C1:
        C1_sl_features_.append(i[:, :-3].numpy())
        C1_ids_.append(i[:, -3:-1].numpy())
        C1_labels_.append(i[:, -1].numpy())

    C2_sl_features_ = [] 
    C2_ids_ = []
    C2_labels_ = []
    for i in C2:
        C2_sl_features_.append(i[:, :-3].numpy())
        C2_ids_.append(i[:, -3:-1].numpy())
        C2_labels_.append(i[:, -1].numpy())

    C3_sl_features_ = [] 
    C3_ids_ = []
    C3_labels_ = []
    for i in C3:
        C3_sl_features_.append(i[:, :-3].numpy())
        C3_ids_.append(i[:, -3:-1].numpy())
        C3_labels_.append(i[:, -1].numpy())
        
    return train_sl_features_, train_ids_, train_labels_, C1_sl_features_, C1_ids_, C1_labels_, C2_sl_features_, C2_ids_, C2_labels_, C3_sl_features_, C3_ids_, C3_labels_, cancer_type


def pair_move_batch_to_device(batch, device):
    sl_data, cancer_type = batch
    sl_features_ = []
    ids_ = []
    labels_ = []


    for i in sl_data:
        sl_features_.append(i[:, :-3].to(device=device))
        ids_.append(i[:, -3:-1].to(device=device))
        labels_.append(i[:, -1].to(device=device))
        
    return sl_features_, ids_, labels_, cancer_type


def cancer_pair_move_batch_to_device(batch, device):
    sl_data, cancer_type = batch
    sl_features_ = []
    ids_ = []
    labels_ = []

    for i in sl_data:
        sl_features_.append(i[:, :-3].to(device))
        ids_.append(i[:, -3:-1])
        labels_.append(i[:, -1])
        
    return sl_features_, ids_, labels_, cancer_type


def gene_move_batch_to_device(batch, device):
    train, C1, C2, C3, cancer_type = batch

    train_sl_features_ = [] 
    train_ids_ = []
    train_labels_ = []
    for i in train:
        train_sl_features_.append(i[:, :-3].to(device=device))
        train_ids_.append(i[:, -3:-1].to(device=device))
        train_labels_.append(i[:, -1].to(device=device))

    C1_sl_features_ = [] 
    C1_ids_ = []
    C1_labels_ = []
    for i in C1:
        C1_sl_features_.append(i[:, :-3].to(device=device))
        C1_ids_.append(i[:, -3:-1].to(device=device))
        C1_labels_.append(i[:, -1].to(device=device))

    C2_sl_features_ = [] 
    C2_ids_ = []
    C2_labels_ = []
    for i in C2:
        C2_sl_features_.append(i[:, :-3].to(device=device))
        C2_ids_.append(i[:, -3:-1].to(device=device))
        C2_labels_.append(i[:, -1].to(device=device))

    C3_sl_features_ = [] 
    C3_ids_ = []
    C3_labels_ = []
    for i in C3:
        C3_sl_features_.append(i[:, :-3].to(device=device))
        C3_ids_.append(i[:, -3:-1].to(device=device))
        C3_labels_.append(i[:, -1].to(device=device))
        
    return train_sl_features_, train_ids_, train_labels_, C1_sl_features_, C1_ids_, C1_labels_, C2_sl_features_, C2_ids_, C2_labels_, C3_sl_features_, C3_ids_, C3_labels_, cancer_type


def collate_fn(samples):
    om_graphs, sl_labels = map(list, zip(*samples))
    return om_graphs, sl_labels

def pair_collate_fn(samples):
    sl_data, cancer_type = map(list, zip(*samples))
    return sl_data, cancer_type

def gene_collate_fn(samples):
    train, c1, c2, c3, cancer_type = map(list, zip(*samples))
    return train, c1, c2, c3, cancer_type



def SplitByPairs(sl_pos, kfold):
    # If no negtive samples, generate them randomly.
    df_sl = pd.DataFrame(generateNSByRandom(sl_pos))
    sl_np_x = df_sl[[0, 1]].to_numpy()
    sl_np_y = df_sl[2].to_numpy()

    # index is the fold
    index = 1

    random_seed = 43
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_seed)

    for test_index, train_index in kf.split(sl_np_x, sl_np_y):
        train_x = sl_np_x[train_index]
        train_y = sl_np_y[train_index]
        train_data = np.concatenate((train_x, train_y), axis=1)

        test_x = sl_np_x[test_index]
        test_y = sl_np_y[test_index].reshape(-1, 1)
        test_data = np.concatenate((test_x, test_y), axis=1)

        break
    return train_data, test_data



def reindex(inter_pairs):
    """reindex function
    Args:
        inter_pairs (triple): 2D
    """
    sl_df = pd.DataFrame(data=inter_pairs)
    set_IDa = set(sl_df[0])
    set_IDb = set(sl_df[1])
    list_all = list(set_IDa | set_IDb)

    orig2id = {}
    id2orig = {}
    for i in range(len(list_all)):
        origin = list_all[i]
        orig2id[origin] = i
        id2orig[i] = origin
    for key in orig2id:
        sl_df.loc[sl_df[0]==key, 0] = orig2id[key]
        sl_df.loc[sl_df[1]==key, 1] = orig2id[key]
    return sl_df.values, orig2id, id2orig, list_all


def generateNSByRandom(inter_pairs, seed):
    all_inters = inter_pairs

    inters_reindex, orig2id, id2orig, gene_list = reindex(inter_pairs)
    len_ = len(gene_list)
    edges = inters_reindex.shape[0]
    adj = sp.coo_matrix((np.ones(edges), (inters_reindex[:, 0], inters_reindex[:, 1])), shape=(len_, len_))

    adj_neg = 1 - adj.todense() - np.eye(len_)
    neg_u, neg_v = np.where(adj_neg != 0)
    np.random.seed(seed)
    neg_eids = np.random.choice(len(neg_u), edges)
    for neg_idx in range(len(neg_eids)):
        all_inters += [[id2orig[neg_u[neg_eids[neg_idx]]], id2orig[neg_v[neg_eids[neg_idx]]], 0]]

    return np.asarray(all_inters)


def get_pairs(sl_pairs, train_genes, test_genes, type):
    pairs_with_genes = []
    for pair in sl_pairs:
        if type==1:
            if pair[0] in train_genes and pair[1] in train_genes:
                pairs_with_genes.append(list(pair))
        elif type==2:
            if (pair[0] in test_genes and pair[1] in train_genes) or (pair[0] in train_genes and pair[1] in test_genes):
                pairs_with_genes.append(list(pair))
        elif type==3:
            if pair[0] in test_genes and pair[1] in test_genes:
                pairs_with_genes.append(list(pair))
    return pairs_with_genes


def SplitByGene(sl_pos, kfold, seed=43):
    positive = np.concatenate((np.array(sl_pos), np.ones(len(sl_pos)).reshape(-1,1)), axis=1)
    gene1 = set(positive[:, 0])
    gene2 = set(positive[:, 1])
    genes = np.array(list(gene1 | gene2))

    index = 1
    random_seed = seed
    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_seed)

    # for test_index, train_index in kf.split(genes):
    for train_index, test_index in kf.split(genes):
        train_genes = genes[train_index]
        test_genes = genes[test_index]
        
        train_positive_pairs = get_pairs(positive, train_genes, test_genes=None, type=1)
        train_data = generateNSByRandom(train_positive_pairs, seed=seed)
        
        test_c2_positive_pairs = get_pairs(positive, train_genes, test_genes, type=2)
        test_c2_data = generateNSByRandom(test_c2_positive_pairs, seed=seed)

        test_c3_positive_pairs = get_pairs(positive, train_genes, test_genes, type=3)
        test_c3_data = generateNSByRandom(test_c3_positive_pairs, seed=seed)

        break
    print(f'Trian: {len(train_data)}, C2: {len(test_c2_data)}, C3: {len(test_c3_data)}')
    return train_data, test_c2_data, test_c3_data



def confirm_exp_name(exp_name, exp_root):
    # auto assign exp_name if it is not manually assigned
    make_dir(exp_root)

    if exp_name is None:
        exist_exp = os.listdir(exp_root)
        exp_nums = [int(name[3:]) for name in exist_exp if name[3:].isdigit()]

        if len(exp_nums) > 0:
            new_num = max(exp_nums) + 1
        else:
            new_num = 1

        exp_name = "exp" + str(new_num)

    return exp_name


def create_dirs(dirs):
    for i in dirs:
        make_dir(i)


def save_config(config, config_path):
    data = {
            'exp_name': config.exp_name,
            'exp_root': config.exp_root,
            # 'data_root': config.data_root,
            'num_epochs': config.num_epochs,
            'inner_lr': config.inner_lr,
            'outer_lr': config.outer_lr,
            'adaption_steps': config.adaption_steps,
            'weight_decay': config.l2,
            'batch_size': config.batch_size,
            'omics_dim': config.omics_dim,
            'kge_dim': config.kge_dim,
            'device': config.device}

    with open(config_path, 'a') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)