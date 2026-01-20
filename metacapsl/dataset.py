from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import json
from dgl.data.utils import save_graphs, load_graphs
import scipy.sparse as sp
import torch
from metacapsl.utils import SplitByGene, SplitByPairs
import random

class CancerSLPairDataset(Dataset):
    def __init__(self, triple_path):
        self.cancer_types = ["ACC","BLCA","BRCA","CESC","CHOL", "COAD","DLBC",
        "ESCA","GBM","HNSC","KICH","KIRC","KIRP","LAML","LGG","LIHC","LUAD","LUSC","OV","PAAD",
        "PCPG","PRAD","READ","SARC","SKCM", "STAD","TGCT","THCA","THYM","UCEC","UCS"]
        self.load_feats()

        test_cancer_type = ['CHOL', 'TGCT', 'KICH', 'LAML', 'DLBC', 'PCPG']
        index = [i for i in range(0, len(self.cancer_types)) if self.cancer_types[i] not in test_cancer_type]

        self.cancer_sl_features_labels = self.get_cancer_sl_features_labels(triple_path, index)
        
    
    def load_feats(self):
        self.kge_features = np.load('./data/KGE/node_embedding.npy', allow_pickle=True).item()
        self.all_cancer_features = json.load(open('./data/CancerSL/gene_expression/cancer_gene_features_270_5_24.json'))
    
    def generateNSByRandom(self, inter_pairs):
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
    
        neg_pairs = []
        inters_reindex, orig2id, id2orig, gene_list = reindex(inter_pairs)
        len_ = len(gene_list)
        edges = inters_reindex.shape[0]
        adj = sp.coo_matrix((np.ones(edges), (inters_reindex[:, 0], inters_reindex[:, 1])), shape=(len_, len_))
        adj_neg = 1 - adj.todense() - np.eye(len_)
        neg_u, neg_v = np.where(adj_neg != 0)
        np.random.seed(43)
        neg_eids = np.random.choice(len(neg_u), edges)
        print(f'Gene numbers: {len_}, Negative SL paris numbers: {len(neg_eids)}')
        for neg_idx in range(len(neg_eids)):
            neg_pairs += [[id2orig[neg_u[neg_eids[neg_idx]]], id2orig[neg_v[neg_eids[neg_idx]]]]]
        return np.asarray(neg_pairs)

    def get_cancer_sl_features_labels(self, save_path, index):
        cancer_types = self.cancer_types
        cancer_sl_feature_labels = []
        for cancer_type in cancer_types:
            file_path = os.path.join(save_path + '/' + cancer_type + '.txt')
            sl_pairs = np.loadtxt(file_path, dtype=int)

            neg_file_path = os.path.join(save_path + '/neg/' + cancer_type + '_neg.txt')
            if os.path.exists(neg_file_path):
                neg_sl_pairs = np.loadtxt(neg_file_path, dtype=int)
            else:
                neg_sl_pairs = self.generateNSByRandom(sl_pairs)
                np.savetxt(neg_file_path, neg_sl_pairs, fmt='%d')

            zero_ = np.zeros(len(sl_pairs))
            one_ = np.ones(len(sl_pairs))
            sl_pairs = np.concatenate((sl_pairs, one_.reshape(-1, 1)), axis=1)
            neg_sl_pairs = np.concatenate((neg_sl_pairs, zero_.reshape(-1, 1)), axis=1)

            all_sl_pairs = np.concatenate((sl_pairs, neg_sl_pairs), axis=0)
            np.random.seed(43)
            np.random.shuffle(all_sl_pairs)

            # get features
            gene_features = self.all_cancer_features[cancer_type]
            sl_features = []
            
            for i in all_sl_pairs:
                omics_a = gene_features[str(int(i[0]))]
                omics_b = gene_features[str(int(i[1]))]
                kge_a = self.kge_features[int(i[0])]
                kge_b = self.kge_features[int(i[1])]
                sl_features.append(np.concatenate((omics_a, omics_b, kge_a, kge_b), axis=0))
            sl_data = np.concatenate((sl_features, all_sl_pairs), axis=1)

            cancer_sl_feature_labels.append([torch.FloatTensor(sl_data), cancer_type])

        res = [cancer_sl_feature_labels[i] for i in index]
        return res
    
    def get_len(self):
        return len(self.cancer_sl_features_labels)

    def __getitem__(self, index):
        return self.cancer_sl_features_labels[index][0], self.cancer_sl_features_labels[index][1]

    def __len__(self):
        return len(self.cancer_sl_features_labels)



class CancerSLGeneDataset(Dataset):
    def __init__(self, triple_path, model, num_support, seed):
        self.all_cancer_types = ["ACC","BLCA","BRCA","CESC","CHOL", "COAD","DLBC",
        "ESCA","GBM","HNSC","KICH","KIRC","KIRP","LAML","LGG","LIHC","LUAD","LUSC","OV","PAAD",
        "PCPG","PRAD","READ","SARC","SKCM", "STAD","TGCT","THCA","THYM","UCEC","UCS"]
        self.load_feats()
        self.seed = seed

        test_cancer_type = ['CHOL', 'TGCT', 'KICH', 'LAML', 'DLBC', 'PCPG']

        # test_cancer_type = [4, 26, 10, 13, 6, 20]

        num_query = 100

        if model=='test':
            self.cancer_type = test_cancer_type
            self.cancer_sl_features_labels = self.get_data(self.cancer_type, triple_path, model, num_support, num_query)
        
    
    def load_feats(self):
        self.kge_features = np.load('./data/KGE/node_embedding.npy', allow_pickle=True).item()
        self.all_cancer_features = json.load(open('./data/CancerSL/gene_expression/cancer_gene_features_270_5_24.json'))


    def get_features(self, gene_features, kge_features, sl_pairs):
        new_features = []
        for i in sl_pairs:
            feature_a = gene_features[str(int(i[0]))]
            feature_b = gene_features[str(int(i[1]))]
            kge_a = kge_features[int(i[0])]
            kge_b = kge_features[int(i[1])]
            new_features.append(np.concatenate((feature_a, feature_b, kge_a, kge_b), axis=0))
        return np.concatenate((new_features, sl_pairs), axis=1)

    def get_data(self, cancer_types, sl_path, model, num_support, num_query):
        cancer_sl_feature_labels = {}
        all_data_tensor = {}

        if model=='test':
            data_path = './data/test_data_' + str(self.seed) + '_' + str(num_support) + '.pt'
            # data_path = './data/test_data_neighbor_20.pt'

        if os.path.exists(data_path):
            all_data_tensor = torch.load(data_path)
            for cancer_type in cancer_types:
                support_data, query_c1_data, query_c2_data, query_c3_data = all_data_tensor[cancer_type]
                cancer_sl_feature_labels[cancer_type] = [support_data, query_c1_data, query_c2_data, query_c3_data, cancer_type]
            
        else:
            for cancer_type in cancer_types:
                file_path = os.path.join(sl_path + '/' + cancer_type + '.txt')
                sl_pairs = np.loadtxt(file_path, dtype=int)
                sl_pos = sl_pairs
                support_set, query_set_c2, query_set_c3 = SplitByGene(sl_pos, kfold=2, seed=self.seed)
                np.random.seed(self.seed)
                np.random.shuffle(support_set)
                np.random.seed(self.seed)
                np.random.shuffle(query_set_c2)
                np.random.seed(self.seed)
                np.random.shuffle(query_set_c3)

                # get features
                gene_features = self.all_cancer_features[cancer_type]
        
                support_data = self.get_features(gene_features, self.kge_features, support_set)

                random.seed(self.seed)
                support_index = random.sample(range(0, len(support_data)), int(num_support/100 * len(support_data)))
                other_index = [i for i in range(0, len(support_data)) if i not in support_index]
                # query_c1_index = random.sample(other_index, num_query)
                query_c1_index = other_index

                support_sl_data = support_data[support_index]
                query_c1_data = support_data[query_c1_index]

                query_c2_data = self.get_features(gene_features, self.kge_features, query_set_c2)

                query_c3_data = self.get_features(gene_features, self.kge_features, query_set_c3)
                
                all_data_tensor[cancer_type] = [torch.FloatTensor(support_sl_data), torch.FloatTensor(query_c1_data), torch.FloatTensor(query_c2_data), torch.FloatTensor(query_c3_data)]
                cancer_sl_feature_labels[cancer_type] = [torch.FloatTensor(support_sl_data), torch.FloatTensor(query_c1_data), torch.FloatTensor(query_c2_data), torch.FloatTensor(query_c3_data), cancer_type]
            
            torch.save(all_data_tensor, data_path)
            
        res = [cancer_sl_feature_labels[i] for i in cancer_types]
        return res
    
    def get_len(self):
        return len(self.cancer_sl_features_labels)

    def __getitem__(self, index):
        return self.cancer_sl_features_labels[index][0], self.cancer_sl_features_labels[index][1], self.cancer_sl_features_labels[index][2], self.cancer_sl_features_labels[index][3], self.cancer_sl_features_labels[index][4]

    def __len__(self):
        return len(self.cancer_sl_features_labels)





class CancerSLDataset(Dataset):
    def __init__(self, triple_path, cancer_type):
        self.cancer_types = ["ACC","BLCA","BRCA","CESC","CHOL", "COAD","DLBC",
        "ESCA","GBM","HNSC","KICH","KIRC","KIRP","LAML","LGG","LIHC","LUAD","LUSC","OV","PAAD",
        "PCPG","PRAD","READ","SARC","SKCM", "STAD","TGCT","THCA","THYM","UCEC","UCS"]
        self.load_feats()

        self.cancer_type = cancer_type

        self.cancer_sl_features_labels = self.get_cancer_sl_features_labels(triple_path)
        
    
    def load_feats(self):
        self.kge_features = np.load('./data/KGE/node_embedding.npy', allow_pickle=True).item()
        self.all_cancer_features = json.load(open('./data/CancerSL/gene_expression/cancer_gene_features_270_5_24.json'))
    

    def get_cancer_sl_features_labels(self, file_path):
        cancer_sl_feature_labels = []
        
        sl_pairs = np.loadtxt(file_path, dtype=int)

        np.random.seed(43)
        np.random.shuffle(sl_pairs)

        # get features
        gene_features = self.all_cancer_features[self.cancer_type]
        sl_features = []
        
        for i in sl_pairs:
            try:
                omics_a = gene_features[str(int(i[0]))]
                omics_b = gene_features[str(int(i[1]))]
                kge_a = self.kge_features[int(i[0])]
                kge_b = self.kge_features[int(i[1])]
                sl_features.append(np.concatenate((omics_a, omics_b, kge_a, kge_b), axis=0))
            except:
                continue
        sl_data = np.concatenate((sl_features, sl_pairs), axis=1)

        return torch.FloatTensor(sl_data)
    
    def get_len(self):
        return len(self.cancer_sl_features_labels)

    def __getitem__(self, index):
        return self.cancer_sl_features_labels, self.cancer_type

    def __len__(self):
        return len(self.cancer_sl_features_labels)

