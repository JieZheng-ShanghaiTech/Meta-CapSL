
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from sklearn.metrics import f1_score


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, seq_len, self.hidden_size)
        k = k.view(batch_size, seq_len, self.hidden_size)
        v = v.view(batch_size, seq_len, self.hidden_size)

        attention_weights = torch.bmm(q, k.transpose(1, 2))
        attention_weights = self.softmax(attention_weights)

        scores = torch.bmm(q, k.transpose(1, 2))
        scaled_scores = scores / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        attention_weights = self.softmax(scaled_scores)
        output = torch.bmm(attention_weights, v)

        return output


class CoSLAttention(nn.Module):
    def __init__(self, omics_dim, kge_dim, feature_view):
        super().__init__()

        self.feature_view = feature_view
        self.omics_dim = omics_dim
        self.kge_dim = kge_dim
        if self.feature_view == 'all':
            self.omics_encoder = nn.Sequential(
                    nn.Linear(270, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 64),
            )
            self.kge_encoder = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 64),
            )
            input_dim = 64

            self.attn =SelfAttention(input_size=input_dim, hidden_size=input_dim)
            self.AN1 = torch.nn.LayerNorm(input_dim)
            self.l1 = torch.nn.Linear(input_dim, input_dim)
            self.AN2 = torch.nn.LayerNorm(input_dim)
            
            self.decoder = nn.Linear(256, 1)


        elif self.feature_view == 'omics':
            self.omics_encoder = nn.Sequential(
                    nn.Linear(270, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 64),
            )
            self.omics_decoder = nn.Linear(128, 1)

        elif self.feature_view == 'kg':
            self.kge_encoder = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 64),
            )
            self.kg_decoder = nn.Linear(128, 1)


    def forward(self, sl_features):
        omics_feature_a = sl_features[:, : self.omics_dim]
        omics_feature_b = sl_features[:, self.omics_dim : 2 * self.omics_dim]

        kge_feature_a = sl_features[:, 2 * self.omics_dim : 2 * self.omics_dim + self.kge_dim]
        kge_feature_b = sl_features[:, 2 * self.omics_dim + self.kge_dim :]

        if self.feature_view == 'all':
            o_a = self.omics_encoder(omics_feature_a)
            o_b = self.omics_encoder(omics_feature_b)
            k_a = self.kge_encoder(kge_feature_a)
            k_b = self.kge_encoder(kge_feature_b)

            X = torch.cat([o_a, k_a], dim=1).view(o_a.shape[0], 2, -1)
            output = self.attn(X)
            X = self.AN1(output + X)
            output = self.l1(X)
            fuse_feat_a = self.AN2(output + X)

            X = torch.cat([o_b, k_b], dim=1).view(o_b.shape[0], 2, -1)
            output = self.attn(X)
            X = self.AN1(output + X)
            output = self.l1(X)
            fuse_feat_b = self.AN2(output + X)

            fuse_feat = torch.cat([fuse_feat_a.view(fuse_feat_a.shape[0], -1), fuse_feat_b.view(fuse_feat_b.shape[0], -1)], dim=1)
            
            output = self.decoder(fuse_feat.view(fuse_feat.shape[0], -1))

        elif self.feature_view == 'omics':
            o_a = self.omics_encoder(omics_feature_a)
            o_b = self.omics_encoder(omics_feature_b)
            fuse_feat = torch.cat([o_a, o_b], dim=1)
            output = self.omics_decoder(fuse_feat)

        elif self.feature_view == 'kg':
            k_a = self.kge_encoder(kge_feature_a)
            k_b = self.kge_encoder(kge_feature_b)
            fuse_feat = torch.cat([k_a, k_b], dim=1)
            output = self.kg_decoder(fuse_feat)

        return fuse_feat, output