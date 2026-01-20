from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, average_precision_score
from metacapsl.utils import SplitByGene, SplitByPairs
from metacapsl.utils import pair_move_batch_to_device, gene_move_batch_to_device
from tqdm import tqdm


class Evaluator():
    def __init__(self, args, fc_classifier, data):
        self.args = args
        self.maml_fintue = fc_classifier
        self.data = data
        # self.criterion = nn.BCELoss(reduction='none')
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # self.optimizer = optim.Adam(self.maml_fintue.parameters(), args.outer_lr)

    def eval_gene_split(self):
        all_auc_C1 = []
        all_auc_C2 = []
        all_auc_C3 = []
        all_aupr_C1 = []
        all_aupr_C2 = []
        all_aupr_C3 = []
        all_loss_C1 = 0
        all_loss_C2 = 0
        all_loss_C3 = 0
        all_embedding_C1 = []
        all_embedding_C2 = []
        all_embedding_C3 = []

        dataloader = DataLoader(self.data, batch_size=self.args.batch_size, shuffle=False, 
            num_workers=self.args.num_workers, collate_fn=self.args.collate_fn)

        for b_idx, batch in enumerate(dataloader):
            train_sl_features, train_ids, train_labels, C1_sl_featuers, C1_ids, C1_labels, C2_sl_featuers, C2_ids, C2_labels, C3_sl_featuers, C3_ids, C3_labels, cancer_type = self.args.move_batch(batch, self.args.device)

            # for each task in the batch
            effective_batch_size = len(batch[0])
            for i in range(effective_batch_size):
                learner = self.maml_fintue.clone()
                # fine tuneing
                for _ in range(self.args.adaption_steps): # adaptation_steps
                    _, support_preds = learner(train_sl_features[i])
                    support_loss = self.criterion(torch.squeeze(support_preds), train_labels[i])
                    print(f'Loss: {support_loss.item()}')
                    learner.adapt(support_loss)

                # prediction
                with torch.no_grad():
                    sl_labels = C1_labels[i]
                    c1_embeddings, logits = learner(C1_sl_featuers[i])
                    loss = self.criterion(torch.squeeze(logits), sl_labels)
                    all_loss_C1 += loss.cpu().detach().numpy().item()

                    auc_, aupr_ = self.calculate_metric_logits(logits, sl_labels)
                    all_auc_C1.append(auc_)
                    all_aupr_C1.append(aupr_)
                    all_embedding_C1.append(torch.cat([c1_embeddings, sl_labels.view(-1, 1)], dim=1).cpu().detach().numpy())
                
                with torch.no_grad():
                    sl_labels = C2_labels[i]
                    c2_embeddings, logits = learner(C2_sl_featuers[i])
                    loss = self.criterion(torch.squeeze(logits), sl_labels)
                    all_loss_C2 += loss.cpu().detach().numpy().item()

                    auc_, aupr_ = self.calculate_metric_logits(logits, sl_labels)
                    all_auc_C2.append(auc_)
                    all_aupr_C2.append(aupr_)
                    all_embedding_C2.append(torch.cat([c2_embeddings, sl_labels.view(-1, 1)], dim=1).cpu().detach().numpy())

                with torch.no_grad():
                    sl_labels = C3_labels[i]
                    c3_embeddings, logits = learner(C3_sl_featuers[i])
                    loss = self.criterion(torch.squeeze(logits), sl_labels)
                    all_loss_C3 += loss.cpu().detach().numpy().item()

                    auc_, aupr_ = self.calculate_metric_logits(logits, sl_labels)
                    all_auc_C3.append(auc_)
                    all_aupr_C3.append(aupr_)
                    all_embedding_C3.append(torch.cat([c3_embeddings, sl_labels.view(-1, 1)], dim=1).cpu().detach().numpy())
                    
                    # all_f1.append(f1)
        return {'loss': all_loss_C1, 'auc': all_auc_C1, 'aupr': all_aupr_C1, 'emb': all_embedding_C1}, \
            {'loss': all_loss_C2, 'auc': all_auc_C2, 'aupr': all_aupr_C2, 'emb': all_embedding_C2}, \
            {'loss': all_loss_C3, 'auc': all_auc_C3, 'aupr': all_aupr_C3, 'emb': all_embedding_C3}, 


    def calculate_metric_logits(self, logits, label):
        m = nn.Sigmoid()
        y_pred = torch.squeeze(m(logits)).cpu().tolist()
        target = label.cpu().tolist()

        auc_ = roc_auc_score(target, y_pred)
        p, r, t = precision_recall_curve(target, y_pred)
        aupr_ = auc(r, p)
        
        return auc_, aupr_
        
