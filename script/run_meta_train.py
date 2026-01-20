import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import os
import argparse
import sys

import learn2learn as l2l


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from metacapsl.dataset import CancerSLPairDataset
from metacapsl.model import CoSLAttention

from metacapsl.log import Logger
from metacapsl.utils import create_dirs
from metacapsl.utils import save_config
from metacapsl.utils import confirm_exp_name





def setup_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



from metacapsl.utils import pair_move_batch_to_device, pair_collate_fn


class Trainer():
    def __init__ (self, args, meta_model, dataset, index, logger):
        self.args = args
        self.index = index
        self.epoch = 0
        self.dataset = dataset
        self.logger = logger
        self.criterion = nn.BCELoss(reduction='none')
        self.meta_model = meta_model
        self.optimizer = optim.Adam(self.meta_model.parameters(), args.outer_lr)
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, 
            num_workers=self.args.num_workers, collate_fn=pair_collate_fn)


    def train_epoch(self):
        for b_idx, batch in enumerate(self.train_dataloader):
            sl_features, sl_ids, sl_labels, cancer_types = pair_move_batch_to_device(batch, self.args.device)

            effective_batch_size = len(batch[0])
            batch_outer_loss = torch.tensor(0., device=self.args.device)
            loss_number = []

            # each task in one batch
            for i in range(effective_batch_size):

                learner = self.meta_model.clone()
                # divide the data into support and query sets
                split_index = int(len(sl_ids[i]) * 1 / (1 + 9))
                support_ids, query_ids = np.split(sl_ids[i], [split_index])
                support_features, query_features = np.split(sl_features[i], [split_index])
                support_labels, query_labels = np.split(sl_labels[i], [split_index])
                
                for _ in range(self.args.adaption_steps): # adaptation steps
                    _, support_preds = learner(support_features)
                    m = nn.Sigmoid()
                    support_loss = self.criterion(torch.squeeze(m(support_preds)), support_labels)
                    inner_loss = torch.sum(support_loss) / support_loss.shape[0]
                    learner.adapt(inner_loss)
                
                _, query_preds = learner(query_features)
                m = nn.Sigmoid()
                query_loss = self.criterion(torch.squeeze(m(query_preds)), query_labels)
                outer_loss = torch.sum(query_loss) / query_loss.shape[0]
                batch_outer_loss += outer_loss
                loss_number.append(outer_loss.item())

            # attention ware optimization
            s = nn.Softmax(dim=0)
            weights = s(torch.tensor(loss_number))

            batch_outer_loss = batch_outer_loss / effective_batch_size

            if b_idx % 1 == 0:
                self.logger.record_train_log(f"Meta Train Loss = {batch_outer_loss.item()}")

            self.optimizer.zero_grad()
            batch_outer_loss.backward()
            self.optimizer.step()


    def train(self):
        for epoch in tqdm(range(1, self.args.num_epochs + 1)):
            self.epoch = epoch
            self.train_epoch()
            if epoch % self.args.save_iter == 0:
                self.save_model()


    def save_model(self):
        torch.save(self.meta_model.state_dict(), os.path.join(self.args.model_dir, 'Epoch_' + str(self.epoch) + '_meta_sl.pth'))

            

def train(args, logger):
    cv_index = 1
    print(f'===> CV_{cv_index} is begin to training')
    train_dataset = CancerSLPairDataset(args.data_path)
    meta_sl = CoSLAttention(omics_dim=args.omics_dim, kge_dim=args.kge_dim, feature_view=args.feature_view).to(device=args.device)
    meta_model = l2l.algorithms.MAML(meta_sl, lr=args.inner_lr, first_order=False, allow_unused=True)
    trainer = Trainer(args, meta_model, train_dataset, cv_index, logger)
    trainer.train()


def configuration():
    parser = argparse.ArgumentParser(description="Meta-CapSL")
    # train_params
    parser.add_argument("--num_epochs", '-ne', help="The number of epochs for training", default=100, type=int)
    parser.add_argument("--inner_lr", '-r', help="Learning late for training", default=0.01, type=float)
    parser.add_argument("--outer_lr",  help="Learning late for training", default=0.001, type=float)
    parser.add_argument("--l2", '-w', help="Weight decay for controling the impact of latent factor", default=5e-4, type=float)
    parser.add_argument("--adaption_steps",type=int, default=5, help="adaption steps")

    parser.add_argument("--save_iter",type=int, default=5, help="save iteration")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument("--omics_dim", help="the dimmension of multi-omics features", default=270, type=int)
    parser.add_argument("--kge_dim", help="the dimmension of knowledge graph features", default=128, type=int)
    parser.add_argument("--num_workers", type=int, default=32, help="Number of dataloading processes")
    parser.add_argument("--device", help="the device id of cuda", default='0', type=int)

    # experiment
    parser.add_argument("--exp_name", type=str, default=None, help="The name of experiment", )
    parser.add_argument('--exp_root', type=str, default="./meta_traning", help="experiment root dir")
    parser.add_argument('--data_path', type=str, default="./data/CancerSL/triples_exp_kge_filter", help="data path")
    parser.add_argument('--feature_view', type=str, default="all", help="data path")

    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    config = configuration()
    config.feature_view = 'all'
    # config.inner_lr = 0.005
    config.adaption_steps = 2


    config.exp_name = confirm_exp_name(config.exp_name, config.exp_root)
    device=config.device
    batch_size = config.batch_size
    exp_root = config.exp_root
    exp_name = config.exp_name
    data_path = config.data_path
    # model_name = config.model_name
    exp_path = os.path.join(exp_root, exp_name)
    log_dir = os.path.join(exp_path, 'log')
    model_dir = os.path.join(exp_path, 'model')
    config_dir = os.path.join(exp_path, 'config')
    config_path = os.path.join(config_dir, 'config.yaml')

    config.model_dir = model_dir
    create_dirs([exp_path, log_dir, model_dir, config_dir])
    save_config(config, config_path)

    logger = Logger(log_dir, exp_name)
    setup_seed(42)
    print("===> Starting pre-traning meta-sl model")
    train(config, logger)
     
