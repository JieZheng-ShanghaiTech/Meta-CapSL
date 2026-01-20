import torch
import argparse
import pandas as pd

import learn2learn as l2l
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from metacapsl.utils import create_dirs
from metacapsl.utils import save_config
from metacapsl.utils import confirm_exp_name
from metacapsl.utils import gene_collate_fn, gene_move_batch_to_device
from metacapsl.model import CoSLAttention
from metacapsl.evaluator import Evaluator
from metacapsl.dataset import CancerSLGeneDataset, CancerSLPairDataset


def setup_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test(args):
    index = 1
    test_index = [4, 26, 10, 13, 6, 20]
    for seed in range(42, 47):
        print(f'===> Seed {index} is begin to training')
        fc_classifier =  CoSLAttention(omics_dim=args.omics_dim, kge_dim=args.kge_dim, feature_view=args.feature_view).to(device=args.device)

        maml = l2l.algorithms.MAML(fc_classifier, lr=args.inner_lr, first_order=False, allow_unused=True, allow_nograd=True)
        if args.pretrain_model_dir:
            print(f'===> Load pretrained model parameters...')
            model_path = os.path.join(args.pretrain_model_dir, 'Epoch_' + str(args.model_epoch) + '_meta_sl.pth')
            maml.load_state_dict(torch.load(model_path))
        else:
            print(f'===> Initilaize parameters randomly...')
            
        args.collate_fn = gene_collate_fn
        args.move_batch = gene_move_batch_to_device

        test_dataset = CancerSLGeneDataset(args.data_path, model='test', num_support=args.num_support, seed=seed)
        test_evaluator = Evaluator(args, maml, test_dataset)

        test_metric_C1, test_metric_C2, test_metric_C3 = test_evaluator.eval_gene_split()
        res = []
        setting = ['C1', 'C2', 'C3']
        flag = 0
        for test_metric in [test_metric_C1, test_metric_C2, test_metric_C3]:
            test_auc, test_aupr = test_metric['auc'], test_metric['aupr']
            for i in range(len(test_metric_C1['auc'])):
                # Save the metric in setting i cnacer j
                res.append([test_index[i], setting[flag], test_auc[i], test_aupr[i]])
                # Save the embedding in setting i cnacer j
            
            flag = flag + 1 
            
        res_df = pd.DataFrame(columns=['Sample', 'Setting', 'AUC', 'AUPR'], data=res)
        res_df.to_csv(os.path.join(args.result_dir, 'Seed_' + str(index) + '_result.csv'), index=None)
        index = index + 1


def configuration():
    parser = argparse.ArgumentParser(description="Meta-SL")
    # train_params
    parser.add_argument("--inner_lr", '-r', help="Learning late for training", default=0.02, type=float)
    parser.add_argument("--outer_lr",  help="Learning late for training", default=0.001, type=float)
    parser.add_argument("--l2", '-w', help="Weight decay for controling the impact of latent factor", default=5e-4, type=float)
    parser.add_argument("--adaption_steps",type=int, default=10, help="adaption steps", )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument("--omics_dim", help="the dimmension of multi-omics features", default=270, type=int)
    parser.add_argument("--kge_dim", help="the dimmension of knowledge graph features", default=128, type=int)
    parser.add_argument("--num_workers", type=int, default=32, help="Number of dataloading processes")
    parser.add_argument("--device", help="the device id of cuda", default='2', type=int)

    # data
    parser.add_argument('--data_path', type=str, default="./data/CancerSL/triples_exp_kge_filter", help="data path")

    # experiment
    parser.add_argument('--exp_root', type=str, default="./meta_testing", help="experiment root dir")

    # modify
    parser.add_argument("--exp_name", type=str, default=None, help="The name of experiment")
    parser.add_argument("--setting", type=str, default='C1', help="The name of experiment")
    parser.add_argument("--pretrain_model_dir", type=str, default=None, help="The name of experiment")
    parser.add_argument("--num_support", type=int, default=20, help="The name of experiment")
    parser.add_argument("--model_epoch", type=int, default=5, help="The name of experiment")
    parser.add_argument('--feature_view', type=str, default="all", help="data path")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    config = configuration()
    
    config.exp_name = str(config.num_support) + '_' + str(config.model_epoch) + '_' + str(config.exp_name)

    config.num_epochs = config.adaption_steps

    config.exp_name = confirm_exp_name(config.exp_name, config.exp_root)
    device=config.device
    batch_size = config.batch_size
    exp_root = config.exp_root
    exp_name = config.exp_name
    data_path = config.data_path
    # model_name = config.model_name
    exp_path = os.path.join(exp_root, exp_name)
    result_dir = os.path.join(exp_path, 'result')
    config_dir = os.path.join(exp_path, 'config')
    config_path = os.path.join(config_dir, 'config.yaml')
    config.result_dir = result_dir
    create_dirs([exp_path, config_dir, result_dir])
    save_config(config, config_path)
    setup_seed(42)
    print("===> Starting fine-tunning meta-sl model")
    test(config)