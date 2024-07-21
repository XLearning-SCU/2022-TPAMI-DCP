import argparse
import itertools
import time
import torch

from model_multiview import DCPMultiView
from utils.get_mask import get_mask
from utils.util import cal_std
from utils.logger_ import get_logger
from utils.datasets import *
from configure.configure_clustering_multiview import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")

dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "LandUse_21"
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0.5', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)  # 使用第一, 三块GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger, plt_name = get_logger(config)

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X_list, Y_list = load_multiview_data(config)

    fold_acc, fold_nmi, fold_ari = [], [], []
    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        # Get Mask
        mask = get_mask(config['view'], X_list[0].shape[0], config['missing_rate'])
        X_list_new = []
        for i in range(config['view']):
            X_list_new.append(X_list[i] * mask[:, i][:, np.newaxis])

        X_list_new = [torch.from_numpy(X_list_new[i]).float().to(device) for i in range(config['view'])]
        mask = torch.from_numpy(mask).long().to(device)

        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        # Set random seeds
        if config['missing_rate'] == 0:
            seed = data_seed * config['seed']
        else:
            seed = config['seed']

        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build model
        DCP_model = DCPMultiView(config)
        optimizer = torch.optim.Adam(DCP_model.parameters(), lr=config['training']['lr'])

        logger.info(DCP_model.autoencoder1)
        logger.info(DCP_model.Prediction_0_1)
        logger.info(optimizer)
        DCP_model.to(device)

        # DCP_model.autoencoder1.to(device), DCP_model.autoencoder2.to(device), DCP_model.autoencoder3.to(device)
        # DCP_model.a2b.to(device), DCP_model.b2a.to(device)
        # DCP_model.b2c.to(device), DCP_model.c2b.to(device)
        # DCP_model.a2c.to(device), DCP_model.c2a.to(device)
        if config['type'] == 'CG':
            acc, nmi, ari = DCP_model.train_completegraph(config, logger, accumulated_metrics, X_list_new,
                                                          Y_list, mask, optimizer, device)
        elif config['type'] == 'CV':
            acc, nmi, ari = DCP_model.train_coreview(config, logger, accumulated_metrics, X_list_new,
                                                     Y_list, mask, optimizer, device)
        else:
            raise ValueError('Training type not match!')

        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)

        print(time.time() - start)

    logger.info('--------------------Training over--------------------')

    acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)


if __name__ == '__main__':
    main()
