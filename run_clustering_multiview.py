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
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]
    x3_train_raw = X_list[2]


    fold_acc, fold_nmi, fold_ari = [], [], []
    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        # Get Mask
        mask = get_mask(3, x1_train_raw.shape[0], config['missing_rate'])
        x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
        x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]
        x3_train = x3_train_raw * mask[:, 2][:, np.newaxis]

        mask_num = []
        mask_num.append(np.arange(x1_train_raw.shape[0])[~np.array(mask[:, 0], dtype=bool)] + 1)
        mask_num.append(np.arange(x1_train_raw.shape[0])[~np.array(mask[:, 1], dtype=bool)] + 1)
        mask_num.append(np.arange(x1_train_raw.shape[0])[~np.array(mask[:, 2], dtype=bool)] + 1)

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        x3_train = torch.from_numpy(x3_train).float().to(device)

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
        optimizer = torch.optim.Adam(
            itertools.chain(DCP_model.autoencoder1.parameters(), DCP_model.autoencoder2.parameters(),
                            DCP_model.autoencoder3.parameters(),
                            DCP_model.a2b.parameters(), DCP_model.b2a.parameters(),
                            DCP_model.a2c.parameters(), DCP_model.c2a.parameters(),
                            DCP_model.b2c.parameters(), DCP_model.c2b.parameters()
                            ),
            lr=config['training']['lr'])

        logger.info(DCP_model.autoencoder1)
        logger.info(DCP_model.a2b)
        logger.info(optimizer)

        DCP_model.autoencoder1.to(device), DCP_model.autoencoder2.to(device), DCP_model.autoencoder3.to(device)
        DCP_model.a2b.to(device), DCP_model.b2a.to(device)
        DCP_model.b2c.to(device), DCP_model.c2b.to(device)
        DCP_model.a2c.to(device), DCP_model.c2a.to(device)

        if config['type'] == 'CG':
            acc, nmi, ari = DCP_model.train_completegraph(config, logger, accumulated_metrics, x1_train, x2_train,
                                                          x3_train,
                                                          Y_list, mask, optimizer, device)
        elif config['type'] == 'CV':
            acc, nmi, ari = DCP_model.train_coreview(config, logger, accumulated_metrics, x1_train, x2_train, x3_train,
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
