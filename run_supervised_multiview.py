import argparse
import itertools
import time
import torch

from model_multiview import DCPMultiView
from utils.get_mask import get_mask
from utils.util import cal_classify
from utils.logger_ import get_logger
from utils.datasets import *
from configure.configure_supervised_multiview import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")

dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "LandUse_21",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='2', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)  # 使用第一, 三块GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] =args.print_num
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
    label_raw = Y_list[0]

    fold_acc, fold_precision, fold_f_measure = [], [], []
    for data_seed in range(1, args.test_time + 1):
        # Get Mask
        start = time.time()
        np.random.seed(data_seed)

        len1 = x1_train_raw.shape[1]
        len2 = x1_train_raw.shape[1] + x2_train_raw.shape[1]
        data = np.concatenate([x1_train_raw, x2_train_raw, x3_train_raw], axis=1)

        x_train, x_test, labels_train, labels_test = train_test_split(data, label_raw, test_size=0.2)

        x1_train = x_train[:, :len1]
        x2_train = x_train[:, len1:len2]
        x3_train = x_train[:, len2:]

        x1_test = x_test[:, :len1]
        x2_test = x_test[:, len1:len2]
        x3_test = x_test[:, len2:]

        mask_train = get_mask(3, x1_train.shape[0], config['missing_rate'])
        x1_train = x1_train * mask_train[:, 0][:, np.newaxis]
        x2_train = x2_train * mask_train[:, 1][:, np.newaxis]
        x3_train = x3_train * mask_train[:, 2][:, np.newaxis]

        mask_test = get_mask(3, x1_test.shape[0], config['missing_rate'])
        x1_test = x1_test * mask_test[:, 0][:, np.newaxis]
        x2_test = x2_test * mask_test[:, 1][:, np.newaxis]
        x3_test = x3_test * mask_test[:, 2][:, np.newaxis]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        x3_train = torch.from_numpy(x3_train).float().to(device)
        mask_train = torch.from_numpy(mask_train).long().to(device)
        x1_test = torch.from_numpy(x1_test).float().to(device)
        x2_test = torch.from_numpy(x2_test).float().to(device)
        x3_test = torch.from_numpy(x3_test).float().to(device)
        mask_test = torch.from_numpy(mask_test).long().to(device)

        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)

        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        # Set random seeds
        if config['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['seed']
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build model
        DCP = DCPMultiView(config)
        optimizer = torch.optim.Adam(
            itertools.chain(DCP.autoencoder1.parameters(), DCP.autoencoder2.parameters(),
                            DCP.autoencoder3.parameters(),
                            DCP.a2b.parameters(), DCP.b2a.parameters(),
                            DCP.a2c.parameters(), DCP.c2a.parameters(),
                            DCP.b2c.parameters(), DCP.c2b.parameters()
                            ),
            lr=config['training']['lr'])

        logger.info(DCP.autoencoder1)
        logger.info(DCP.a2b)
        logger.info(optimizer)

        DCP.autoencoder1.to(device), DCP.autoencoder2.to(device), DCP.autoencoder3.to(device)
        DCP.a2b.to(device), DCP.b2a.to(device)
        DCP.b2c.to(device), DCP.c2b.to(device)
        DCP.a2c.to(device), DCP.c2a.to(device)

        if config['type'] == 'CG':
            acc, precision, f_measure = DCP.train_completegraph_supervised(config, logger, accumulated_metrics,
                                                                           x1_train, x2_train, x3_train, x1_test,
                                                                           x2_test, x3_test, labels_train,
                                                                           labels_test, mask_train, mask_test,
                                                                           optimizer, device)
        elif config['type'] == 'CV':
            acc, precision, f_measure = DCP.train_coreview_supervised(config, logger, accumulated_metrics,
                                                                      x1_train, x2_train, x3_train, x1_test,
                                                                      x2_test, x3_test, labels_train,
                                                                      labels_test, mask_train, mask_test,
                                                                      optimizer, device)
        else:
            raise ValueError('Training type not match!')

        fold_acc.append(acc)
        fold_precision.append(precision)
        fold_f_measure.append(f_measure)
        print(time.time() - start)

    logger.info('--------------------Training over--------------------')
    acc, precision, f_measure = cal_classify(logger, fold_acc, fold_precision, fold_f_measure)


if __name__ == '__main__':
    main()
