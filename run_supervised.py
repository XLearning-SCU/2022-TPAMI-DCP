import argparse
import itertools
import time
import torch

from model import DCP
from utils.get_mask import get_mask
from utils.util import cal_classify
from utils.logger_ import get_logger
from utils.datasets import *
from configure.configure_supervised import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")


dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "NoisyMNIST",
    3: "LandUse_21",
}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='3', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
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
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]
    label_raw = Y_list[0]


    fold_acc, fold_precision, fold_f_measure = [], [], []
    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        len1 = x1_train_raw.shape[1]
        data = np.concatenate([x1_train_raw, x2_train_raw], axis=1)

        x_train, x_test, labels_train, labels_test = train_test_split(data, label_raw, test_size=0.2)


        x1_train = x_train[:, :len1]
        x2_train = x_train[:, len1:]
        x1_test = x_test[:, :len1]
        x2_test = x_test[:, len1:]

        # Get Mask
        mask_train = get_mask(2, x1_train.shape[0], config['missing_rate'])
        x1_train = x1_train * mask_train[:, 0][:, np.newaxis]
        x2_train = x2_train * mask_train[:, 1][:, np.newaxis]

        mask_test = get_mask(2, x1_test.shape[0], config['missing_rate'])
        x1_test = x1_test * mask_test[:, 0][:, np.newaxis]
        x2_test = x2_test * mask_test[:, 1][:, np.newaxis]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        mask_train = torch.from_numpy(mask_train).long().to(device)
        x1_test = torch.from_numpy(x1_test).float().to(device)
        x2_test = torch.from_numpy(x2_test).float().to(device)
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
        DCP_model = DCP(config)
        optimizer = torch.optim.Adam(
            itertools.chain(DCP_model.autoencoder1.parameters(), DCP_model.autoencoder2.parameters(),
                            DCP_model.img2txt.parameters(), DCP_model.txt2img.parameters()),
            lr=config['training']['lr'])

        # Print the models
        logger.info(DCP_model.autoencoder1)
        logger.info(DCP_model.img2txt)
        logger.info(optimizer)

        DCP_model.autoencoder1.to(device), DCP_model.autoencoder2.to(device)
        DCP_model.img2txt.to(device), DCP_model.txt2img.to(device)

        # Training
        acc, precision, f_measure = DCP_model.train_supervised(config, logger, accumulated_metrics, x1_train,
                                                                      x2_train, x1_test,
                                                                      x2_test, labels_train,
                                                                      labels_test, mask_train, mask_test, optimizer,
                                                                      device)
        fold_acc.append(acc)
        fold_precision.append(precision)
        fold_f_measure.append(f_measure)

        print(time.time() - start)
    logger.info('--------------------Training over--------------------')
    acc, precision, f_measure = cal_classify(logger, fold_acc, fold_precision, fold_f_measure)



if __name__ == '__main__':
    main()
