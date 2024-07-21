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
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
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
    label_raw = Y_list[0]

    fold_acc, fold_precision, fold_f_measure = [], [], []
    for data_seed in range(1, args.test_time + 1):
        # Get Mask
        start = time.time()
        np.random.seed(data_seed)

        data = np.concatenate(X_list, axis=1)
        x_train, x_test, labels_train, labels_test = train_test_split(data, label_raw, test_size=0.2)

        train_views = []
        test_views = []
        current_index = 0

        for i in range(config['view']):
            len_view = X_list[i].shape[1]
            train_views.append(x_train[:, current_index:current_index + len_view])
            test_views.append(x_test[:, current_index:current_index + len_view])
            current_index += len_view

        mask_train = get_mask(config['view'], train_views[0].shape[0], config['missing_rate'])
        mask_test = get_mask(config['view'], test_views[0].shape[0], config['missing_rate'])

        # mask every view
        for i in range(config['view']):
            train_views[i] = train_views[i] * mask_train[:, i][:, np.newaxis]
            test_views[i] = test_views[i] * mask_test[:, i][:, np.newaxis]

        train_views = [torch.from_numpy(view).float().to(device) for view in train_views]
        test_views = [torch.from_numpy(view).float().to(device) for view in test_views]
        mask_train = torch.from_numpy(mask_train).long().to(device)
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
        DCP_model = DCPMultiView(config)
        optimizer = torch.optim.Adam(DCP_model.parameters(), lr=config['training']['lr'])

        logger.info(DCP_model.autoencoder1)
        logger.info(DCP_model.Prediction_0_1)
        logger.info(optimizer)
        DCP_model.to(device)

        if config['type'] == 'CG':
            acc, precision, f_measure = DCP_model.train_completegraph_supervised(config, logger, accumulated_metrics, train_views, test_views,
                                                                                 labels_train, labels_test, mask_train, mask_test, optimizer, device)
        elif config['type'] == 'CV':
            acc, precision, f_measure = DCP_model.train_coreview_supervised(config, logger, accumulated_metrics, train_views, test_views,
                                                                            labels_train, labels_test, mask_train, mask_test, optimizer, device)
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
