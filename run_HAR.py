import argparse
import itertools
import time
import torch

from model import DCP
from utils.util import cal_HAR
from utils.logger_ import get_logger
from utils.datasets import *
from configure.configure_supervised import get_default_config
import collections

dataset = {
    0: "DHA",
    1: "UWA30",
}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')

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

    fold_rgb, fold_depth, fold_rgbdepth, = [], [], []
    fold_onlyrgb, fold_onlydepth = [], []

    for data_seed in range(1, args.test_time + 1):
        start = time.time()

        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        seed = config['seed'] * data_seed

        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Load data
        train_data = data_loader_HAR(config['dataset'])
        train_data.read_train()

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
        rgb, depth, rgb_depth, onlyrgb, onlydepth = DCP_model.train_HAR(config, logger, accumulated_metrics, train_data,
                                                                        optimizer, device)
        fold_rgb.append(rgb)
        fold_depth.append(depth)
        fold_rgbdepth.append(rgb_depth)
        fold_onlyrgb.append(onlyrgb)
        fold_onlydepth.append(onlydepth)

        print(time.time() - start)
    logger.info('--------------------Training over--------------------')
    cal_HAR(logger, fold_rgb, fold_depth, fold_rgbdepth, fold_onlyrgb, fold_onlydepth)


if __name__ == '__main__':
    main()
