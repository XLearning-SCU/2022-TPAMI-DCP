import os
import random
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import util


def load_data(config):
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []

    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene_15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse_21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)  # 30000
        for view in [1, 2]:
            x = train_x[view][index]
            y = np.squeeze(mat['Y']).astype('int')[index]
            X_list.append(x)
            Y_list.append(y)


    elif data_name in ['NoisyMNIST']:
        data = sio.loadmat('./data/NoisyMNIST.mat')
        train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
        X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
        X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
        Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))

    elif data_name in ['DHA', 'UWA30']:
        train_data = data_loader_HAR(data_name)
        train_data.read_train()
        train_data_x, train_data_y, test_data_x, test_data_y, label = train_data.get_data()
        X_list.append(np.concatenate([train_data_x, test_data_x], axis=0))
        X_list.append(np.concatenate([train_data_y, test_data_y], axis=0))
        Y_list.append(label)
        Y_list.append(label)

    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X = mat['X'][0]
        for view in [3, 4]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            y = np.squeeze(mat['Y']).astype('int')
            X_list.append(x)
            Y_list.append(y)

    return X_list, Y_list


def load_multiview_data(config):
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []

    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene_15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))  # 20
        X_list.append(X[1].astype('float32'))  # 59
        X_list.append(X[2].astype('float32'))  # 40
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse_21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)
        for view in [1, 2, 0]:
            x = train_x[view][index]
            y = np.squeeze(mat['Y']).astype('int')[index]
            X_list.append(x)
            Y_list.append(y)

    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X = mat['X'][0]
        for view in [3, 4, 5]:
            # 48, 40, 254, 1984, 512, 928
            # Gabor, wavelet, centrist, hog, gist, lbp
            x = X[view]
            x = util.normalize(x).astype('float32')
            y = np.squeeze(mat['Y']).astype('int')
            X_list.append(x)
            Y_list.append(y)

    return X_list, Y_list


class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # assert images.shape[3] == 1
            # images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]


def load_NoisyMNIST():
    data = sio.loadmat('./data/NoisyMNIST.mat')

    train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])

    tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])

    test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])

    return train, tune, test


class data_loader_HAR:

    def __init__(self, database_name='DHA'):

        self.filename = database_name
        self.data_x = []  # training and testing RGB + depth feature
        self.data_label = []  # training and testing depth label
        self.train_data_x = []  # training depth feature
        self.train_data_y = []  # training RGB feature
        self.train_data_label = []  # training label
        self.test_data_x = []  # testing depth feature
        self.test_data_y = []  # testing RGB feature
        self.test_data_label = []  # testing label
        # self.train_data_xy = []  # training RGB + depth feature
        # self.test_data_xy = []  # testing RGB + depth feature
        self.cluster = 0

    def read_train(self):

        # Depth feature -> 110-dimension
        # RGB feature -> 3x2048 dimension
        feature_num1 = 110
        feature_num2 = 3 * 2048
        feature_num = feature_num1 + feature_num2
        num = 0

        # load .csv file for training
        f = open('data/' + self.filename + '_total_train.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.train_data_x.append(row[0:feature_num1])
            self.train_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            # self.train_data_xy.append(row[0:feature_num1 + feature_num2])
            self.train_data_label.append(row[feature_num1 + feature_num2:])
        f.close()

        # load .csv file for training
        f = open('data/' + self.filename + '_total_test.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.test_data_x.append(row[0:feature_num1])
            self.test_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            # self.test_data_xy.append(row[0:feature_num1 + feature_num2])
            self.test_data_label.append(row[feature_num1 + feature_num2:])
        f.close()

        # random split training and test data
        train_data_x, train_data_y, test_data_x, test_data_y, label = self.get_data()
        data_x = np.concatenate([train_data_x, test_data_x], axis=0)
        data_y = np.concatenate([train_data_y, test_data_y], axis=0)
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y, self.train_data_label, self.test_data_label = train_test_split(
            data_x, data_y, label, test_size=0.5)

        # got the sample number
        self.sample_total_num = len(self.data_x)
        self.sample_train_num = len(self.train_data_x)
        self.sample_test_num = len(self.test_data_x)
        print(self.sample_total_num)

        self.cluster = len(self.data_label[0])

    def get_data(self):
        train_data_x = np.array(self.train_data_x)
        train_data_y = np.array(self.train_data_y)
        test_data_x = np.array(self.test_data_x)
        test_data_y = np.array(self.test_data_y)

        label = np.concatenate([self.train_data_label, self.test_data_label], axis=0)
        # label_new = [np.argmax(one_hot) for one_hot in label]

        return train_data_x, train_data_y, test_data_x, test_data_y, label

    # randomly choose _batch_size RGB and depth feature in the training set
    def train_next_batch(self, _batch_size):
        xx = []  # training batch of depth features
        yy = []  # training batch of RGB features
        zz = []  # training batch of labels
        for sample_num in random.sample(range(self.sample_train_num), _batch_size):
            xx.append(self.train_data_x[sample_num])
            yy.append(self.train_data_y[sample_num])
            zz.append(self.train_data_label[sample_num])
        return yy, xx, zz

    # randomly choose _batch_size RGB and depth feature in the testing set
    def test_next_batch(self, _batch_size):
        xx = []  # testing batch of depth features
        yy = []  # testing batch of RGB features
        zz = []  # testing batch of labels
        for sample_num in random.sample(range(self.sample_test_num), _batch_size):
            xx.append(self.test_data_x[sample_num])
            yy.append(self.test_data_y[sample_num])
            zz.append(self.test_data_label[sample_num])
        return yy, xx, zz
