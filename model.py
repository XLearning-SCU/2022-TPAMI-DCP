from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import instance_contrastive_Loss, category_contrastive_loss
from utils import clustering, classify
from utils.next_batch import next_batch_gt, next_batch


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class DCP():
    # Dual contrastive prediction
    def __init__(self, config):
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])

        # Dual predictions.
        # To illustrate easily, we use "img" and "txt" to denote two different views.
        self.img2txt = Prediction(self._dims_view1)
        self.txt2img = Prediction(self._dims_view2)

    def train_clustering(self, config, logger, accumulated_metrics, x1_train,
                         x2_train, Y_list, mask, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information
              accumulated_metrics: list of metrics
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari

        """
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # Get complete data for training
        flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
        flag = (flag[:, 1] + flag[:, 0]) == 2
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]

        for k in range(epochs_total):
            X1, X2 = shuffle(train_view1, train_view2)
            all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_No in next_batch(X1, X2, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                reconstruction_loss = recon1 + recon2

                # Cross-view Contrastive_Loss
                z_1, z_2 = z_half1, z_half2
                loss_icl = instance_contrastive_Loss(z_1, z_2, config['training']['alpha'])

                # Cross-view Dual-Prediction Loss
                img2txt, _ = self.img2txt(z_half1)
                txt2img, _ = self.txt2img(z_half2)
                recon3 = F.mse_loss(img2txt, z_half2)
                recon4 = F.mse_loss(txt2img, z_half1)
                dualprediction_loss = (recon3 + recon4)

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += recon3.item()
                map2 += recon4.item()
                all_icl += loss_icl.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, all1, all2, map1, map2, all_icl, all0)

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval()
                    self.img2txt.eval(), self.txt2img.eval()

                    img_idx_eval = mask[:, 0] == 1
                    txt_idx_eval = mask[:, 1] == 1
                    img_missing_idx_eval = mask[:, 0] == 0
                    txt_missing_idx_eval = mask[:, 1] == 0

                    imgs_latent_eval = self.autoencoder1.encoder(x1_train[img_idx_eval])
                    txts_latent_eval = self.autoencoder2.encoder(x2_train[txt_idx_eval])

                    latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)

                    if x2_train[img_missing_idx_eval].shape[0] != 0:
                        img_missing_latent_eval = self.autoencoder2.encoder(x2_train[img_missing_idx_eval])
                        txt_missing_latent_eval = self.autoencoder1.encoder(x1_train[txt_missing_idx_eval])

                        txt2img_recon_eval, _ = self.txt2img(img_missing_latent_eval)
                        img2txt_recon_eval, _ = self.img2txt(txt_missing_latent_eval)

                        latent_code_img_eval[img_missing_idx_eval] = txt2img_recon_eval
                        latent_code_txt_eval[txt_missing_idx_eval] = img2txt_recon_eval

                    latent_code_img_eval[img_idx_eval] = imgs_latent_eval
                    latent_code_txt_eval[txt_idx_eval] = txts_latent_eval
                    latent_fusion = torch.cat([latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list, accumulated_metrics['acc'],
                                                  accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'], accumulated_metrics['f-mea'])
                    logger.info("\033[2;29m" + 'trainingset_view_concat ' + str(scores) + "\033[0m")

                    self.autoencoder1.train(), self.autoencoder2.train()
                    self.img2txt.train(), self.txt2img.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]

    def train_supervised(self, config, logger, accumulated_metrics, x1_train, x2_train, x1_test, x2_test, labels_train,
                         labels_test, mask_train, mask_test, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information
              accumulated_metrics: list of metrics
              x1_train/x1_test: training/test data of view 1
              x2_train/x2_test: training/test data of view 2
              labels_train/test: labels of training/test data
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              classification performance: acc, precision, f-measure

        """
        epochs = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # Get complete data for training
        flag = (torch.LongTensor([1, 1]).to(device) == mask_train).int()
        flag = (flag[:, 1] + flag[:, 0]) == 2
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        GT = torch.from_numpy(labels_train).long().to(device)[flag]

        classes = np.unique(np.concatenate([labels_train, labels_test])).size

        flag_gt = False
        if torch.min(GT) == 1:
            flag_gt = True
        for k in range(epochs):
            X1, X2, gt = shuffle(train_view1, train_view2, GT)
            all_ccl, all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, gt_batch, batch_No in next_batch_gt(X1, X2, gt, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                reconstruction_loss = recon1 + recon2

                # Instance-level contrastive loss
                z_1, z_2 = z_half1, z_half2
                loss_icl = instance_contrastive_Loss(z_1, z_2, config['training']['alpha'])

                # Category-level contrastive loss
                loss_ccl = category_contrastive_loss(torch.cat([z_half1, z_half2], dim=1), gt_batch,
                                                     classes, flag_gt)

                # Cross-view Dual-Prediction Loss
                img2txt, _ = self.img2txt(z_half1)
                txt2img, _ = self.txt2img(z_half2)
                recon3 = F.mse_loss(img2txt, z_half2)
                recon4 = F.mse_loss(txt2img, z_half1)
                dualprediction_loss = (recon3 + recon4)



                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss = all_loss + config['training']['lambda1'] * dualprediction_loss
                    all_loss += loss_ccl

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all_ccl += loss_ccl.item()
                all_icl += loss_icl.item()
                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += recon3.item()
                map2 += recon4.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> Loss_icl = {:.4e} ===> Loss_ccl = {:.4e} ===> All loss = {:.4e}" \
                .format((k + 1), epochs, all1, all2, map1, map2, all_icl, all_ccl, all0)

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval()
                    self.img2txt.eval(), self.txt2img.eval()

                    # Training data
                    img_idx_eval = mask_train[:, 0] == 1
                    txt_idx_eval = mask_train[:, 1] == 1
                    img_missing_idx_eval = mask_train[:, 0] == 0
                    txt_missing_idx_eval = mask_train[:, 1] == 0

                    imgs_latent_eval = self.autoencoder1.encoder(x1_train[img_idx_eval])
                    txts_latent_eval = self.autoencoder2.encoder(x2_train[txt_idx_eval])

                    latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(device)
                    latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(device)

                    if x2_train[img_missing_idx_eval].shape[0] != 0:
                        img_missing_latent_eval = self.autoencoder2.encoder(x2_train[img_missing_idx_eval])
                        txt_missing_latent_eval = self.autoencoder1.encoder(x1_train[txt_missing_idx_eval])

                        txt2img_recon_eval, _ = self.txt2img(img_missing_latent_eval)
                        img2txt_recon_eval, _ = self.img2txt(txt_missing_latent_eval)

                        latent_code_img_eval[img_missing_idx_eval] = txt2img_recon_eval
                        latent_code_txt_eval[txt_missing_idx_eval] = img2txt_recon_eval

                    latent_code_img_eval[img_idx_eval] = imgs_latent_eval
                    latent_code_txt_eval[txt_idx_eval] = txts_latent_eval
                    latent_fusion_train = torch.cat([latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()

                    # Test data
                    img_idx_eval = mask_test[:, 0] == 1
                    txt_idx_eval = mask_test[:, 1] == 1
                    img_missing_idx_eval = mask_test[:, 0] == 0
                    txt_missing_idx_eval = mask_test[:, 1] == 0

                    imgs_latent_eval = self.autoencoder1.encoder(x1_test[img_idx_eval])
                    txts_latent_eval = self.autoencoder2.encoder(x2_test[txt_idx_eval])

                    latent_code_img_eval = torch.zeros(x1_test.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_txt_eval = torch.zeros(x2_test.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)

                    if x2_test[img_missing_idx_eval].shape[0] != 0:
                        img_missing_latent_eval = self.autoencoder2.encoder(x2_test[img_missing_idx_eval])
                        txt_missing_latent_eval = self.autoencoder1.encoder(x1_test[txt_missing_idx_eval])

                        txt2img_recon_eval, _ = self.txt2img(img_missing_latent_eval)
                        img2txt_recon_eval, _ = self.img2txt(txt_missing_latent_eval)

                        latent_code_img_eval[img_missing_idx_eval] = txt2img_recon_eval
                        latent_code_txt_eval[txt_missing_idx_eval] = img2txt_recon_eval

                    latent_code_img_eval[img_idx_eval] = imgs_latent_eval
                    latent_code_txt_eval[txt_idx_eval] = txts_latent_eval

                    latent_fusion_test = torch.cat([latent_code_img_eval, latent_code_txt_eval],
                                                   dim=1).cpu().numpy()

                    from sklearn.metrics import accuracy_score
                    from sklearn.metrics import precision_score
                    from sklearn.metrics import f1_score
                    label_pre = classify.ave(latent_fusion_train, latent_fusion_test, labels_train)

                    scores = accuracy_score(labels_test, label_pre)

                    precision = precision_score(labels_test, label_pre, average='macro')
                    precision = np.round(precision, 2)

                    f_score = f1_score(labels_test, label_pre, average='macro')
                    f_score = np.round(f_score, 2)

                    accumulated_metrics['acc'].append(scores)
                    accumulated_metrics['precision'].append(precision)
                    accumulated_metrics['f_measure'].append(f_score)
                    logger.info('\033[2;29m Accuracy on the test set is {:.4f}'.format(scores))
                    logger.info('\033[2;29m Precision on the test set is {:.4f}'.format(precision))
                    logger.info('\033[2;29m F_score on the test set is {:.4f}'.format(f_score))

                    self.autoencoder1.train(), self.autoencoder2.train()
                    self.img2txt.train(), self.txt2img.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][
            -1]

    def train_HAR(self, config, logger, accumulated_metrics, train_data, optimizer, device):
        """Training the Human action recognition.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information
              accumulated_metrics: list of metrics
              train_data: data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              classification performance: Accuracy

        """
        epochs = config['training']['epoch']
        batch_size = config['training']['batch_size']

        classes = train_data.cluster
        flag_gt = False

        for k in range(epochs):
            all_ccl, all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0, 0
            batch_x1, batch_x2, gt_batch = train_data.train_next_batch(batch_size)

            batch_x1 = torch.from_numpy(np.array(batch_x1)).float().to(device)
            batch_x2 = torch.from_numpy(np.array(batch_x2)).float().to(device)
            gt_batch = [np.argmax(one_hot) for one_hot in gt_batch]
            gt_batch = torch.from_numpy(np.array(gt_batch)).to(device)

            z_half1 = self.autoencoder1.encoder(batch_x1)
            z_half2 = self.autoencoder2.encoder(batch_x2)
            recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
            recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
            reconstruction_loss = recon1 + recon2

            z_1, z_2 = z_half1, z_half2
            loss_icl = instance_contrastive_Loss(z_1, z_2, config['training']['alpha'])

            img2txt, _ = self.img2txt(z_half1)
            txt2img, _ = self.txt2img(z_half2)
            recon3 = F.mse_loss(img2txt, z_half2)
            recon4 = F.mse_loss(txt2img, z_half1)
            dualprediction_loss = (recon3 + recon4)

            loss_ccl = category_contrastive_loss(torch.cat([z_half1, z_half2], dim=1), gt_batch,
                                                 classes, flag_gt)
            all_loss = loss_icl + reconstruction_loss * config['training']['lambda2']

            if k >= config['training']['start_dual_prediction']:
                all_loss += loss_ccl
                all_loss = all_loss + config['training']['lambda1'] * dualprediction_loss

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            all_icl += loss_icl.item()
            all_ccl += loss_ccl.item()
            all0 += all_loss.item()
            all1 += recon1.item()
            all2 += recon2.item()
            map1 += recon3.item()
            map2 += recon4.item()

            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> Loss_icl = {:.4e} ===> Loss_ccl = {:.4e} ===> All loss = {:.4e}" \
                .format((k + 1), epochs, all1, all2, map1, map2, all_icl, all_ccl, all0)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # Evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval()
                    self.img2txt.eval(), self.txt2img.eval()

                    # Training data
                    batch_x1 = train_data.train_data_y
                    batch_x2 = train_data.train_data_x
                    batch_x1 = torch.from_numpy(np.array(batch_x1)).float().to(device)
                    batch_x2 = torch.from_numpy(np.array(batch_x2)).float().to(device)

                    labels_train = train_data.train_data_label
                    labels_train = [np.argmax(one_hot) for one_hot in labels_train]
                    labels_train = np.array(labels_train)

                    imgs_latent_eval = self.autoencoder1.encoder(batch_x1)
                    txts_latent_eval = self.autoencoder2.encoder(batch_x2)

                    latent_code_img_eval = imgs_latent_eval
                    latent_code_txt_eval = txts_latent_eval

                    latent_img_train = latent_code_img_eval.cpu().numpy()
                    latent_txt_train = latent_code_txt_eval.cpu().numpy()
                    latent_fusion_train = torch.cat([latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()

                    # Test data

                    batch_x1 = train_data.test_data_y
                    batch_x2 = train_data.test_data_x
                    gt_batch = train_data.test_data_label

                    batch_x1 = torch.from_numpy(np.array(batch_x1)).float().to(device)
                    batch_x2 = torch.from_numpy(np.array(batch_x2)).float().to(device)
                    gt_batch = [np.argmax(one_hot) for one_hot in gt_batch]

                    imgs_latent_eval = self.autoencoder1.encoder(batch_x1)
                    txts_latent_eval = self.autoencoder2.encoder(batch_x2)

                    # R->D
                    latent_code_img_eval_RD = imgs_latent_eval
                    txt_recover_latent_eval_RD, _ = self.img2txt(imgs_latent_eval)
                    latent_code_txt_eval_RD = txt_recover_latent_eval_RD
                    latent_fusion_test_RD = torch.cat([latent_code_img_eval_RD, latent_code_txt_eval_RD],
                                                      dim=1).cpu().numpy()

                    # D->R
                    latent_code_txt_eval_DR = txts_latent_eval
                    img_recover_latent_eval_DR, _ = self.txt2img(txts_latent_eval)
                    latent_code_img_eval_DR = img_recover_latent_eval_DR

                    latent_fusion_test_DR = torch.cat([latent_code_img_eval_DR, latent_code_txt_eval_DR],
                                                      dim=1).cpu().numpy()
                    # R+D
                    latent_fusion_test = torch.cat([imgs_latent_eval, txts_latent_eval],
                                                   dim=1).cpu().numpy()

                    from sklearn.metrics import accuracy_score

                    label_pre = classify.vote(latent_fusion_train, latent_fusion_test_RD, labels_train)
                    scores_RD = accuracy_score(gt_batch, label_pre)
                    accumulated_metrics['RGB'].append(scores_RD)

                    label_pre = classify.vote(latent_fusion_train, latent_fusion_test_DR, labels_train)
                    scores_DR = accuracy_score(gt_batch, label_pre)
                    accumulated_metrics['Depth'].append(scores_DR)

                    label_pre = classify.vote(latent_fusion_train, latent_fusion_test, labels_train)
                    scores = accuracy_score(gt_batch, label_pre)
                    accumulated_metrics['RGB-D'].append(scores)

                    label_pre = classify.vote(latent_img_train, imgs_latent_eval.cpu().numpy(), labels_train)
                    scores_onlyrgb = accuracy_score(gt_batch, label_pre)
                    accumulated_metrics['onlyRGB'].append(scores_onlyrgb)

                    label_pre = classify.vote(latent_txt_train, txts_latent_eval.cpu().numpy(), labels_train)
                    scores_onlydepth = accuracy_score(gt_batch, label_pre)

                    accumulated_metrics['onlyDepth'].append(scores_onlydepth)

                    logger.info('\033[2;29m RGB   Accuracy on the test set is {:.4f}'.format(scores_RD))
                    logger.info('\033[2;29m Depth Accuracy on the test set is {:.4f}'.format(scores_DR))
                    logger.info('\033[2;29m RGB+D Accuracy on the test set is {:.4f}'.format(scores))
                    logger.info('\033[2;29m onlyRGB Accuracy on the test set is {:.4f}'.format(scores_onlyrgb))
                    logger.info('\033[2;29m onlyDepth Accuracy on the test set is {:.4f}'.format(scores_onlydepth))

                    self.autoencoder1.train(), self.autoencoder2.train()
                    self.img2txt.train(), self.txt2img.train()

        return accumulated_metrics['RGB'][-1], accumulated_metrics['Depth'][-1], accumulated_metrics['RGB-D'][
            -1], accumulated_metrics['onlyRGB'][-1], accumulated_metrics['onlyDepth'][-1]
