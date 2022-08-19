from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import instance_contrastive_Loss, category_contrastive_loss
from utils import clustering, classify
from utils.next_batch import next_batch_gt, next_batch, next_batch_3view, next_batch_gt_3view
from model import Autoencoder, Prediction


class DCPMultiView():
    # Dual contrastive prediction for multi-view
    def __init__(self, config):
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        self._config = config

        self._latent_dim = config['Autoencoder']['arch1'][-1]

        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']
        self._dims_view3 = [self._latent_dim] + self._config['Prediction']['arch3']

        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder3 = Autoencoder(config['Autoencoder']['arch3'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])

        self.a2b = Prediction(self._dims_view1)
        self.b2a = Prediction(self._dims_view2)

        self.a2c = Prediction(self._dims_view1)
        self.c2a = Prediction(self._dims_view3)

        self.b2c = Prediction(self._dims_view2)
        self.c2b = Prediction(self._dims_view3)

    def train_completegraph(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train, Y_list, mask,
                            optimizer, device):
        """Training the model with complete graph for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              x1_train: data of view 1
              x2_train: data of view 2
              x3_train: data of view 3
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari

        """

        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # select the complete samples
        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_view3 = x3_train[flag]

        for k in range(epochs_total):
            X1, X2, X3 = shuffle(train_view1, train_view2, train_view3)
            all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_x3, batch_No in next_batch_3view(X1, X2, X3, batch_size):
                # get the hidden states for each view
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)

                reconstruction_loss = recon1 + recon2 + recon3

                # Instance-level Contrastive_Loss
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl3 = instance_contrastive_Loss(z_half2, z_half3, config['training']['alpha'])
                loss_icl = (loss_icl1 + 0.1 * loss_icl2 + 0.1 * loss_icl3) / 3

                # Cross-view Dual-Prediction Loss
                a2b, _ = self.a2b(z_half1)
                b2a, _ = self.b2a(z_half2)
                a2c, _ = self.a2c(z_half1)
                c2a, _ = self.c2a(z_half3)
                b2c, _ = self.b2c(z_half2)
                c2b, _ = self.c2b(z_half3)

                pre1 = F.mse_loss(a2b, z_half2)
                pre2 = F.mse_loss(b2a, z_half1)
                pre3 = F.mse_loss(a2c, z_half3)
                pre4 = F.mse_loss(c2a, z_half1)
                pre5 = F.mse_loss(b2c, z_half3)
                pre6 = F.mse_loss(c2b, z_half2)
                dualprediction_loss = (pre1 + pre2 + pre3 + pre4 + pre5 + pre6) / 3

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += pre1.item()
                map2 += pre2.item()
                all_icl += loss_icl.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, all1, all2, map1, map2, all_icl, all0)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
                    self.a2b.eval(), self.b2a.eval()
                    self.b2c.eval(), self.c2b.eval()
                    self.a2c.eval(), self.c2a.eval()

                    # get the missing index
                    a_idx_eval = mask[:, 0] == 1
                    b_idx_eval = mask[:, 1] == 1
                    c_idx_eval = mask[:, 2] == 1
                    a_missing_idx_eval = mask[:, 0] == 0
                    b_missing_idx_eval = mask[:, 1] == 0
                    c_missing_idx_eval = mask[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_train[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_train[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_train[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    # for view a
                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    # for view b
                    if b_missing_idx_eval.sum() != 0:
                        bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
                        bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

                        bno_aonlyhas = self.autoencoder1.encoder(x1_train[bno_aonlyhas_idx])
                        bno_aonlyhas, _ = self.a2b(bno_aonlyhas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_train[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2b(bno_conlyhas)

                        bno_acbothhas_1 = self.autoencoder1.encoder(x1_train[bno_acbothhas_idx])
                        bno_acbothhas_2 = self.autoencoder3.encoder(x3_train[bno_acbothhas_idx])
                        bno_acbothhas = (self.a2b(bno_acbothhas_1)[0] + self.c2b(bno_acbothhas_2)[0]) / 2.0

                        latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
                        latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

                    # for view c
                    if c_missing_idx_eval.sum() != 0:
                        cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
                        cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_train[cno_aonlyhas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_train[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2c(cno_bonlyhas)

                        cno_abbothhas_1 = self.autoencoder1.encoder(x1_train[cno_abbothhas_idx])
                        cno_abbothhas_2 = self.autoencoder2.encoder(x2_train[cno_abbothhas_idx])
                        cno_abbothhas = (self.a2c(cno_abbothhas_1)[0] + self.b2c(cno_abbothhas_2)[0]) / 2.0

                        latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
                        latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

                    # fill the existing views
                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    # recovered fusion representation
                    latent_fusion = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                              dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list, accumulated_metrics['acc'],
                                                  accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'], accumulated_metrics['f-mea'])
                    logger.info("\033[2;29m" + 'trainingset_view_concat ' + str(scores) + "\033[0m")

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
                    self.a2b.train(), self.b2a.train()
                    self.b2c.train(), self.c2b.train()
                    self.a2c.train(), self.c2a.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][
            -1]

    def train_coreview(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train, Y_list, mask, optimizer,
                       device):
        """Training the model with cove view for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              x1_train: data of view 1
              x2_train: data of view 2
              x3_train: data of view 3
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_view3 = x3_train[flag]

        for k in range(epochs_total):
            X1, X2, X3 = shuffle(train_view1, train_view2, train_view3)
            all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_x3, batch_No in next_batch_3view(X1, X2, X3, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)

                reconstruction_loss = recon1 + recon2 + recon3

                # Instance-level Contrastive Loss
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl = (loss_icl1 + 0.1 * loss_icl2) / 2

                # Cross-view Dual-Prediction Loss
                a2b, _ = self.a2b(z_half1)
                b2a, _ = self.b2a(z_half2)
                a2c, _ = self.a2c(z_half1)
                c2a, _ = self.c2a(z_half3)

                pre1 = F.mse_loss(a2b, z_half2)
                pre2 = F.mse_loss(b2a, z_half1)
                pre3 = F.mse_loss(a2c, z_half3)
                pre4 = F.mse_loss(c2a, z_half1)

                dualprediction_loss = (pre1 + pre2 + pre3 + pre4) / 2

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += pre1.item()
                map2 += pre2.item()
                all_icl += loss_icl.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, all1, all2, map1, map2, all_icl, all0)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
                    self.a2b.eval(), self.b2a.eval()
                    self.a2c.eval(), self.c2a.eval()

                    a_idx_eval = mask[:, 0] == 1
                    b_idx_eval = mask[:, 1] == 1
                    c_idx_eval = mask[:, 2] == 1
                    a_missing_idx_eval = mask[:, 0] == 0
                    b_missing_idx_eval = mask[:, 1] == 0
                    c_missing_idx_eval = mask[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_train[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_train[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_train[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        #  to recover view b, utilizing the core view unless core view is missing
                        bno_ahas_idx = b_missing_idx_eval * a_idx_eval

                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval

                        bno_ahas = self.autoencoder1.encoder(x1_train[bno_ahas_idx])
                        bno_ahas, _ = self.a2b(bno_ahas)

                        # predicting twice
                        bno_conlyhas = self.autoencoder3.encoder(x3_train[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2a(bno_conlyhas)
                        bno_conlyhas, _ = self.a2b(bno_conlyhas)

                        latent_code_b_eval[bno_ahas_idx] = bno_ahas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas

                    if c_missing_idx_eval.sum() != 0:
                        cno_ahas_idx = c_missing_idx_eval * a_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_train[cno_ahas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_train[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2a(cno_bonlyhas)
                        cno_bonlyhas, _ = self.a2c(cno_bonlyhas)

                        latent_code_c_eval[cno_ahas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                              dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list, accumulated_metrics['acc'],
                                                  accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'], accumulated_metrics['f-mea'])
                    logger.info("\033[2;29m" + 'trainingset_view_concat ' + str(scores) + "\033[0m")

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
                    self.a2b.train(), self.b2a.train()
                    self.a2c.train(), self.c2a.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][
            -1]

    def train_completegraph_supervised(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train, x1_test,
                                       x2_test, x3_test, labels_train, labels_test, mask_train, mask_test, optimizer,
                                       device):
        """Training the model with complete graph for classification

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              x*_train: training data of view *
              x*_test: test data of view *
              labels_train/test: label of training/test data
              mask *: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              classification performance: acc, precision, f-measure

        """
        epochs = config['training']['epoch']
        batch_size = config['training']['batch_size']

        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask_train == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)

        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_view3 = x3_train[flag]

        GT = torch.from_numpy(labels_train).long().to(device)[flag]
        classes = np.unique(np.concatenate([labels_train, labels_test])).size
        flag_gt = False
        if torch.min(GT) == 1:
            flag_gt = True

        for k in range(epochs):
            X1, X2, X3, gt = shuffle(train_view1, train_view2, train_view3, GT)
            all_ccl, all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_x3, gt_batch, batch_No in next_batch_gt_3view(X1, X2, X3, gt, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)
                reconstruction_loss = recon1 + recon2 + recon3

                # Instance-level contrastive loss
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl3 = instance_contrastive_Loss(z_half2, z_half3, config['training']['alpha'])
                loss_icl = (loss_icl1 + 0.1 * loss_icl2 + 0.1 * loss_icl3) / 3

                # Cross-view Dual-Prediction Loss
                a2b, _ = self.a2b(z_half1)
                b2a, _ = self.b2a(z_half2)
                a2c, _ = self.a2c(z_half1)
                c2a, _ = self.c2a(z_half3)
                b2c, _ = self.b2c(z_half2)
                c2b, _ = self.c2b(z_half3)

                pre1 = F.mse_loss(a2b, z_half2)
                pre2 = F.mse_loss(b2a, z_half1)
                pre3 = F.mse_loss(a2c, z_half3)
                pre4 = F.mse_loss(c2a, z_half1)
                pre5 = F.mse_loss(b2c, z_half3)
                pre6 = F.mse_loss(c2b, z_half2)

                dualprediction_loss = (pre1 + pre2 + pre3 + pre4 + pre5 + pre6) / 3

                # Category-level contrastive loss
                loss_ccl = category_contrastive_loss(torch.cat([z_half1, z_half2, z_half3], dim=1), gt_batch, classes,
                                                     flag_gt)

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2'] + loss_ccl
                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all_icl += loss_icl.item()
                all_ccl += loss_ccl.item()
                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += pre1.item()
                map2 += pre2.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> Loss_icl = {:.4e} ===> Los_ccl = {:.4e} ===> All loss = {:.4e}" \
                .format((k + 1), epochs, all1, all2, map1, map2, all_icl, all_ccl, all0)

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                # if True:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
                    self.a2b.eval(), self.b2a.eval()
                    self.b2c.eval(), self.c2b.eval()
                    self.a2c.eval(), self.c2a.eval()

                    # Training data
                    a_idx_eval = mask_train[:, 0] == 1
                    b_idx_eval = mask_train[:, 1] == 1
                    c_idx_eval = mask_train[:, 2] == 1
                    a_missing_idx_eval = mask_train[:, 0] == 0
                    b_missing_idx_eval = mask_train[:, 1] == 0
                    c_missing_idx_eval = mask_train[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_train[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_train[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_train[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
                        bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

                        bno_aonlyhas = self.autoencoder1.encoder(x1_train[bno_aonlyhas_idx])
                        bno_aonlyhas, _ = self.a2b(bno_aonlyhas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_train[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2b(bno_conlyhas)

                        bno_acbothhas_1 = self.autoencoder1.encoder(x1_train[bno_acbothhas_idx])
                        bno_acbothhas_2 = self.autoencoder3.encoder(x3_train[bno_acbothhas_idx])
                        bno_acbothhas = (self.a2b(bno_acbothhas_1)[0] + self.c2b(bno_acbothhas_2)[0]) / 2.0

                        latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
                        latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

                    if c_missing_idx_eval.sum() != 0:
                        #   b缺
                        cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
                        cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_train[cno_aonlyhas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_train[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2c(cno_bonlyhas)

                        cno_abbothhas_1 = self.autoencoder1.encoder(x1_train[cno_abbothhas_idx])
                        cno_abbothhas_2 = self.autoencoder2.encoder(x2_train[cno_abbothhas_idx])
                        cno_abbothhas = (self.a2c(cno_abbothhas_1)[0] + self.b2c(cno_abbothhas_2)[0]) / 2.0

                        latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
                        latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                                    dim=1).cpu().numpy()

                    # Test data
                    a_idx_eval = mask_test[:, 0] == 1
                    b_idx_eval = mask_test[:, 1] == 1
                    c_idx_eval = mask_test[:, 2] == 1
                    a_missing_idx_eval = mask_test[:, 0] == 0
                    b_missing_idx_eval = mask_test[:, 1] == 0
                    c_missing_idx_eval = mask_test[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_test[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_test[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_test[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_test.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_test.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_test.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_test[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_test[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_test[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_test[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
                        bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

                        bno_aonlyhas = self.autoencoder1.encoder(x1_test[bno_aonlyhas_idx])
                        bno_aonlyhas, _ = self.a2b(bno_aonlyhas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_test[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2b(bno_conlyhas)

                        bno_acbothhas_1 = self.autoencoder1.encoder(x1_test[bno_acbothhas_idx])
                        bno_acbothhas_2 = self.autoencoder3.encoder(x3_test[bno_acbothhas_idx])
                        bno_acbothhas = (self.a2b(bno_acbothhas_1)[0] + self.c2b(bno_acbothhas_2)[0]) / 2.0

                        latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
                        latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

                    if c_missing_idx_eval.sum() != 0:
                        cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
                        cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_test[cno_aonlyhas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_test[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2c(cno_bonlyhas)

                        cno_abbothhas_1 = self.autoencoder1.encoder(x1_test[cno_abbothhas_idx])
                        cno_abbothhas_2 = self.autoencoder2.encoder(x2_test[cno_abbothhas_idx])
                        cno_abbothhas = (self.a2c(cno_abbothhas_1)[0] + self.b2c(cno_abbothhas_2)[0]) / 2.0

                        latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
                        latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion_test = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
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

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
                    self.a2b.train(), self.b2a.train()
                    self.b2c.train(), self.c2b.train()
                    self.a2c.train(), self.c2a.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][
            -1]

    def train_coreview_supervised(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train, x1_test,
                                  x2_test, x3_test, labels_train, labels_test, mask_train, mask_test, optimizer,
                                  device):
        """Training the model with cove view for classification

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              x*_train: training data of view *
              x*_test: test data of view *
              labels_train/test: label of training/test data
              mask *: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              classification performance: acc, precision, f-measure


        """
        epochs = config['training']['epoch']
        batch_size = config['training']['batch_size']

        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask_train == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)

        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_view3 = x3_train[flag]

        GT = torch.from_numpy(labels_train).long().to(device)[flag]
        classes = np.unique(np.concatenate([labels_train, labels_test])).size
        flag_gt = False
        if torch.min(GT) == 1:
            flag_gt = True
        for k in range(epochs):
            X1, X2, X3, gt = shuffle(train_view1, train_view2, train_view3, GT)
            all_ccl, all_icl, all0, all1, all2, map1, map2 = 0, 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_x3, gt_batch, batch_No in next_batch_gt_3view(X1, X2, X3, gt, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)
                reconstruction_loss = recon1 + recon2 + recon3

                # Instance-level contrastive loss
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl = (loss_icl1 + 0.1 * loss_icl2) / 2

                # Cross-view Dual-Prediction Loss
                a2b, _ = self.a2b(z_half1)
                b2a, _ = self.b2a(z_half2)
                a2c, _ = self.a2c(z_half1)
                c2a, _ = self.c2a(z_half3)

                pre1 = F.mse_loss(a2b, z_half2)
                pre2 = F.mse_loss(b2a, z_half1)
                pre3 = F.mse_loss(a2c, z_half3)
                pre4 = F.mse_loss(c2a, z_half1)
                dualprediction_loss = (pre1 + pre2 + pre3 + pre4) / 2

                # Category-level contrastive loss
                loss_ccl = category_contrastive_loss(torch.cat([z_half1, z_half2, z_half3], dim=1), gt_batch,
                                                     classes, flag_gt)

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2'] + loss_ccl
                if k >= config['training']['start_dual_prediction']:
                    all_loss = all_loss + config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all_ccl += loss_ccl.item()
                all_icl += loss_icl.item()
                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += pre1.item()
                map2 += pre2.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> Loss_icl = {:.4e} ===> Loss_ccl = {:.4e} ===> All loss = {:.4e}" \
                .format((k + 1), epochs, all1, all2, map1, map2, all_icl, all_ccl, all0)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
                    self.a2b.eval(), self.b2a.eval()
                    self.a2c.eval(), self.c2a.eval()

                    # Training data
                    a_idx_eval = mask_train[:, 0] == 1
                    b_idx_eval = mask_train[:, 1] == 1
                    c_idx_eval = mask_train[:, 2] == 1
                    a_missing_idx_eval = mask_train[:, 0] == 0
                    b_missing_idx_eval = mask_train[:, 1] == 0
                    c_missing_idx_eval = mask_train[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_train[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_train[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_train[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        bno_ahas_idx = b_missing_idx_eval * a_idx_eval
                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval

                        bno_ahas = self.autoencoder1.encoder(x1_train[bno_ahas_idx])
                        bno_ahas, _ = self.a2b(bno_ahas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_train[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2a(bno_conlyhas)
                        bno_conlyhas, _ = self.a2b(bno_conlyhas)

                        latent_code_b_eval[bno_ahas_idx] = bno_ahas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas

                    if c_missing_idx_eval.sum() != 0:
                        cno_ahas_idx = c_missing_idx_eval * a_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_train[cno_ahas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_train[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2a(cno_bonlyhas)
                        cno_bonlyhas, _ = self.a2c(cno_bonlyhas)

                        latent_code_c_eval[cno_ahas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                                    dim=1).cpu().numpy()

                    # Test data
                    a_idx_eval = mask_test[:, 0] == 1
                    b_idx_eval = mask_test[:, 1] == 1
                    c_idx_eval = mask_test[:, 2] == 1
                    a_missing_idx_eval = mask_test[:, 0] == 0
                    b_missing_idx_eval = mask_test[:, 1] == 0
                    c_missing_idx_eval = mask_test[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_test[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_test[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_test[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_test.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_test.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_test.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        #   核心a缺失
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_test[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_test[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_test[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_test[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        bno_ahas_idx = b_missing_idx_eval * a_idx_eval

                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval

                        bno_ahas = self.autoencoder1.encoder(x1_test[bno_ahas_idx])
                        bno_ahas, _ = self.a2b(bno_ahas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_test[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2a(bno_conlyhas)
                        bno_conlyhas, _ = self.a2b(bno_conlyhas)

                        latent_code_b_eval[bno_ahas_idx] = bno_ahas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas

                    if c_missing_idx_eval.sum() != 0:
                        cno_ahas_idx = c_missing_idx_eval * a_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_test[cno_ahas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_test[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2a(cno_bonlyhas)
                        cno_bonlyhas, _ = self.a2c(cno_bonlyhas)

                        latent_code_c_eval[cno_ahas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion_test = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
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

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
                    self.a2b.train(), self.b2a.train()
                    self.a2c.train(), self.c2a.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][
            -1]
