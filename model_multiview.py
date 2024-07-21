from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import instance_contrastive_Loss, category_contrastive_loss
from utils import clustering, classify
from utils.next_batch import next_batch_multiview, next_batch_gt_multiview
from model import Autoencoder, Prediction


class DCPMultiView(torch.nn.Module):
    # Dual contrastive prediction for multi-view
    def __init__(self, config):
        super(DCPMultiView, self).__init__()
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        self._config = config
        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self.view_num = config['view']
        for i in range(self.view_num):
            autoencoder = Autoencoder(config['Autoencoder'][f'arch{i + 1}'], config['Autoencoder']['activations'], config['Autoencoder']['batchnorm'])
            self.add_module('autoencoder{}'.format(i), autoencoder)
            dims_view = [self._latent_dim] + self._config['Prediction'][f'arch{i + 1}']
            for j in range(self.view_num):
                if j != i:
                    Prediction_ij = Prediction(dims_view)
                    self.add_module('Prediction_{}_{}'.format(i, j), Prediction_ij)

    def train_completegraph(self, config, logger, accumulated_metrics, X_list, Y_list, mask, optimizer, device):
        """Training the model with complete graph for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              X_list: list data of all view
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
        flag = torch.ones(len(X_list)).to(device)
        flag = (mask == flag.long())
        flag = torch.all(flag == 1, dim=1)
        train_views = [x[flag] for x in X_list]
        for k in range(epochs_total):
            shuffled_views = shuffle(*train_views)
            loss_all, rec, dul, icl = 0, 0, 0, 0
            for batch_view, batch_No in next_batch_multiview(shuffled_views, batch_size):
                # (Todo) Currently, view 0 is the core view as default
                latent_view_z = []
                reconstruction_loss = 0
                # Within-view Reconstruction Loss
                for i in range(self.view_num):
                    autoencoder = getattr(self, f'autoencoder{i}')
                    # get the hidden states for each view
                    latent_view_z.append(autoencoder.encoder(batch_view[i]))
                    reconstruction_loss += F.mse_loss(autoencoder.decoder(latent_view_z[i]), batch_view[i])
                reconstruction_loss /= self.view_num - 2

                # Instance-level Contrastive_Loss
                icl_loss = 0
                for i in range(self.view_num):
                    for j in range(i + 1, self.view_num):
                        weight = 1 if i == 0 and j == 1 else 0.1
                        icl_loss += weight * instance_contrastive_Loss(latent_view_z[i], latent_view_z[j], config['training']['alpha'])
                icl_loss /= self.view_num

                # Cross-view Dual-Prediction Loss
                dualprediction_loss = 0
                for i in range(self.view_num):
                    for j in range(self.view_num):
                        if i != j:
                            prediction = getattr(self, 'Prediction_{}_{}'.format(i, j))
                            dualprediction_loss = (dualprediction_loss + F.mse_loss(prediction(latent_view_z[i])[0], latent_view_z[j]))
                dualprediction_loss = dualprediction_loss / self.view_num

                all_loss = icl_loss + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                loss_all += all_loss.item()
                rec += reconstruction_loss.item()
                dul += dualprediction_loss.item()
                icl += icl_loss.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}" \
                     "===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, rec, dul, icl, loss_all)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.eval()

                    latent_codes_eval = [torch.zeros(X_list[i].shape[0], self._latent_dim).to(device) for i in range(self.view_num)]

                    for i in range(self.view_num):
                        # get the hidden states for existing samples
                        existing_idx_eval = mask[:, i] == 1
                        latent_codes_eval[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(X_list[i][existing_idx_eval])
                    for i in range(self.view_num):
                        missing_idx_eval = mask[:, i] == 0
                        accumulated = (mask[:, i] == 1).float().to(device)
                        if missing_idx_eval.sum() != 0:
                            for j in range(self.view_num):
                                if i != j:
                                    jhas_idx = missing_idx_eval * (mask[:, j] == 1)  # i missing but j has
                                    accumulated += jhas_idx.float()
                                    if jhas_idx.sum() != 0:
                                        jhas_latent = latent_codes_eval[j][jhas_idx]
                                        predicted_latent, _ = getattr(self, f'Prediction_{j}_{i}')(jhas_latent)
                                        latent_codes_eval[i][jhas_idx] += predicted_latent
                        latent_codes_eval[i] = latent_codes_eval[i] / torch.unsqueeze(accumulated, 1)
                    latent_fusion = torch.cat(latent_codes_eval, dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list, accumulated_metrics['acc'], accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'], accumulated_metrics['f-mea'])
                    logger.info("\033[2;29m" + 'trainingset_view_concat ' + str(scores) + "\033[0m")

                    self.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]

    def train_coreview(self, config, logger, accumulated_metrics, X_list, Y_list, mask, optimizer, device):
        """Training the model with cove view for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              X_list: list data of all view
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
        flag = torch.ones(len(X_list)).to(device)
        flag = (mask == flag.long())
        flag = torch.all(flag == 1, dim=1)
        train_views = [x[flag] for x in X_list]

        for k in range(epochs_total):
            shuffled_views = shuffle(*train_views)
            loss_all, rec, dul, icl = 0, 0, 0, 0
            for batch_view, batch_No in next_batch_multiview(shuffled_views, batch_size):
                # (Todo) Currently, view 0 is the core view as default
                latent_view_z = []
                reconstruction_loss = 0
                # Within-view Reconstruction Loss
                for i in range(self.view_num):
                    autoencoder = getattr(self, f'autoencoder{i}')
                    # Get the hidden states for each view
                    latent_view_z.append(autoencoder.encoder(batch_view[i]))
                    reconstruction_loss += F.mse_loss(autoencoder.decoder(latent_view_z[i]), batch_view[i])
                reconstruction_loss /= self.view_num - 2

                # Instance-level Contrastive Loss
                icl_loss = 0
                for j in range(1, self.view_num):
                    weight = 1 if j == 1 else 0.1
                    icl_loss += weight * instance_contrastive_Loss(latent_view_z[0], latent_view_z[j], config['training']['alpha'])
                icl_loss /= self.view_num - 1

                # Cross-view Dual-Prediction Loss
                dualprediction_loss = 0
                for j in range(1, self.view_num):
                    prediction = getattr(self, f'Prediction_0_{j}')
                    dualprediction_loss += F.mse_loss(prediction(latent_view_z[0])[0], latent_view_z[j])
                    prediction = getattr(self, f'Prediction_{j}_0')
                    dualprediction_loss += F.mse_loss(prediction(latent_view_z[j])[0], latent_view_z[0])
                dualprediction_loss /= self.view_num - 1

                all_loss = icl_loss + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                loss_all += all_loss.item()
                rec += reconstruction_loss.item()
                dul += dualprediction_loss.item()
                icl += icl_loss.item()

            output = (f"Epoch : {k + 1}/{epochs_total} ===> Reconstruction loss = {rec:.4f}"
                      f" ===> Dual prediction loss = {dul:.4f} ===> Instance-level contrastive loss = {icl:.4e}"
                      f" ===> Total loss = {loss_all:.4e}")
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.eval()

                    latent_codes_eval = [torch.zeros(X_list[i].shape[0], self._latent_dim).to(device) for i in range(self.view_num)]

                    for i in range(self.view_num):
                        existing_idx_eval = mask[:, i] == 1
                        latent_codes_eval[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(X_list[i][existing_idx_eval])
                    for i in range(self.view_num):
                        if i == 0:  # first recover core view
                            missing_idx_eval = mask[:, i] == 0
                            accumulated = (mask[:, i] == 1).float().to(device)
                            if missing_idx_eval.sum() != 0:
                                for j in range(1, self.view_num):
                                    jhas_idx = missing_idx_eval * (mask[:, j] == 1)  # i missing but j has
                                    accumulated += jhas_idx.float()
                                    if jhas_idx.sum() != 0:
                                        jhas_latent = latent_codes_eval[j][jhas_idx]
                                        predicted_latent, _ = getattr(self, f'Prediction_{j}_0')(jhas_latent)
                                        latent_codes_eval[i][jhas_idx] += predicted_latent
                            latent_codes_eval[i] = latent_codes_eval[i] / torch.unsqueeze(accumulated, 1)
                        else:
                            #  to recover view b, utilizing the core view
                            missing_idx_eval = mask[:, i] == 0
                            if missing_idx_eval.sum() != 0:
                                core_latent = latent_codes_eval[0][missing_idx_eval]  # core view have been recovered when i == 0
                                predicted_latent, _ = getattr(self, f'Prediction_{0}_{i}')(core_latent)
                                latent_codes_eval[i][missing_idx_eval] = predicted_latent

                    latent_fusion = torch.cat(latent_codes_eval, dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list, accumulated_metrics['acc'], accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'], accumulated_metrics['f-mea'])
                    logger.info("\033[2;29m" + 'trainingset_view_concat ' + str(scores) + "\033[0m")

                    self.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]

    def train_completegraph_supervised(self, config, logger, accumulated_metrics, X_list, X_test_list, labels_train, labels_test, mask_train,
                                       mask_test, optimizer, device):
        """Training the model with complete graph for classification

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              X_list: training data
              X_test_list: test data
              labels_train/test: label of training/test data
              mask *: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              classification performance: acc, precision, f-measure

        """
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # select the complete samples
        flag = torch.ones(len(X_list)).to(device)
        flag = (mask_train == flag.long())
        flag = torch.all(flag == 1, dim=1)
        train_views = [x[flag] for x in X_list]
        GT = torch.from_numpy(labels_train).long().to(device)[flag]
        classes = np.unique(np.concatenate([labels_train, labels_test])).size
        flag_gt = False
        if torch.min(GT) == 1:
            flag_gt = True

        for k in range(epochs_total):
            shuffled_data = shuffle(*train_views, GT)
            shuffled_views = shuffled_data[:-1]  # view data
            training_labels = shuffled_data[-1]  # gt data
            loss_all, ccl, rec, dul, icl = 0, 0, 0, 0, 0
            for batch_view, gt_batch, batch_No in next_batch_gt_multiview(shuffled_views, training_labels, batch_size):
                # (Todo) Currently, view 0 is the core view as default
                latent_view_z = []
                reconstruction_loss = 0
                # Within-view Reconstruction Loss
                for i in range(self.view_num):
                    autoencoder = getattr(self, f'autoencoder{i}')
                    # get the hidden states for each view
                    latent_view_z.append(autoencoder.encoder(batch_view[i]))
                    reconstruction_loss += F.mse_loss(autoencoder.decoder(latent_view_z[i]), batch_view[i])
                reconstruction_loss /= self.view_num - 2

                # Instance-level Contrastive_Loss
                icl_loss = 0
                for i in range(self.view_num):
                    for j in range(i + 1, self.view_num):
                        weight = 1 if i == 0 and j == 1 else 0.1
                        icl_loss += weight * instance_contrastive_Loss(latent_view_z[i], latent_view_z[j], config['training']['alpha'])
                icl_loss /= self.view_num

                # Cross-view Dual-Prediction Loss
                dualprediction_loss = 0
                for i in range(self.view_num):
                    for j in range(self.view_num):
                        if i != j:
                            prediction = getattr(self, 'Prediction_{}_{}'.format(i, j))
                            dualprediction_loss = (dualprediction_loss + F.mse_loss(prediction(latent_view_z[i])[0], latent_view_z[j]))
                dualprediction_loss = dualprediction_loss / self.view_num

                # Category-level contrastive loss
                ccl_loss = category_contrastive_loss(torch.cat(latent_view_z, dim=1), gt_batch, classes, flag_gt)

                all_loss = icl_loss + reconstruction_loss * config['training']['lambda2'] + ccl_loss

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                loss_all += all_loss.item()
                rec += reconstruction_loss.item()
                dul += dualprediction_loss.item()
                icl += icl_loss.item()
                ccl += ccl_loss.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}" \
                     "===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> Los_ccl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, rec, dul, icl, ccl, loss_all)

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                # if True:
                with torch.no_grad():
                    self.eval()

                    # obtain representation of training samples
                    latent_codes_train = [torch.zeros(X_list[i].shape[0], self._latent_dim).to(device) for i in range(self.view_num)]
                    for i in range(self.view_num):
                        # get the hidden states for existing samples
                        existing_idx_eval = mask_train[:, i] == 1
                        latent_codes_train[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(X_list[i][existing_idx_eval])
                    for i in range(self.view_num):
                        missing_idx_eval = mask_train[:, i] == 0
                        accumulated = (mask_train[:, i] == 1).float().to(device)
                        if missing_idx_eval.sum() != 0:
                            for j in range(self.view_num):
                                if i != j:
                                    jhas_idx = missing_idx_eval * (mask_train[:, j] == 1)  # i missing but j has
                                    accumulated += jhas_idx.float()
                                    if jhas_idx.sum() != 0:
                                        jhas_latent = latent_codes_train[j][jhas_idx]
                                        predicted_latent, _ = getattr(self, f'Prediction_{j}_{i}')(jhas_latent)
                                        latent_codes_train[i][jhas_idx] += predicted_latent
                        latent_codes_train[i] = latent_codes_train[i] / torch.unsqueeze(accumulated, 1)
                    latent_fusion_train = torch.cat(latent_codes_train, dim=1).cpu().numpy()

                    # obtain representation of test samples
                    latent_codes_test = [torch.zeros(X_test_list[i].shape[0], self._latent_dim).to(device) for i in range(self.view_num)]
                    for i in range(self.view_num):
                        # get the hidden states for existing samples
                        existing_idx_eval = mask_test[:, i] == 1
                        latent_codes_test[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(X_test_list[i][existing_idx_eval])
                    for i in range(self.view_num):
                        missing_idx_eval = mask_test[:, i] == 0
                        accumulated = (mask_test[:, i] == 1).float().to(device)
                        if missing_idx_eval.sum() != 0:
                            for j in range(self.view_num):
                                if i != j:
                                    jhas_idx = missing_idx_eval * (mask_test[:, j] == 1)  # i missing but j has
                                    accumulated += jhas_idx.float()
                                    if jhas_idx.sum() != 0:
                                        jhas_latent = latent_codes_test[j][jhas_idx]
                                        predicted_latent, _ = getattr(self, f'Prediction_{j}_{i}')(jhas_latent)
                                        latent_codes_test[i][jhas_idx] += predicted_latent
                        latent_codes_test[i] = latent_codes_test[i] / torch.unsqueeze(accumulated, 1)
                    latent_fusion_test = torch.cat(latent_codes_test, dim=1).cpu().numpy()

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

                    self.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][-1]

    def train_coreview_supervised(self, config, logger, accumulated_metrics, X_list, X_test_list, labels_train, labels_test, mask_train, mask_test,
                                  optimizer, device):
        """Training the model with cove view for classification

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              X_list: training data
              X_test_list: test data
              labels_train/test: label of training/test data
              mask *: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              classification performance: acc, precision, f-measure


        """
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # select the complete samples
        flag = torch.ones(len(X_list)).to(device)
        flag = (mask_train == flag.long())
        flag = torch.all(flag == 1, dim=1)
        train_views = [x[flag] for x in X_list]
        GT = torch.from_numpy(labels_train).long().to(device)[flag]
        classes = np.unique(np.concatenate([labels_train, labels_test])).size
        flag_gt = False
        if torch.min(GT) == 1:
            flag_gt = True
        for k in range(epochs_total):
            shuffled_data = shuffle(*train_views, GT)
            shuffled_views = shuffled_data[:-1]  # view data
            training_labels = shuffled_data[-1]  # gt data
            loss_all, ccl, rec, dul, icl = 0, 0, 0, 0, 0
            for batch_view, gt_batch, batch_No in next_batch_gt_multiview(shuffled_views, training_labels, batch_size):
                # (Todo) Currently, view 0 is the core view as default
                latent_view_z = []
                reconstruction_loss = 0
                # Within-view Reconstruction Loss
                for i in range(self.view_num):
                    autoencoder = getattr(self, f'autoencoder{i}')
                    # Get the hidden states for each view
                    latent_view_z.append(autoencoder.encoder(batch_view[i]))
                    reconstruction_loss += F.mse_loss(autoencoder.decoder(latent_view_z[i]), batch_view[i])
                reconstruction_loss /= self.view_num - 2

                # Instance-level Contrastive Loss
                icl_loss = 0
                for j in range(1, self.view_num):
                    weight = 1 if j == 1 else 0.1
                    icl_loss += weight * instance_contrastive_Loss(latent_view_z[0], latent_view_z[j], config['training']['alpha'])
                icl_loss /= self.view_num - 1

                # Cross-view Dual-Prediction Loss
                dualprediction_loss = 0
                for j in range(1, self.view_num):
                    prediction = getattr(self, f'Prediction_0_{j}')
                    dualprediction_loss += F.mse_loss(prediction(latent_view_z[0])[0], latent_view_z[j])
                    prediction = getattr(self, f'Prediction_{j}_0')
                    dualprediction_loss += F.mse_loss(prediction(latent_view_z[j])[0], latent_view_z[0])
                dualprediction_loss /= self.view_num - 1

                # Category-level contrastive loss
                ccl_loss = category_contrastive_loss(torch.cat(latent_view_z, dim=1), gt_batch, classes, flag_gt)

                all_loss = icl_loss + reconstruction_loss * config['training']['lambda2'] + ccl_loss

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                loss_all += all_loss.item()
                rec += reconstruction_loss.item()
                dul += dualprediction_loss.item()
                icl += icl_loss.item()
                ccl += ccl_loss.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}" \
                     "===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> Los_ccl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, rec, dul, icl, ccl, loss_all)

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.eval()

                    # obtain representation of training samples
                    latent_codes_train = [torch.zeros(X_list[i].shape[0], self._latent_dim).to(device) for i in range(self.view_num)]
                    for i in range(self.view_num):
                        existing_idx_eval = mask_train[:, i] == 1
                        latent_codes_train[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(X_list[i][existing_idx_eval])
                    for i in range(self.view_num):
                        if i == 0:  # core view
                            missing_idx_eval = mask_train[:, i] == 0
                            accumulated = (mask_train[:, i] == 1).float().to(device)
                            if missing_idx_eval.sum() != 0:
                                for j in range(1, self.view_num):
                                    jhas_idx = missing_idx_eval * (mask_train[:, j] == 1)  # i missing but j has
                                    accumulated += jhas_idx.float()
                                    if jhas_idx.sum() != 0:
                                        jhas_latent = latent_codes_train[j][jhas_idx]
                                        # jhas_latent = getattr(self, f'autoencoder{j}').encoder(X_list[j][jhas_idx])
                                        predicted_latent, _ = getattr(self, f'Prediction_{j}_0')(jhas_latent)
                                        latent_codes_train[i][jhas_idx] += predicted_latent
                            latent_codes_train[i] = latent_codes_train[i] / torch.unsqueeze(accumulated, 1)
                        else:
                            #  to recover view b, utilizing the core view unless core view is missing
                            missing_idx_eval = mask_train[:, i] == 0
                            if missing_idx_eval.sum() != 0:
                                core_latent = latent_codes_train[0][missing_idx_eval]  # core view have been recovered when i == 0
                                predicted_latent, _ = getattr(self, f'Prediction_{0}_{i}')(core_latent)
                                latent_codes_train[i][missing_idx_eval] = predicted_latent
                    latent_fusion_train = torch.cat(latent_codes_train, dim=1).cpu().numpy()

                    # obtain representation of test samples
                    latent_codes_test = [torch.zeros(X_test_list[i].shape[0], self._latent_dim).to(device) for i in range(self.view_num)]
                    for i in range(self.view_num):
                        existing_idx_eval = mask_test[:, i] == 1
                        latent_codes_test[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(X_test_list[i][existing_idx_eval])
                    for i in range(self.view_num):
                        if i == 0:  # first recover core view
                            missing_idx_eval = mask_test[:, i] == 0
                            accumulated = (mask_test[:, i] == 1).float().to(device)
                            if missing_idx_eval.sum() != 0:
                                for j in range(1, self.view_num):
                                    jhas_idx = missing_idx_eval * (mask_test[:, j] == 1)  # i missing but j has
                                    accumulated += jhas_idx.float()
                                    if jhas_idx.sum() != 0:
                                        jhas_latent = latent_codes_test[j][jhas_idx]
                                        # jhas_latent = getattr(self, f'autoencoder{j}').encoder(X_test_list[j][jhas_idx])
                                        predicted_latent, _ = getattr(self, f'Prediction_{j}_0')(jhas_latent)
                                        latent_codes_test[i][jhas_idx] += predicted_latent
                            latent_codes_test[i] = latent_codes_test[i] / torch.unsqueeze(accumulated, 1)
                        else:
                            #  to recover view b, utilizing the core view
                            missing_idx_eval = mask_test[:, i] == 0
                            if missing_idx_eval.sum() != 0:
                                core_latent = latent_codes_test[0][missing_idx_eval]  # core view have been recovered when i == 0
                                predicted_latent, _ = getattr(self, f'Prediction_{0}_{i}')(core_latent)
                                latent_codes_test[i][missing_idx_eval] = predicted_latent

                    latent_fusion_test = torch.cat(latent_codes_test, dim=1).cpu().numpy()

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

                    self.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][-1]
