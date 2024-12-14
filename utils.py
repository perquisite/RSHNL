from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_multilabel
import numpy as np
from losses import SupConLoss
import torch.optim as optim
from model import IDCM_NN, Embedding
from sklearn.mixture import GaussianMixture
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
def calc_loss(predict1, predict2, labels, configs):
    q = configs.q
    tmp1 = (1 - q) * (1. - torch.sum(labels.float() * predict1, dim=1) ** q).div(q) + q * (1 - torch.sum(labels.float() * predict1, dim=1))
    tmp2 = (1 - q) * (1. - torch.sum(labels.float() * predict2, dim=1) ** q).div(q) + q * (1 - torch.sum(labels.float() * predict2, dim=1))
    return tmp1, tmp2
def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t()) 
    return Sim
def measure_noise_ratio(data_loaders, input_data_par, configs):
    binary_bits = 128
    model = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], output_dim=binary_bits, num_class=input_data_par['num_class'], tanh=True).cuda()
    emb = Embedding(configs.data_class, binary_bits).cuda()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam([{'params': emb.parameters(), 'lr': configs.lr},
                            {'params': model.parameters(), 'lr': configs.lr}]
                           )
    num_epochs = configs.tp
    max_metric_value = 0
    img_loss_array = np.zeros(input_data_par['num_train'])
    txt_loss_array = np.zeros(input_data_par['num_train'])
    target_labels = []
    for imgs, txts, labels, ori_labels, index in data_loaders['train']:
        labels = labels.cuda()
        target_labels.append(labels.cpu().numpy())
    target_labels = np.concatenate(target_labels)
    labels_num_all = np.sum(target_labels, axis=1).reshape(-1, 1)
    target_labels = torch.tensor(target_labels).cuda().float()
    labels_num_all = torch.tensor(labels_num_all).cuda().float()
    MAP_list = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        model.train()
            # Iterate over data.
        for imgs, txts, labels, ori_labels, index in data_loaders['train']:
            if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
                    print("Data contains Nan.")
                # zero the parameter gradients
            optimizer.zero_grad()
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels.cuda()
                ori_labels = ori_labels.cuda()
            # Forward
            W = emb(torch.eye(configs.data_class).cuda())
            view1_feature, view2_feature = model(imgs, txts)
            view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(W.T), dim=1)
            view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(W.T), dim=1)
            loss_img, loss_txt = calc_loss(view1_predict, view2_predict, labels, configs)
            img_loss_array[index] = loss_img.detach().cpu().numpy()
            txt_loss_array[index] = loss_txt.detach().cpu().numpy()
            loss = loss_img.mean() + loss_txt.mean()
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
        tmp_img, tmp_txt = np.max(img_loss_array), np.max(txt_loss_array)
        img_loss_array, txt_loss_array = img_loss_array / tmp_img, txt_loss_array / tmp_txt
        # X = np.concatenate([img_loss_array, txt_loss_array]).reshape([-1,2])
        X = (img_loss_array + txt_loss_array).reshape(-1,1)
        X = (X-X.min())/(X.max()-X.min()) 
        gmm = GaussianMixture(n_components=2,max_iter=100,tol=1e-2,reg_covar=5e-4)
        gmm.fit(X)
        means = gmm.means_
        cur_metric_value = np.mean(abs(means[0] - means[1]))
        prob = gmm.predict_proba(X)
        prob = prob[:, 1]
        num = np.sum(prob > configs.threshold)
        noise_ratio = num / input_data_par['num_train']

        print('noise_ratio: ', noise_ratio)
        print('cur_metric_value: ', cur_metric_value)

    noise_ratio_measured = noise_ratio
    print('noise_ratio_measured: ', noise_ratio)
    return noise_ratio_measured
        