from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_label, fx_calc_map_multilabel
import numpy as np
from matplotlib import pyplot as plt
from losses import SupConLoss
import scipy.io as sio
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t()) 
    return Sim

def train_model(model, emb, data_loaders, optimizer, configs):
    num_epochs = configs.MAX_EPOCH
    gamma = configs.gamma
    # setup loss
    criterion = SupConLoss(loss_type=configs.loss_type, temperature=configs.tau,  data_class=configs.data_class, gamma=gamma).cuda()
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    clean_indexs = []
    noisy_indexs = []

    for imgs, txts, labels, ori_labels, index in data_loaders['train']:
        clean_index = np.argmax(labels, axis=1) == np.argmax(ori_labels, axis=1)
        clean_indexs.append(index[clean_index])
        noisy_index = np.argmax(labels, axis=1) != np.argmax(ori_labels, axis=1)
        noisy_indexs.append(index[noisy_index])
    clean_indexs = np.concatenate(clean_indexs)
    noisy_indexs = np.concatenate(noisy_indexs)
    MAP_list = []
    clean_indexs = torch.tensor(clean_indexs, requires_grad=False).cuda()
    noisy_indexs = torch.tensor(noisy_indexs, requires_grad=False).cuda()
    clean_weights_list = []
    noisy_weights_list = []
    isclean = None
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        clean_weights = torch.zeros_like(clean_indexs, requires_grad=False).float().cuda()
        noisy_weights = torch.zeros_like(noisy_indexs, requires_grad=False).float().cuda()
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            # Iterate over data.
            for imgs, txts, labels, ori_labels, index in data_loaders[phase]:
                if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                        ori_labels = ori_labels.cuda()


                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward
                    W = emb(torch.eye(configs.data_class).cuda())
                    view1_feature, view2_feature = model(imgs, txts)
                    view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(W.T), dim=1)
                    view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(W.T), dim=1)
                    # if epoch >= configs.tp and configs.self_paced:
                    #     # clean
                    #     isclean = torch.zeros_like(index)
                    #     clean_index = torch.argmax(labels, dim=1) == torch.argmax(ori_labels, dim=1)
                    #     isclean[clean_index] = 1
                    loss, weight = criterion(view1_feature, view2_feature, view1_predict, 
                                     view2_predict, labels, epoch, configs, isclean)
                    if epoch >= configs.tp and  phase == 'train' and configs.self_paced:
                        # clean
                        clean_index = torch.argmax(labels, dim=1) == torch.argmax(ori_labels, dim=1)
                        clean_index1 = index[clean_index].cuda()
                        mask = torch.isin(clean_indexs, clean_index1)
                        clean_index2 = torch.where(mask)[0]
                        tmp = weight[clean_index]
                        clean_weights[clean_index2] = tmp
                        # noisy
                        noisy_index = torch.argmax(labels, dim=1) != torch.argmax(ori_labels, dim=1).int()
                        noisy_index1 = index[noisy_index].cuda()
                        mask = torch.isin(noisy_indexs, noisy_index1)
                        noisy_index2 = torch.where(mask)[0]
                        tmp = weight[noisy_index]
                        noisy_weights[noisy_index2] = tmp
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item()
                clean_index = torch.argmax(labels, dim=1) == torch.argmax(ori_labels, dim=1)
                view1_predict = view1_predict[clean_index]
                view2_predict = view2_predict[clean_index]
                ori_labels = ori_labels[clean_index]
                running_corrects_img += torch.sum(torch.argmax(view1_predict, dim=1) == torch.argmax(ori_labels, dim=1))
                running_corrects_txt += torch.sum(torch.argmax(view2_predict, dim=1) == torch.argmax(ori_labels, dim=1))
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            t_imgs, t_txts, t_labels = [], [], []
            if phase == 'train':
                with torch.no_grad():
                    for imgs, txts, labels, ori_labels, index in data_loaders['valid']:
                        if torch.cuda.is_available():
                                imgs = imgs.cuda()
                                txts = txts.cuda()
                                labels = labels.cuda()
                        t_view1_feature, t_view2_feature = model(imgs, txts)
                        t_imgs.append(t_view1_feature.sign().cpu().numpy())
                        t_txts.append(t_view2_feature.sign().cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                # t_labels = np.concatenate(t_labels).argmax(1)
                t_labels = np.concatenate(t_labels)
                img2txt = fx_calc_map_multilabel(t_imgs, t_txts, t_labels, metric='hamming')
                txt2img = fx_calc_map_multilabel(t_txts, t_imgs, t_labels, metric='hamming')
                MAP_list.append((img2txt + txt2img) / 2.)
                clean_weights_list.append(clean_weights.detach().cpu().numpy())
                noisy_weights_list.append(noisy_weights.detach().cpu().numpy())

            print(phase + '  Loss: %.4f Img2Txt: %.4f  Txt2Img: %.4f  lr: %g'%(epoch_loss, img2txt, txt2img, optimizer.param_groups[0]['lr']))

            # deep copy the model
            if phase == 'valid' and (img2txt + txt2img) / 2. > best_acc:
                best_acc = (img2txt + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                test_img_acc_history.append(img2txt)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # sio.savemat('SPL_weights/' + configs.dataset + '_weights_%.1f_%d_'%(configs.noisy_ratio,configs.tp) +'.mat',{'clean_weights_list':clean_weights_list,
    #                                                                                                             'noisy_weights_list':noisy_weights_list,
    #                                                                                                             'max_epoch':configs.MAX_EPOCH})
    return model, MAP_list