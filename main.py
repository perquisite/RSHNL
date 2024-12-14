import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from model import IDCM_NN, Embedding
from train_model import train_model
from load_data import get_loader
from evaluate import  fx_calc_map_multilabel
import scipy.io as sio
from to_seed import to_seed
import os
import numpy as np
from utils import measure_noise_ratio
######################################################################
# Start running
import argparse
# Training settings
parser = argparse.ArgumentParser(description='dorefa-net implementation')
def str2bool(str):
    return True if str.lower() == 'true' else False
#########################
#### data parameters ####
#########################
parser.add_argument("--dataset", type=str, default="wiki") # wiki xmedia INRIA-Websearch xmedianet 
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--alpha", type=float, default=0.05) #MC 0.05 0.10 0.30 0.05
parser.add_argument("--gamma", type=float, default=4) # 3 5 8 9
parser.add_argument("--lamda", type=float, default=1) # LS
parser.add_argument("--MAX_EPOCH", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--bit", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4) #learning rate
parser.add_argument("--noisy_ratio", type=float, default=0.2) #0.2 0.4 0.6 0.8
parser.add_argument("--noise_mode", type=str, default='sym') #sym asym
parser.add_argument("--tp", type=int, default=5) #turning point
parser.add_argument("--q", type=float, default=0.01)
parser.add_argument("--loss_type", type=str, default='RSHNL') #
parser.add_argument("--tau", type=float, default=1.0) #temperature
parser.add_argument('--self_paced',  type=str2bool, default='True')#'self_paced learning schedule'
parser.add_argument('--linear',  type=str2bool, default='True')#'self_paced learning schedule'
parser.add_argument("--GPU", type=int, default=0)

args = parser.parse_args()
print(args)


if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = args.dataset # IAPR MIRFlickr nuswide mscoco
    device = torch.device("cuda:%d"%args.GPU if torch.cuda.is_available() else "cpu")
    seed = args.seed
    to_seed(seed)
    # data parameters
    batch_size = args.batch_size
    bit = args.bit
    lr = args.lr
    weight_decay = 0
    noisy_ratio = args.noisy_ratio 
    noise_mode = args.noise_mode # sym asym
    print('...Data loading is beginning...')
    print('The noise_radio is: ', noisy_ratio)

    data_loader, input_data_par = get_loader(dataset, batch_size, noisy_ratio, noise_mode)

    print('...Data loading is completed...')
    args.data_class = input_data_par['num_class']
    warm_up_epoch = args.tp
    
    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],output_dim=bit, num_class=input_data_par['num_class']).to(device)
    params_to_update = list(model_ft.parameters())
    emb = Embedding(args.data_class, args.bit).cuda()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam([{'params': emb.parameters(), 'lr': args.lr},
                            {'params': model_ft.parameters(), 'lr': args.lr}]
                           )
    print('...Training is beginning...')
    # Train and evaluate
    model_ft, MAP_list = train_model(model_ft, emb, data_loader, optimizer, args)
    print('...Training is completed...')

    print('...Evaluation on testing data...')
    view1_feature, view2_feature = model_ft(torch.tensor(input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))
    label = input_data_par['label_test']
    view1_feature = view1_feature.sign().detach().cpu().numpy()
    view2_feature = view2_feature.sign().detach().cpu().numpy()
    sio.savemat('Avg_MAP_data_noSPL/' + dataset + '_Avg_MAP_%.1f_%d_'%(noisy_ratio,warm_up_epoch) +'.mat',{'map_list':MAP_list,'max_epoch':args.MAX_EPOCH})
    # Avg MAP曲线
    # plt.figure(figsize=(10,10))
    # plt.clf()
    # MAP_list.insert(0,0)
    # plt.plot(range(args.MAX_EPOCH+1), MAP_list, color = 'red')
    # plt.savefig('Avg_MAP_figure/' + dataset + '_Avg_MAP_%.1f_%d_'%(noisy_ratio,warm_up_epoch) + noise_mode + '.jpg')

    img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric='hamming')
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric='hamming')
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))

    view1_feature, view2_feature = input_data_par['img_train'], input_data_par['text_train']
    label = input_data_par['label_train']
    sio.savemat('features/' + dataset + '_' + str(0) + '_train_ori.mat', {'train_fea':view1_feature,
                                                                'train_lab':label})
    sio.savemat('features/' + dataset + '_' + str(1) + '_train_ori.mat', {'train_fea':view2_feature,
                                                                'train_lab':label})

    view1_feature, view2_feature = model_ft(torch.tensor(input_data_par['img_train']).to(device), torch.tensor(input_data_par['text_train']).to(device))
    label = input_data_par['label_train']
    view1_feature = view1_feature.sign().detach().cpu().numpy()
    view2_feature = view2_feature.sign().detach().cpu().numpy()
    sio.savemat('features/' + dataset + '_' + str(0) + '_train.mat', {'train_fea':view1_feature,
                                                                'train_lab':label})
    sio.savemat('features/' + dataset + '_' + str(1) + '_train.mat', {'train_fea':view2_feature,
                                                                'train_lab':label})
