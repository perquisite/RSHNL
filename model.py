import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import numpy as np
class Embedding(torch.nn.Module):
        def __init__(self, data_class, binary_bits):
            super(Embedding, self).__init__()
            mid_num1 = 4096
            mid_num2 = 4096
            self.fc1 = nn.Linear(data_class, mid_num1)
            self.fc2 = nn.Linear(mid_num1, mid_num2)
            self.Embedding = nn.Linear(mid_num2, binary_bits)
            nn.init.uniform_( self.Embedding.weight, -1. / np.sqrt(np.float32(data_class)), 1. / np.sqrt(np.float32(data_class)) )

        def forward(self, x):
            out1 = F.relu(self.fc1(x))
            out2 = F.relu(self.fc2(out1))

            out3 = self.Embedding(out2).tanh()
            norm = torch.norm(out3, p=2, dim=1, keepdim=True)
            out3 = out3 / norm
            return  out3

class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=28*28, output_dim=20, tanh = True):
        super(ImgNN, self).__init__()
        mid_num1 = 4096
        mid_num2 = 4096
        self.tanh = tanh
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)

        self.fc3 = nn.Linear(mid_num2, output_dim, bias=False)
        nn.init.uniform_( self.fc3.weight, -1. / np.sqrt(np.float32(input_dim)), 1. / np.sqrt(np.float32(input_dim)) )


    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))

        out3 = self.fc3(out2).tanh() if self.tanh else self.fc3(out2)
        norm = torch.norm(out3, p=2, dim=1, keepdim=True)
        out3 = out3 / norm

        return  out3


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=28*28, output_dim=20, tanh = True):
        super(TextNN, self).__init__()
        mid_num1 = 4096
        mid_num2 = 4096
        self.tanh = tanh
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)

        self.fc3 = nn.Linear(mid_num2, output_dim, bias=False)
        nn.init.uniform_( self.fc3.weight, -1. / np.sqrt(np.float32(input_dim)), 1. / np.sqrt(np.float32(input_dim)) )


    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2).tanh() if self.tanh else self.fc3(out2)
        norm = torch.norm(out3, p=2, dim=1, keepdim=True)
        out3 = out3 / norm

        return  out3


class IDCM_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=4096, text_input_dim=1024, output_dim =1024, num_class=10, tanh = True):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, output_dim = output_dim, tanh = tanh)
        self.text_net = TextNN(text_input_dim, output_dim = output_dim, tanh = tanh)
        # W = torch.Tensor(output_dim, output_dim)
        # self.W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: num_class]
        # self.W = self.W.clone().detach().cuda()
        # self.W.requires_grad=False

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)

        # view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(self.W), dim=1)
        # view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(self.W), dim=1)

        return view1_feature, view2_feature
    def reset_parameters(self):
        for layer in self.img_net.children():
            layer.reset_parameters()
        for layer in self.text_net.children():
            layer.reset_parameters()

