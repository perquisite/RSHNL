import torch
import torch.nn as nn
# def rank_loss(features1, features2, margin):
#     cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) * 2.
#     sim = cos(features1, features2)
#     diag = torch.diag(sim)
#     sim = sim - diag.view(-1,1)
#     sim[sim + margin < 0] = 0
#     return sim.mean()
def cross_modal_contrastive_ctriterion_q(fea, tau=1., q = 1, opt = None):
        # q = opt.q
        n_view = 2
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())
        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p1 = diag1 / sim.sum(1)
        loss1 = (1 - q) * (1. - (p1) ** q).div(q) + q * (1 - p1)

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p2 = diag2 / sim.sum(1)
        loss2 = (1 - q) * (1. - (p2) ** q).div(q) + q * (1 - p2)
        return loss1.mean() + loss2.mean()
class SupConLoss(nn.Module):
    def __init__(self, loss_type='ours', temperature=0.3, data_class=10, gamma = 3):
        super(SupConLoss, self).__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.data_class = data_class
        self.gamma = gamma

    def forward(self, features1, features2, predict1, predict2, labels=None, epoch=0, opt=None, isclean=None):
        gamma = self.gamma
        alpha = opt.alpha
        lamda = opt.lamda
        if opt.loss_type == 'RSHNL':
            q = opt.q
            tmp1 = (1 - q) * (1. - torch.sum(labels.float() * predict1, dim=1) ** q).div(q) + q * (1 - torch.sum(labels.float() * predict1, dim=1))
            tmp2 = (1 - q) * (1. - torch.sum(labels.float() * predict2, dim=1) ** q).div(q) + q * (1 - torch.sum(labels.float() * predict2, dim=1))
        elif opt.loss_type == 'CE':
            tmp1 = - (labels * predict1.log()).sum(1)
            tmp2 = - (labels * predict2.log()).sum(1)
        elif opt.loss_type == 'GCE':
            q = min(1., 0.01 * (epoch + 1))
            tmp1 = (1 - q) * (1. - torch.sum(labels.float() * predict1, dim=1) ** q).div(q)
            tmp2 = (1 - q) * (1. - torch.sum(labels.float() * predict2, dim=1) ** q).div(q)
        elif opt.loss_type == 'RC':
            tmp1 = torch.log(1-(labels * predict1)).sum(1)
            tmp2 = torch.log(1-(labels * predict2)).sum(1)
        term1 = tmp1 + tmp2
        weight = 0
        if opt.self_paced and epoch >= opt.tp:
            alpha = opt.alpha
            lamda = opt.lamda
            if opt.linear:
                weight = (1 - 1/gamma * term1).detach()
                # weight = (1 - 1/gamma * term1)
                weight[weight<0] = 0
                # weight[weight>0] = 1
                regularization_term = gamma * (1/2 * (weight**2) - weight)
            else:
                weight = (term1 < gamma).float().detach()
                regularization_term = -gamma * weight
            # print(weight.max(), term1.min())
            # weight = weight * isclean.cuda()
            term1 = weight * term1 + regularization_term # 原设置
            # term1 = weight * term1
        else:
            pass
        term3 = cross_modal_contrastive_ctriterion_q([features1, features2], tau=opt.tau, q=q, opt=opt)
        return lamda * term1.mean() + alpha * term3, weight
        