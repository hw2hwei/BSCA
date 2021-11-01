import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def one_hot(labl, n_classes):
    labl = labl.view(-1, 1)
    labl_ = torch.zeros(labl.size(0), n_classes)
    labl_ = labl_.scatter_(1, labl.type(torch.LongTensor), 1).cuda().detach()

    return labl_

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd

        return output, None


def grad_reverse(x, lambd=1.0):
    return ReverseLayerF.apply(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(clasifier, feat, lamda, eta=1.0):
    pred = clasifier(feat, reverse=True, eta=-eta)
    pred = F.softmax(pred)
    loss_ent = -lamda * torch.mean(torch.sum(pred *
                                             (torch.log(pred + 1e-5)), 1))
    return loss_ent


def adentropy(clasifier, feat, lamda, eta=1.0):
    pred = clasifier(feat, reverse=True, eta=eta)
    pred = F.softmax(pred)
    loss_adent = lamda * torch.mean(torch.sum(pred *
                                              (torch.log(pred + 1e-5)), 1))
    return loss_adent


def split(feat, pred, thr=0.5):
    prob = F.softmax(pred.detach(), dim=1)
    prob, labl = prob.max(dim=1)
    value, index = torch.sort(prob, dim=0, descending=True) 
    value = value.ge(thr).float()
    feat = feat[index.cpu().numpy()]
    pred = pred[index.cpu().numpy()]
    labl = labl[index.cpu().numpy()]
    feat_h = None
    pred_h = None
    labl_h = None
    feat_l = None
    pred_l = None
    
    n_t = int(value.sum().cpu().numpy())
    if n_t == 0:
        feat_l = feat
        pred_l = pred
    elif n_t == feat.size(0):
        feat_h = feat
        pred_h = pred
        labl_h = labl
    else:
        feat_h = feat[:n_t]
        pred_h = pred[:n_t]
        labl_h = labl[:n_t]
        feat_l = feat[n_t:]
        pred_l = pred[n_t:]

    return feat_h, pred_h, labl_h


def pl(extractor, clasifier, imag, imag_bar, thr):
    feat = extractor(imag)
    pred = clasifier(feat)
    feat_bar = extractor(imag_bar)
    pred_bar = clasifier(feat_bar)

    # feat_h, pred_h, labl_h = split(feat, pred, thr)
    # loss_pl = 0 
    # if pred_h != None:
    #     loss_pl += criterion(pred_h, labl_h)

    feat_h, pred_h, labl_h = split(feat, pred, thr)
    feat_h_bar, _, _ = split(feat_bar, pred, thr)
    max_probs, pseudo_labels = torch.max(torch.softmax(pred.detach_(), dim=-1), dim=-1)
    mask = max_probs.ge(thr).float()
    
    # loss for pseudo labeling
    loss_pl = (F.cross_entropy(pred_bar, pseudo_labels, reduction='none') * mask).mean()

    return loss_pl, feat_h, labl_h, feat_h_bar


class Align_Loss(nn.Module):
    def __init__(self, n_classes, feat_dim, sample_per_class=3, memory_per_class=24):
        super(Align_Loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.memory_per_class = memory_per_class
        self.sample_per_class = sample_per_class
        self.cls_loss = nn.CrossEntropyLoss()
        self.source_memory = {}
        self.target_memory = {}
        self.target_memory_ul = {}
        for i in range(self.n_classes):
            self.source_memory[i] = []
            self.target_memory[i] = []
            self.target_memory_ul[i] = []

    def calc_centriods(self, memory):
        centriods = np.zeros((self.n_classes, self.feat_dim))
        for i in range(self.n_classes):
            # print (len(memory[i]))            
            if memory[i] != []:
                memory_i = np.array(memory[i])
                centriods[i] = memory_i.mean(axis=0)

        return torch.tensor(centriods).cuda().detach()

    def return_centroids(self):
        source_memory = self.source_memory
        source_centriods = self.calc_centriods(source_memory)
        target_memory = {}
        for i in range(self.n_classes):
            target_memory[i] = self.target_memory[i] + self.target_memory_ul[i]           
        target_centriods = self.calc_centriods(target_memory)

        return source_centriods, target_centriods

    def update_samples(self, memory, feat, labl, memory_len):
        batch_size = feat.size(0)
        for i in range(batch_size):
            labl_i = int(labl[i].cpu().numpy())
            # print (len(memory[labl_i]))
            if len(memory[labl_i]) < memory_len:
                memory[labl_i].append(feat[i].clone().detach().cpu().numpy())
            else:
                del memory[labl_i][0]             
                memory[labl_i].append(feat[i].clone().detach().cpu().numpy())

    def update_memory(self, feat, labl, data_type='source'):
        if data_type == 'source':
            self.update_samples(self.source_memory, feat, labl, self.memory_per_class)
        elif data_type == 'target':
            self.update_samples(self.target_memory, feat, labl, self.sample_per_class*2)
        elif data_type == 'target_ul':
            self.update_samples(self.target_memory_ul, feat, labl, self.memory_per_class-self.sample_per_class*2)

    def forward(self, feat, labl, centriods, dist_type='MSE'):
        cent_mask = 1 - ((centriods.sum(dim=1)==0) + 0)  # N
        cent_mask = cent_mask.unsqueeze(dim=0).unsqueeze(dim=-1)  # 1 x N x 1

        feat = feat.unsqueeze(dim=1)  # B x 1 x C
        centriods = centriods.unsqueeze(dim=0)  # 1 x N x C
        labl_mask = one_hot(labl, self.n_classes).unsqueeze(dim=-1) # B x N x 1
        cls_mask = labl_mask * cent_mask  # B x N x 1

        if dist_type == 'MSE':
            alg_loss = ((feat - centriods).pow(2) * cls_mask).mean(dim=2).sum(dim=1).mean()
        elif dist_type == 'Cosine': 
            dist = 1 - F.cosine_similarity(feat, centriods, dim=-1)
            cls_mask = cls_mask.squeeze(dim=-1)
            alg_loss = (dist * cls_mask).sum(dim=1).mean()

        return alg_loss
