import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils.utils import one_hot 



def split(feat, feat_bar, pred, thr=0.5):
    prob = F.softmax(pred.detach_(), dim=1)
    prob, labl = prob.max(dim=1)
    # prob = mask*prob
    value, index = torch.sort(prob, dim=0, descending=True) 
    value = value.ge(thr).float()

    feat = feat[index.cpu().numpy()]
    feat_bar = feat_bar[index.cpu().numpy()]
    labl = labl[index.cpu().numpy()]

    n_t = int(value.sum().cpu().numpy())
    if n_t == 0:
        feat_h = None
        feat_h_bar = None
        labl_h = None
    elif n_t == feat.size(0):
        feat_h = feat
        feat_h_bar = feat_bar
        labl_h = labl
    else:
        feat_h = feat[:n_t]
        feat_h_bar = feat_bar[:n_t]
        labl_h = labl[:n_t]

    return feat_h, feat_h_bar, labl_h


def PL_Loss(feat, feat_bar, pred, pred_bar, centroid, thr):
    
    # split feat_u
    feat_h, feat_h_bar, labl_h = split(feat, feat_bar, pred, thr)

    # loss for pseudo labeling
    max_probs, pseudo_labels = torch.max(torch.softmax(pred.detach_(), dim=-1), dim=-1)
    pseudo_mask = max_probs.ge(thr).float()

    feat_simi = feat.clone().detach().unsqueeze(dim=1)  # B x 1 x C
    cent_simi = centroid.clone().detach().unsqueeze(dim=0)  # 1 x K x C
    simi_map = F.cosine_similarity(feat_simi, cent_simi, dim=2)  # B x K    
    simi_probs, simi_labels = torch.max(simi_map, dim=-1)  
    simi_mask = (simi_labels==pseudo_labels).float() 

    mask = pseudo_mask*simi_mask

    loss_pl_bar = (F.cross_entropy(pred_bar, pseudo_labels, reduction='none') * mask).mean()

    # criterion_reduce = nn.CrossEntropyLoss(reduce=False).cuda()
    # prob_u = pred.data.max(1)[1].detach()
    # ent = - torch.sum(F.softmax(pred, 1) * (torch.log(F.softmax(pred, 1) + 1e-5)), 1)
    # mask_reliable = (ent < 0.5).float().detach()    
    # loss_pl_bar = (mask_reliable * criterion_reduce(pred, prob_u)).sum(0) / (1e-5 + mask_reliable.sum())

    return loss_pl_bar, feat_h, feat_h_bar, labl_h



