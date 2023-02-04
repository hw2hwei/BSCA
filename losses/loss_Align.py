import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils.utils import one_hot 


class Align_Loss(nn.Module):
    def __init__(self, n_classes, feat_dim, sample_per_class=3, memory_per_class=64):
        super(Align_Loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.memory_per_class = memory_per_class
        self.sample_per_class = sample_per_class
        self.target_per_class = sample_per_class*2
        # self.target_per_class = 10
        self.cls_loss = nn.CrossEntropyLoss()
        self.source_memory = {}
        self.target_memory = {}
        self.target_memory_ul = {}
        for i in range(self.n_classes):
            self.source_memory[i] = []
            self.target_memory[i] = []
            self.target_memory_ul[i] = []

    def calc_centroids(self, memory):
        centroids = np.zeros((self.n_classes, self.feat_dim))
        for i in range(self.n_classes):
            # print (len(memory[i]))            
            if memory[i] != []:
                memory_i = np.array(memory[i])
                centroids[i] = memory_i.mean(axis=0)

        return torch.tensor(centroids).cuda().detach()

    def return_centroids(self):
        source_memory = self.source_memory
        source_centroids = self.calc_centroids(source_memory)
        target_memory = {}
        for i in range(self.n_classes):
            target_memory[i] = self.target_memory[i] + self.target_memory_ul[i]           
        target_centroids = self.calc_centroids(target_memory)

        return source_centroids, target_centroids

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
            self.update_samples(self.target_memory, feat, labl, self.target_per_class)
        elif data_type == 'target_ul':
            self.update_samples(self.target_memory_ul, F.normalize(feat), labl, self.memory_per_class-self.target_per_class)

    def forward(self, feat, labl, centroids):
        cent_mask = 1 - ((centroids.sum(dim=1)==0) + 0)  # N
        cent_mask = cent_mask.unsqueeze(dim=0).unsqueeze(dim=-1)  # 1 x N x 1
        # feat = F.normalize(feat)
        feat = feat.unsqueeze(dim=1)  # B x 1 x C
        centroids = centroids.unsqueeze(dim=0)  # 1 x N x C
        labl_mask = one_hot(labl, self.n_classes).unsqueeze(dim=-1) # B x N x 1
        cls_mask = labl_mask * cent_mask  # B x N x 1

        alg_loss = ((feat - centroids).pow(2) * cls_mask).mean(dim=2).sum(dim=1).mean()

        return alg_loss
