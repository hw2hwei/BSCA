from __future__ import print_function


import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from models.resnet import resnet34
from models.basenet import Efficientnet_B0_Base, Efficientnet_B4_Base, GoogLeNet_Base, Predictor, Predictor_deep, Predictor_DANN, Predictor_APE, Predictor_deep_APE, Predictor_deep_DANN
from utils.return_dataset import return_dataset_test
from utils.tSNE import tSNE
import torch.nn.functional as F
import numpy as np


# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--checkpath', type=str, default='./save_model',
                    help='dir to save checkpoint')
parser.add_argument('--method', type=str, default='BSCA',
                    choices=['S', 'S+T', 'ENT', 'MME', 'PL', 'PL+Alg', 'FixMatch', 'DANN', 'APE', 'BSCA', 'CDAC'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         'S+T is training only on labeled examples')
parser.add_argument('--output', type=str, default='./output.txt',
                    help='path to store result file')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network ')
parser.add_argument('--root', type=str, default='/home/Datasets/RSSC',
                    help='source domain')
parser.add_argument('--source', type=str, default='NWPU-RESISC45', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='AID', metavar='B',
                    help='board dir')
parser.add_argument('--sample_per_class', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--thr', type=float, default=0.5,
                    help='number of labeled examples in the target')


args = parser.parse_args()
print('source %s target %s network %s' %
      (args.source, args.target, args.net))
source_loader, target_loader_unl, class_list = return_dataset_test(args)

use_gpu = torch.cuda.is_available()

if args.net == 'resnet34':
    extractor = resnet34()
    inc = 512
elif args.net == "efficientnet":
    extractor = Efficientnet_Base()
    inc = 1280
elif args.net == "googlenet":
    extractor = GoogLeNet_Base()
    inc = 1024
else:
    raise ValueError('Model cannot be recognized.')

if 'resnet' in args.net:
    clasifier = Predictor_deep(num_class=len(class_list), inc=inc)    
    if args.method == 'DANN':
        clasifier = Predictor_deep_DANN(num_class=len(class_list), inc=inc,
                                    temp=args.T)
    elif args.method == 'APE':
        clasifier = Predictor_deep_APE(num_class=len(class_list), inc=inc,
                                    temp=args.T)
else:
    clasifier = Predictor(num_class=len(class_list), inc=inc)    
    if args.method == 'DANN':
        clasifier = Predictor_DANN(num_class=len(class_list), inc=inc,
                                    temp=args.T)
    elif args.method == 'APE':
        clasifier = Predictor_APE(num_class=len(class_list), inc=inc,
                                    temp=args.T)

extractor.cuda()
clasifier.cuda()
extractor_pth = os.path.join(args.checkpath,
                             "extractor_{}_{}_{}_"
                             "to_{}_{}_thr_{}_best.pth.tar".
                             format(args.net, args.method, 
                                    args.source, args.target,
                                    args.sample_per_class,
                                    args.thr))
clasifier_pth = os.path.join(args.checkpath,
                             "clasifier_{}_{}_{}_"
                             "to_{}_{}_thr_{}_best.pth.tar".
                             format(args.net, args.method, 
                                    args.source, args.target,
                                    args.sample_per_class,
                                    args.thr))
extractor.load_state_dict(torch.load(extractor_pth))
clasifier.load_state_dict(torch.load(clasifier_pth))

im_data = torch.FloatTensor(1)
gt_labels = torch.LongTensor(1)

im_data = im_data.cuda()
gt_labels = gt_labels.cuda()

im_data = Variable(im_data)
gt_labels = Variable(gt_labels)
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)



def eval(source_loader, target_loader, output_file="output.txt"):
    extractor.eval()
    clasifier.eval()
    size = 0
    with open(output_file, "w") as f:
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(source_loader)):
                im_data.resize_(data[0].size()).copy_(data[0])
                gt_labels.resize_(data[1].size()).copy_(data[1])
                feat = extractor(im_data)
                if args.method!='S+T' and args.method!='S' and args.method!='ENT' and args.method!='MME' and args.method!='CDAC':
                    _, output = clasifier(feat)                
                else:
                    output = clasifier(feat)                
                size += im_data.size(0)
                pred = output.max(1)[1]
                output = F.softmax(output, dim=1)
                feat_s = torch.cat((feat_s, feat), dim=0)  if 'feat_s' in locals()  else  feat
                prob_s = torch.cat((prob_s, output), dim=0)  if 'prob_s' in locals()  else  output
                labl_s = torch.cat((labl_s, gt_labels), dim=0)  if 'labl_s' in locals()  else  gt_labels
                # for i, path in enumerate(paths):
                #     f.write("%s %d\n" % (path, pred[i]))       
            for batch_idx, data in tqdm(enumerate(target_loader)):
                im_data.resize_(data[0].size()).copy_(data[0])
                gt_labels.resize_(data[1].size()).copy_(data[1])
                feat = extractor(im_data)
                if args.method!='S+T' and args.method!='S' and args.method!='ENT' and args.method!='MME' and args.method!='CDAC':
                    _, output = clasifier(feat)                
                else:
                    output = clasifier(feat)                
                size += im_data.size(0)
                pred = output.max(1)[1]
                output = F.softmax(output, dim=1)
                feat_t = torch.cat((feat_t, feat), dim=0)  if 'feat_t' in locals()  else  feat
                labl_t = torch.cat((labl_t, gt_labels), dim=0)  if 'labl_t' in locals()  else  gt_labels
                # for i, path in enumerate(paths):
                #     f.write("%s %d\n" % (path, pred[i]))

    feat_s, feat_t = feat_s.cpu().detach().numpy(), feat_t.cpu().detach().numpy()
    labl_s, labl_t = labl_s.cpu().detach().numpy(), labl_t.cpu().detach().numpy()
    feat = np.concatenate((feat_s, feat_t), axis=0)
    labl = np.concatenate((labl_s, labl_t), axis=0)

    len_s = feat_s.shape[0]
    png_path = args.source + '_' + args.target + '_'  \
             + args.net + '_' + args.method + '_feat.png'
    png_path = os.path.join('tSNE', png_path)
    tSNE(feat, labl, png_path, len_s, n_classes=len(class_list))


# eval(source_loader, target_loader_unl, output_file="%s_%s.txt" % (args.method, args.net))






from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(adap_scenario, true_list, pred_list, label2cls_list):
    labels = []
    for key, value in label2cls_list.items():
        labels.append(value)
    tick_marks = np.float32(np.array(range(len(labels)))) + 0.5

    cm = confusion_matrix(true_list, pred_list)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    
    fontsize_axis = 16
    fontsize_prop = 10
    barsize = 14
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_norm[y_val][x_val]
        if c > 0.01:
            color="white" if c > 0.5 else "black"
            plt.text(x_val, y_val, '%0.2f'%(c,), color=color, fontsize=fontsize_prop, va='center', ha='center')

    plt.gca().set_xticks(tick_marks)
    plt.gca().set_yticks(tick_marks)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues) 
    xlocations = np.array(range(len(labels))) 
    plt.xticks(xlocations, labels, fontsize=fontsize_axis, rotation=45) 
    plt.yticks(xlocations, labels, fontsize=fontsize_axis) 

    cb = plt.colorbar() 
    cb.ax.tick_params(labelsize=barsize)

    plt.tight_layout()

    plt.savefig('./tSNE/' + args.method + '_confusion_matrix_' + adap_scenario + '.pdf', format='pdf')
    # plt.show()

gt_list = []
pred_list = []
extractor.eval()
clasifier.eval()
for batch_idx, data in tqdm(enumerate(target_loader_unl)):
    im_data.resize_(data[0].size()).copy_(data[0])
    gt_labels.resize_(data[1].size()).copy_(data[1])
    feat = extractor(im_data)
    output = clasifier(feat)                
    pred = output[1].max(1)[1]
    gt_list += list(gt_labels.cpu().detach().numpy())
    pred_list += list(pred.cpu().detach().numpy())

label2cls_list = {  \
     2: "industrial",
     3: "meadow",
     5: "residential", 
     0: "farmland", 
     6: "forest",
     1: "parking",
     4: "river",
}

adap_scenario = 'NWPU-to-AID'
plot_confusion_matrix(adap_scenario, gt_list, pred_list, label2cls_list)

