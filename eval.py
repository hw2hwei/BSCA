from __future__ import print_function


import argparse
import os
import torch
from model.resnet import resnet34, resnet50
from torch.autograd import Variable
from tqdm import tqdm
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.return_dataset import return_dataset_test
from utils.tSNE import tSNE
import torch.nn.functional as F
import numpy as np


# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--step', type=int, default=5000, metavar='step',
                    help='loading step')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME', 'PL', 'DSCA'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         'S+T is training only on labeled examples')
parser.add_argument('--output', type=str, default='./output.txt',
                    help='path to store result file')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network ')
parser.add_argument('--source', type=str, default='Real', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='Product', metavar='B',
                    help='board dir')
parser.add_argument('--dataset', type=str, default='office_home',
                    choices=['multi_all', 'office_home'],
                    help='the name of dataset, multi is large scale dataset')
parser.add_argument('--root', type=str, default='../Datasets',
                    help='source domain')
parser.add_argument('--sample_per_class', type=int, default=3,
                    help='number of labeled examples in the target')

args = parser.parse_args()
print('dataset %s source %s target %s network %s' %
      (args.dataset, args.source, args.target, args.net))
source_loader, target_loader_unl, class_list = return_dataset_test(args)

use_gpu = torch.cuda.is_available()


if args.net == 'resnet34':
    extractor = resnet34()
    inc = 512
elif args.net == 'resnet50':
    extractor = resnet50()
    inc = 2048
elif args.net == "alexnet":
    extractor = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    extractor = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if "resnet" in args.net:
    clasifier = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    clasifier = Predictor(num_class=len(class_list), inc=inc, cosine=True, temp=args.T)
extractor.cuda()
clasifier.cuda()
extractor_checkpath = os.path.join(args.checkpath,
                                   "extractor_{}_{}_{}_{}_"
                                   "to_{}_resume_step.pth.tar".
                                   format(args.dataset, args.net, args.method, 
                                          args.source, args.target))
clasifier_checkpath = os.path.join(args.checkpath,
                                   "clasifier_{}_{}_{}_{}_"
                                   "to_{}_resume_step.pth.tar".
                                   format(args.dataset, args.net, args.method, 
                                          args.source, args.target))
extractor_checkpath = extractor_checkpath \
                      .replace('resume_step', str(args.step))
clasifier_checkpath = clasifier_checkpath \
                      .replace('resume_step', str(args.step))
extractor.load_state_dict(torch.load(extractor_checkpath))
clasifier.load_state_dict(torch.load(clasifier_checkpath))

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
                paths = data[2]
                feat = extractor(im_data)
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
                paths = data[2]
                feat = extractor(im_data)
                output = clasifier(feat)
                size += im_data.size(0)
                pred = output.max(1)[1]
                output = F.softmax(output, dim=1)
                feat_t = torch.cat((feat_t, feat), dim=0)  if 'feat_t' in locals()  else  feat
                prob_t = torch.cat((prob_t, output), dim=0)  if 'prob_t' in locals()  else  output
                labl_t = torch.cat((labl_t, gt_labels), dim=0)  if 'labl_t' in locals()  else  gt_labels
                # for i, path in enumerate(paths):
                #     f.write("%s %d\n" % (path, pred[i]))

    feat_s = feat_s.cpu().detach().numpy()
    feat_t = feat_t.cpu().detach().numpy()
    prob_s = prob_s.cpu().detach().numpy()
    prob_t = prob_t.cpu().detach().numpy()
    labl_s = labl_s.cpu().detach().numpy()
    labl_t = labl_t.cpu().detach().numpy()
    feat = np.concatenate((feat_s, feat_t), axis=0)
    prob = np.concatenate((prob_s, prob_t), axis=0)
    labl = np.concatenate((labl_s, labl_t), axis=0)

    len_s = feat_s.shape[0]
    png_path = args.dataset + '_' + args.source + '_' + args.target + '_'  \
             + args.net + '_' + args.method + '_' + str(args.step) + '_feat.png'
    png_path = os.path.join('tSNE', png_path)
    tSNE(feat, labl, png_path, len_s, n_classes=len(class_list))
    tSNE(prob, labl, png_path.replace('feat', 'prob'), len_s, n_classes=len(class_list))


eval(source_loader, target_loader_unl, output_file="%s_%s_%s.txt" % (args.method, args.net,
                                                                     args.step))



