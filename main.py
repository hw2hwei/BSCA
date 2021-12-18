import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.resnet import resnet34
from models.basenet import Efficientnet_Base, GoogLeNet_Base, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from losses.loss_Align import Align_Loss
from losses.loss_PL import PL_Loss

# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--resume_step', type=int, default=0, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='BSCA',
                    choices=['BSCA'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--thr', type=float, default=0.95, metavar='LR',
                    help='pseudo label thr (default: 0.8)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.5, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--pth', type=str, default='./save_model',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--n_workers', type=int, default=0,
                    help='number of workers of python')
parser.add_argument('--root', type=str, default='../Datasets',
                    help='source domain')
parser.add_argument('--source', type=str, default='Real',
                    help='source domain')
parser.add_argument('--target', type=str, default='Product',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='office_home',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--sample_per_class', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--memory_per_class', type=int, default=96,
                    help='number of labeled examples in the target')

args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.sample_per_class, args.net))
source_loader, target_loader, target_loader_u, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.sample_per_class))

torch.cuda.manual_seed(args.seed)
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

params = []
for key, value in dict(extractor.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if 'resnet' in args.net:
    clasifier = Predictor_deep(num_class=len(class_list), inc=inc)    
else:
    clasifier = Predictor(num_class=len(class_list), inc=inc)    

weights_init(clasifier)
extractor_pth = os.path.join(args.pth,
                                   "extractor_{}_{}_{}_{}_"
                                   "to_{}_resume_step.pth.tar".
                                   format(args.dataset, args.net, args.method, 
                                          args.source, args.target))
clasifier_pth = os.path.join(args.pth,
                                   "clasifier_{}_{}_{}_{}_"
                                   "to_{}_resume_step.pth.tar".
                                   format(args.dataset, args.net, args.method, 
                                          args.source, args.target))
extractor_pth1 = extractor_pth \
                        .replace('resume_step', str(args.resume_step))
clasifier_pth1 = clasifier_pth \
                        .replace('resume_step', str(args.resume_step))
if os.path.exists(extractor_pth1):
    print ('load ' + extractor_pth1)
    extractor.load_state_dict(torch.load(extractor_pth1))
if os.path.exists(clasifier_pth1):
    print ('load ' + clasifier_pth1)
    clasifier.load_state_dict(torch.load(clasifier_pth1))
extractor = extractor.cuda()
clasifier = clasifier.cuda()
extractor = torch.nn.DataParallel(extractor)
clasifier = torch.nn.DataParallel(clasifier)

imag_s = torch.FloatTensor(1).cuda()
imag_t = torch.FloatTensor(1).cuda()
imag_u = torch.FloatTensor(1).cuda()
imag_s_bar = torch.FloatTensor(1).cuda()
imag_t_bar = torch.FloatTensor(1).cuda()
imag_u_bar = torch.FloatTensor(1).cuda()
labl_s = torch.LongTensor(1).cuda()
labl_t = torch.LongTensor(1).cuda()
sample_labl_t = torch.LongTensor(1).cuda()
sample_labl_s = torch.LongTensor(1).cuda()

if os.path.exists(args.pth) == False:
    os.mkdir(args.pth)


def train():
    extractor.train()
    clasifier.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(clasifier.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    cls_criterion = nn.CrossEntropyLoss().cuda()
    alg_criterion = Align_Loss(n_classes=len(class_list), \
                               feat_dim=inc,   \
                               sample_per_class=args.sample_per_class,  \
                               memory_per_class=args.memory_per_class).cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_u = iter(target_loader_u)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_u)
    best_acc_val = 0

    for step in range(args.resume_step, all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_u = iter(target_loader_u)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_u = next(data_iter_u)
        data_s = next(data_iter_s)
        imag_s.resize_(data_s[0].size()).copy_(data_s[0])
        labl_s.resize_(data_s[1].size()).copy_(data_s[1])
        imag_t.resize_(data_t[0].size()).copy_(data_t[0])
        labl_t.resize_(data_t[1].size()).copy_(data_t[1])
        imag_u.resize_(data_u[0].size()).copy_(data_u[0])
        # imag_s_bar.resize_(data_s[2].size()).copy_(data_s[2])
        # imag_t_bar.resize_(data_t[2].size()).copy_(data_t[2])
        imag_u_bar.resize_(data_u[2].size()).copy_(data_u[2])
        imag_st = torch.cat((imag_s, imag_t), dim=0)
        labl_st = torch.cat((labl_s, labl_t), dim=0)
        # imag_st_bar = torch.cat((imag_s_bar, imag_t_bar), dim=0)
        ns, nt = imag_s.size(0), imag_t.size(0)

        # feature extractor
        zero_grad_all()
        feat_st = extractor(imag_st)
        pred_st = clasifier(feat_st)
        feat_s = feat_st[:ns]
        feat_t = feat_st[ns:]
        # feat_st_bar = extractor(imag_st_bar)
        # pred_st_bar = clasifier(feat_st_bar)
        # feat_s_bar = feat_st_bar[:ns]
        # feat_t_bar = feat_st_bar[ns:]
        # pred_s_bar = pred_st_bar[:ns]
        # pred_t_bar = pred_st_bar[ns:]

        # CE loss
        # loss = (cls_criterion(pred_st, labl_st) + cls_criterion(pred_st_bar, labl_st)) / 2
        loss = cls_criterion(pred_st, labl_st)

        # PL loss
        source_centroids, target_centroids = alg_criterion.return_centroids()
        loss_pl, feat_u, feat_u_bar, labl_u = PL_Loss(extractor, clasifier, imag_u, imag_u_bar, \
                                                      target_centroids, args.thr) 
        loss += loss_pl
        
        # Align loss
        # loss_alg = (alg_criterion(feat_s, labl_s, target_centroids) +   \
        #             alg_criterion(feat_s_bar, labl_s, target_centroids)) / 2
        loss_alg = alg_criterion(feat_s, labl_s, target_centroids)
        if feat_u != None:
            feat_tu = torch.cat((feat_t, feat_u_bar), dim=0)
            # feat_tu_bar = torch.cat((feat_t_bar, feat_u_bar), dim=0)
            labl_tu = torch.cat((labl_t, labl_u), dim=0)
        else:
            feat_tu = feat_t
            # feat_tu_bar = feat_t_bar
            labl_tu = labl_t
        # loss_alg += (alg_criterion(feat_tu, labl_tu, source_centroids) +   \
        #              alg_criterion(feat_tu_bar, labl_tu, source_centroids)) / 2 
        loss_alg += alg_criterion(feat_tu, labl_tu, source_centroids) 
        loss_alg *= (0.5*args.lamda)
        loss += loss_alg

        # update the memory
        with torch.no_grad():
            alg_criterion.update_memory(feat_s, labl_s, data_type='source')
            alg_criterion.update_memory(feat_t, labl_t, data_type='target')
            if feat_u != None:
                alg_criterion.update_memory(feat_u, labl_u, data_type='target_ul')

        # loss backpropogation
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()

        log_train = 'S: {}   T: {}   Train Ep: {}   lr: {:.6f} ' \
                    '  Loss CE: {:.6f}   Loss PL: {:.6f}   Loss Alignment: {:.6f} '\
                    '  Method: {}'.format(args.source, args.target,
                                          step, lr, (loss-loss_alg-loss_pl), loss_pl,  
                                          loss_alg, args.method)
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(target_loader_test, is_log=True)
            loss_val, acc_val = test(target_loader_val, is_log=False)
            extractor.train()
            clasifier.train()
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                best_acc_test = acc_test
            print('best_acc_test: %.1f  best_acc_val: %.1f \n' % (best_acc_test,
                                                                  best_acc_val))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %.1f final %.1f \n' % (step,
                                                             best_acc_test,
                                                             acc_val))
            extractor.train()
            clasifier.train()
            if args.save_check:
                print('saving model \n')
                torch.save(extractor.module.state_dict(), \
                           extractor_pth.replace('resume_step', str(step)))
                torch.save(clasifier.module.state_dict(), \
                           clasifier_pth.replace('resume_step', str(step)))
                if step > args.save_interval and os.path.exists(extractor_pth.replace('resume_step', str(step-args.save_interval))):
                    os.remove(extractor_pth.replace('resume_step', str(step-args.save_interval))) 
                    os.remove(clasifier_pth.replace('resume_step', str(step-args.save_interval))) 


def test(loader, is_log=False):
    extractor.eval()
    clasifier.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    pred_all = np.zeros((0, num_class))
    cls_criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    uh_num = 0
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            imag_t.resize_(data_t[0].size()).copy_(data_t[0])
            labl_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = extractor(imag_t)
            pred = clasifier(feat)
            pred_all = np.r_[pred_all, pred.data.cpu().numpy()]
            size += imag_t.size(0)
            pred1 = pred.data.max(1)[1]
            for t, p in zip(labl_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(labl_t.data).cpu().sum()
            probs, _ = torch.max(torch.softmax(pred.detach_(), dim=-1), dim=-1)
            uh_num += probs.ge(args.thr).float().sum()
            test_loss += cls_criterion(pred, labl_t) / len(loader)
    print('\nTest set: Average loss: {:.4f},   '
          'Accuracy: {}/{} ({:.1f}%)    High Target Num: {}/{} \n'.
          format(test_loss, correct, size,
                 100. * correct / size, int(uh_num), size))
    file = 'save_record/' + str(args.dataset) + '_'  \
                          + str(args.sample_per_class) + '_'   \
                          + str(args.source) + '_'   \
                          + str(args.target) + '_'   \
                          + str(args.method) + '.txt'
    if is_log:
        with open(file,mode='a',encoding='utf8') as tf3:
            tf3.write(str(int(uh_num))+ ' ' + str(int(correct)) + ' ' + str(size) + '\n')

    return test_loss, 100. * float(correct) / size



train()
