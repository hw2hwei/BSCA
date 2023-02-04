import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.resnet import resnet34
from models.basenet import Efficientnet_B0_Base, Efficientnet_B4_Base, GoogLeNet_Base, VGG16_Base, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from losses.loss_Align import Align_Loss
from losses.loss_PL import PL_Loss
from losses.loss_MMD import mmd_loss
import torch.nn.functional as F


# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--resume_step', type=int, default=0, metavar='RN',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--steps', type=int, default=2010, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='BSCA',
                    help='BSCA is proposed method, ENT is entropy minimization,'
                         'S+T, PL, Alg_s, Alg_u, BSCA')
parser.add_argument('--thr', type=float, default=0.5, metavar='THR',
                    help='entropy pseudo label thr (default: 0.8)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.02, metavar='LAM',  # previous: 0.1
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=True,
                    help='save checkpoint or not')
parser.add_argument('--pth', type=str, default='./save_model',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='efficientnet_b0',
                    help='which network to use')
parser.add_argument('--n_workers', type=int, default=4,
                    help='number of workers of python')
parser.add_argument('--root', type=str, default='/home/Datasets/RSSC',
                    help='source domain')
parser.add_argument('--source', type=str, default='NWPU-RESISC45',
                    help='source domain')
parser.add_argument('--target', type=str, default='AID',
                    help='target domain')
parser.add_argument('--sample_per_class', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--memory_per_class', type=int, default=32,
                    help='number of labeled examples in the target')


args = parser.parse_args()
print('Source %s Target %s Labeled num perclass %s Network %s' %
      (args.source, args.target, args.sample_per_class, args.net))
source_loader, target_loader, target_loader_u, target_loader_val, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()


torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    extractor = resnet34()
    inc = 512
elif args.net == "efficientnet_b0":
    extractor = Efficientnet_B0_Base()
    inc = 1280
elif args.net == "efficientnet_b4":
    extractor = Efficientnet_B4_Base()
    inc = 1792
elif args.net == "googlenet":
    extractor = GoogLeNet_Base()
    inc = 1024
elif args.net == "vgg16":
    extractor = VGG16_Base()
    inc = 512
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
clasifier = Predictor_deep(num_class=len(class_list), inc=inc,
                                temp=args.T)
weights_init(clasifier)

# class Fullmodel(nn.Module):
#     def __init__(self, extractor, clasifier):
#         super(Fullmodel, self).__init__()
#         self.clasifier = clasifier
#         self.extractor = extractor

#     def forward(self, x):
#         x = self.extractor(x)
#         x = self.clasifier(x)
#         return x

# model = Fullmodel(extractor, clasifier)

# from thop import profile
# input = torch.randn(1, 3, 224, 224)
# macs, params = profile(model, inputs=(input, ))
# print (macs, params)


if os.path.exists(args.pth) == False:
    os.mkdir(args.pth)
extractor_pth = os.path.join(args.pth,
                             "extractor_{}_{}_{}_"
                             "to_{}_{}_thr_{}_best.pth.tar".
                             format(args.net, args.method, 
                                    args.source, args.target,
                                    args.sample_per_class,
                                    args.thr))
clasifier_pth = os.path.join(args.pth,
                             "clasifier_{}_{}_{}_"
                             "to_{}_{}_thr_{}_best.pth.tar".
                             format(args.net, args.method, 
                                    args.source, args.target,
                                    args.sample_per_class,
                                    args.thr))

extractor = extractor.cuda()
clasifier = clasifier.cuda()
extractor = nn.DataParallel(extractor)
clasifier = nn.DataParallel(clasifier)

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

train_file = 'save_record/train_{}_{}_to_{}.txt'.format(args.method, args.source, args.target)
test_file  = 'save_record/test_{}_num_{}_thr_{}_best.txt'.\
                format(args.method, args.sample_per_class, args.thr)

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

    for step in range(args.resume_step, all_step+1):
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
        imag_s_bar.resize_(data_s[2].size()).copy_(data_s[2])
        imag_t_bar.resize_(data_t[2].size()).copy_(data_t[2])
        imag_u_bar.resize_(data_u[2].size()).copy_(data_u[2])

        zero_grad_all()
        imag_st = torch.cat((imag_s, imag_t), 0)
        labl_st = torch.cat((labl_s, labl_t), 0)
        feat_st = extractor(imag_st)
        feat_s = feat_st[:imag_s.size(0)]
        feat_t = feat_st[imag_s.size(0):]
        lant_st, pred_st = clasifier(feat_st)    
        loss_cls = cls_criterion(pred_st, labl_st)

        if args.method != 'S+T':
            feat_t_bar = extractor(imag_t_bar)

            feat_u = extractor(imag_u)            
            lant_u, pred_u = clasifier(feat_u)
            feat_u_bar = extractor(imag_u_bar)            
            _, pred_u_bar = clasifier(feat_u_bar)

            # mmd loss
            loss_mmd = 10*mmd_loss(lant_st, lant_u)

            # pl loss
            criterion_reduce = nn.CrossEntropyLoss(reduce=False).cuda()
            prob_u = pred_u.data.max(1)[1].detach()
            ent = - torch.sum(F.softmax(pred_u, 1) * (torch.log(F.softmax(pred_u, 1) + 1e-5)), 1)
            mask = (ent < args.thr).float().detach()
            loss_pl = (mask * criterion_reduce(pred_u, prob_u)).sum(0) / (1e-5 + mask.sum())

            # alg loss
            value, index = torch.sort(mask, dim=0, descending=True) 
            cnt = int(mask.sum().cpu().numpy())
            feat_h = feat_u[index.cpu().numpy()]
            feat_h_bar = feat_u_bar[index.cpu().numpy()]
            labl = prob_u[index.cpu().numpy()]
            if cnt == 0:
                feat_h = None
                feat_h_bar = None
                labl_h = None
            else:
                feat_h = feat_h[:cnt]
                feat_h_bar = feat_h_bar[:cnt]
                labl_h = labl[:cnt]

            # pl loss
            source_centroids, target_centroids = alg_criterion.return_centroids()
            # loss_alg = alg_criterion(feat_s_bar, labl_s, target_centroids)
            loss_alg = alg_criterion(feat_s, labl_s, target_centroids)
            if feat_h_bar != None:
            # if feat_h != None:
                feat_th_bar = torch.cat((feat_t_bar, feat_h_bar), dim=0)
                # feat_th = torch.cat((feat_t, feat_h), dim=0)
                labl_th = torch.cat((labl_t, labl_h), dim=0)
            else:
                # feat_th_bar = feat_t_bar
                feat_th_bar = feat_t_bar
                labl_th = labl_t
            loss_alg += alg_criterion(feat_th_bar, labl_th, source_centroids) 
            loss_alg *= args.lamda

            # update the memory
            with torch.no_grad():
                alg_criterion.update_memory(feat_s, labl_s, data_type='source')
                alg_criterion.update_memory(feat_t, labl_t, data_type='target')
                if feat_h != None:
                    alg_criterion.update_memory(feat_h, labl_h, data_type='target_ul')


        # loss backpropogation
        if args.method == 'S+T':
            loss = loss_cls
            loss_pl = loss_cls
            loss_alg = loss_cls         
            loss_mmd = loss_cls   
        elif args.method == 'PL':
            loss = loss_cls + loss_pl
        elif args.method == 'Alg_s':
            loss = loss_cls + loss_pl + loss_alg
        elif args.method == 'Alg_u':
            loss = loss_cls + loss_pl + loss_mmd
        else:
            loss = loss_cls + loss_pl + loss_alg + loss_mmd
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()

        extractor.zero_grad()
        clasifier.zero_grad()
        zero_grad_all()
        if step % args.log_interval==0:
            log_train = 'S: {}   T: {}   Train Ep: {}   lr: {:.6f} ' \
                        '  Loss cls: {:.6f}   Loss pl: {:.6f}   Loss alg: {:.6f}  Loss mmd: {:.6f}  ' \
                        '  Method: {}'.format(args.source, args.target,
                                                step, lr, loss_cls,
                                                loss_pl, loss_alg, loss_mmd, args.method)
            print(log_train)

        if step % args.save_interval==0 and step!=0:
            loss_val, acc_val = test(target_loader_val)
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                if args.save_check:
                    print('saving model \n')
                    torch.save(extractor.module.state_dict(), extractor_pth)
                    torch.save(clasifier.module.state_dict(), clasifier_pth)
            print ('record %s' % train_file)
            with open(train_file, mode='a') as tf:
                tf.write(str(acc_val) + '\n')

    # test
    extractor.module.load_state_dict(torch.load(extractor_pth))
    clasifier.module.load_state_dict(torch.load(clasifier_pth))

    _, acc_test = test(target_loader_u)
    print ('acc_test: {:.1f} '.format(acc_test))
    with open(test_file, mode='a') as tf:
        tf.write('{}_{}_to_{}_test_accuracy: {}\n'\
                .format(args.net, args.source, args.target, str(np.around(acc_test,1))))


def test(loader):
    extractor.eval()
    clasifier.eval()
    test_loss = 0
    correct = 0
    size = 0
    uh_num = 0
    num_class = len(class_list)
    pred_all = np.zeros((0, num_class))
    cls_criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            imag_t.resize_(data_t[0].size()).copy_(data_t[0])
            labl_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = extractor(imag_t)
            pred = clasifier(feat)[1]
            pred_all = np.r_[pred_all, pred.data.cpu().numpy()]
            size += imag_t.size(0)
            pred1 = pred.data.max(1)[1]
            for t, p in zip(labl_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(labl_t.data).cpu().sum()
            test_loss += cls_criterion(pred, labl_t) / len(loader)
    print('Test set: Average loss: {:.4f},   '
          'Accuracy: {}/{} ({:.1f}%)  '.
          format(test_loss, correct, size,
                 100. * correct / size))

    extractor.train()
    clasifier.train()

    return test_loss, 100. * float(correct) / size


train()






