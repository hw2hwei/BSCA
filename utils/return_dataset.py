import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist
from .randaugment import  RandAugmentMC


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(args):
    root_s = os.path.join(args.root, args.source, 'splits')
    root_t = os.path.join(args.root, args.target, 'splits')
    image_set_file_s = \
        os.path.join(root_s,
                     'source_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(root_t,
                     'target_labeled_' + 
                     str(args.sample_per_class) + '_' +
                     args.target + '.txt')
    image_set_file_t_unl = \
        os.path.join(root_t,
                     'target_unlabeled_' +
                     str(args.sample_per_class) + '_' +
                     args.target + '.txt')
    image_set_file_t_val = \
        os.path.join(root_t,
                     'target_validate_' +
                     str(args.sample_per_class) + '_' +
                     args.target + '.txt')

    crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'strong': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=args.root,
                                      transform=data_transforms['train'],
                                          strong_transform=data_transforms['strong'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=args.root,
                                      transform=data_transforms['train'],
                                          strong_transform=data_transforms['strong'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_t_unl, root=args.root,
                                          transform=data_transforms['train'],
                                          strong_transform=data_transforms['strong'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=args.root,
                                          transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=args.n_workers, 
                                                shuffle=True, drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=args.n_workers,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs*2, num_workers=args.n_workers,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=bs*2,
                                    num_workers=args.n_workers,
                                    shuffle=True, drop_last=False)
    return source_loader, target_loader, target_loader_unl, target_loader_val, class_list


def return_dataset_test(args):
    root_s = os.path.join(args.root, args.source, 'splits')
    root_t = os.path.join(args.root, args.target, 'splits')
    image_set_file_s = \
        os.path.join(root_s,
                     'source_' +
                     args.source + '.txt')
    image_set_file_test = \
        os.path.join(root_t,
                     'target_unlabeled_' +
                     str(args.sample_per_class) + '_' +
                     args.target + '.txt')

    print ('source: ', image_set_file_s)
    print ('target: ', image_set_file_test)

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=args.root,
                                        transform=data_transforms['test'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=args.root,
                                          transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    bs = 64
    source_loader = \
        torch.utils.data.DataLoader(source_dataset,
                                    batch_size=bs * 2, num_workers=8,
                                    shuffle=False, drop_last=False)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=8,
                                    shuffle=False, drop_last=False)

    return source_loader, target_loader_unl, class_list
