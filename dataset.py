import os
from torch.utils.data import DataLoader
from torchvision.datasets import StanfordCars, CIFAR10, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
transforms.RandAugment


def create_dataloader(args, dataset='cifar10', randaug=False):
    nw = os.cpu_count()  # number of workers
    if dataset == 'cifar10':
        if randaug:
            transform_train = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandAugment(magnitude=10),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761)),
            ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        trainset = CIFAR10(root="./data",
                                train=True,
                                download=True,
                                transform=transform_train)
        testset = CIFAR10(root="./data",
                               train=False,
                               download=True,
                               transform=transform_test)

        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=nw,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.batch_size,
                                 num_workers=nw,
                                 pin_memory=True)
    elif dataset == 'StanfordCars':
        if randaug:
            transform_train = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandAugment(magnitude=10),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
        ])

        trainset = StanfordCars(root="./data",
                                split='train',
                                download=True,
                                transform=transform_train)
        testset = StanfordCars(root="./data",
                               split='test',
                               download=True,
                               transform=transform_test)

        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=nw,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.batch_size,
                                 num_workers=nw,
                                 pin_memory=True)
    elif dataset == 'cinic':
        # wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
        cinic_directory = "./data/cinic-10"
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]

        if randaug:
            transform_train = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandAugment(magnitude=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean, std=cinic_std),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean, std=cinic_std),
            ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
        ])

        train_loader = DataLoader(ImageFolder(cinic_directory + '/train',
                                              transform=transform_train), batch_size=args.batch_size, shuffle=True, num_workers=nw)

        test_loader = DataLoader(ImageFolder(cinic_directory + '/test',
                                             transform=transform_test), batch_size=args.batch_size, shuffle=True, num_workers=nw)
    return train_loader, test_loader
