import os
import logging
import argparse
import torch as T
import torch.nn as nn
import torchvision.transforms as transforms
from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import *
from Dataloader import Loader
from Retrieval import DoRetrieval

def get_args_parser():
    parser = argparse.ArgumentParser('DHD', add_help=False)

    parser.add_argument('--gpu_id', default="0,1", type=str, help="""Define GPU id.""")
    parser.add_argument('--data_dir', default="data", type=str, help="""Path to dataset.""")
    parser.add_argument('--dataset', default="coco", type=str, help="""Dataset name: imagenet, nuswide_m, coco.""")
    
    parser.add_argument('--batch_size', default=32, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--num_workers', default=4, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--encoder', default="SwinT", type=str, help="""Encoder network: ResNet, AlexNet, ViT, DeiT, SwinT.""")
    parser.add_argument('--N_bits', default=64, type=int, help="""Number of bits to retrieval.""")
    parser.add_argument('--init_lr', default=3e-4, type=float, help="""Initial learning rate.""")
    parser.add_argument('--warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
    parser.add_argument('--lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on self knowledge distillation.""")
    parser.add_argument('--lambda2', default=0.1, type=float, help="""Balancing hyper-paramter on bce quantization.""")
    parser.add_argument('--std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
    parser.add_argument('--temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
    parser.add_argument('--transformation_scale', default=0.5, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")

    parser.add_argument('--max_epoch', default=2, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--eval_epoch', default=1, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--eval_init', default=0, type=int, help="""Compute mAP after N-th epoch.""")
    parser.add_argument('--output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")

    return parser

class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(T.FloatTensor(NB_CLS, N_bits), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        X = self.Hash(X)
        return T.tanh(X)

def train(args):
    # Set CUDA_VISIBLE_DEVICES environment variable at the start
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Specify device(s)
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    # Setup logging
    log_file = os.path.join(args.output_dir, 'training.log')
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    path = args.data_dir
    dname = args.dataset

    N_bits = args.N_bits
    init_lr = args.init_lr
    batch_size = args.batch_size

    if dname == 'imagenet':
        NB_CLS = 100
        Top_N = 1000
    elif dname == 'nuswide':
        NB_CLS = 21
        Top_N = 5000
    elif dname == 'nuswide_m':
        NB_CLS = 21
        Top_N = 5000
    elif dname == 'coco':
        NB_CLS = 80
        Top_N = 5000
    else:
        logging.error("Wrong dataset name.")
        return

    Img_dir = path + '/' + dname + '256'
    Train_dir = path + '/' + dname + '_Train.txt'
    Gallery_dir = path + '/' + dname + '_DB.txt'
    Query_dir = path + '/' + dname + '_Query.txt'
    org_size = 256
    input_size = 224
    
    AugS = Augmentation(org_size, 1.0)
    AugT = Augmentation(org_size, args.transformation_scale)

    Crop = nn.Sequential(transforms.CenterCrop(input_size))
    Norm = nn.Sequential(transforms.Normalize(mean=T.tensor([0.485, 0.456, 0.406]), std=T.tensor([0.229, 0.224, 0.225])))

    trainset = Loader(Img_dir, Train_dir, NB_CLS)
    trainloader = T.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
                                            shuffle=True, num_workers=args.num_workers)
    
    if args.encoder == 'AlexNet':
        Baseline = AlexNet()
        fc_dim = 4096
    elif args.encoder == 'ResNet':
        Baseline = ResNet()
        fc_dim = 2048
    elif args.encoder == 'ViT':
        Baseline = ViT('vit_base_patch16_224')
        fc_dim = 768
    elif args.encoder == 'DeiT':
        Baseline = DeiT('deit_base_distilled_patch16_224')
        fc_dim = 768
    elif args.encoder == 'SwinT':
        Baseline = SwinT('swin_base_patch4_window7_224')
        fc_dim = 1024
    else:
        logging.error("Wrong dataset name.")
        return

    H = Hash_func(fc_dim, N_bits, NB_CLS)
    net = nn.Sequential(Baseline, H)

    # Enable DataParallel
    if T.cuda.device_count() > 1:
        logging.info("Let's use {} GPUs!".format(T.cuda.device_count()))
        net = nn.DataParallel(net)
    
    net.to(device)

    HP_criterion = HashProxy(args.temp)
    HD_criterion = HashDistill()
    REG_criterion = BCEQuantization(args.std)

    params = [{'params': Baseline.parameters(), 'lr': 0.05 * init_lr},
              {'params': H.parameters()}]

    optimizer = T.optim.Adam(params, lr=init_lr, weight_decay=10e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0, last_epoch=-1)
    
    MAX_mAP = 0.0
    mAP = 0.0

    for epoch in range(args.max_epoch):  # loop over the dataset multiple times
        logging.info('Epoch: %d, LR: %f, %f', epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            l1 = T.tensor(0., device=device)
            l2 = T.tensor(0., device=device)
            l3 = T.tensor(0., device=device)

            Is = Norm(Crop(AugS(inputs)))
            It = Norm(Crop(AugT(inputs)))

            Xt = net(It)
            l1 = HP_criterion(Xt, H.P, labels)

            Xs = net(Is)
            l2 = HD_criterion(Xs, Xt) * args.lambda1
            l3 = REG_criterion(Xt) * args.lambda2

            loss = l1 + l2 + l3
            loss.backward()
            
            optimizer.step()
            
            # print statistics
            C_loss += l1.item()
            S_loss += l2.item()
            R_loss += l3.item()

            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                logging.info('[%3d] C: %.4f, S: %.4f, R: %.4f, mAP: %.4f, MAX mAP: %.4f',
                    i + 1, C_loss / 10, S_loss / 10, R_loss / 10, mAP, MAX_mAP)
                C_loss = 0.0
                S_loss = 0.0
                R_loss = 0.0
 
        if epoch >= args.warm_up:
            scheduler.step()

        if (epoch + 1) % args.eval_epoch == 0 and (epoch + 1) >= args.eval_init:
            mAP = DoRetrieval(device, net.eval(), Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args)

            if mAP > MAX_mAP:
                MAX_mAP = mAP
                # Save checkpoint
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                T.save({'epoch': epoch + 1,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'mAP': mAP}, checkpoint_path)
                logging.info(f'Checkpoint saved at {checkpoint_path}')

            net.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DHD', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)
