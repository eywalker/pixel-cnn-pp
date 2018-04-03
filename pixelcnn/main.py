import time
import os
import argparse
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from .utils import * 
from .model import * 
from PIL import Image
import os

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='saved',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist|cifarbw')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Point to checkpoint file to restore training from')
parser.add_argument('-z', '--resume', type=int, default=None,
                    help='Epoch number to resume from')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()

model_dir = os.path.join(args.save_dir, 'models')
image_dir = os.path.join(args.save_dir, 'images')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
if args.resume is None:
    assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

sample_batch_size = 25

obs_map = {
  'mnist': (1, 28, 28),
  'cifarbw': (1, 32, 32),
  'cifar': (3, 32, 32)
}

obs = obs_map['args.dataset']
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':2, 'pin_memory':True, 'drop_last':True}
transform_list = [transforms.ToTensor(), rescaling]

if args.dataset == 'mnist':
    ds_transforms = transforms.Compose(transform_list)
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif args.dataset=='cifarbw':
    transform_list = [transforms.Grayscale(), transforms.ToTensor(), rescaling]
    ds_transforms = transforms.Compose(transform_list)
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif args.dataset=='cifar' :
    ds_transforms = transforms.Compose(transform_list)
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifarbw, cifar}'.format(args.dataset))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = DataParallel(model, [0, 1, 2, 3])
model = model.cuda()
epoch_start = 0

if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')
elif args.resume is not None:
    load_part_of_model(model, '{}/{}_{}.pth'.format(model_dir, model_name, args.resume))
    epoch_start = args.resume + 1
    print('model from epoch {} loaded'.format(args.resume))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
writes = 0
for epoch in range(epoch_start, args.max_epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.cuda(async=True)
        input = Variable(input)
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        if (batch_idx +1) % args.print_every == 0 : 
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno), 
                (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()
            

    # decrease learning rate
    scheduler.step()
    
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda(async=True)
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.data[0]
        del loss, output

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))
    
    if (epoch + 1) % args.save_interval == 0:
        torch.save(model.state_dict(), '{}/{}_{}.pth'.format(model_dir, model_name, epoch))
        print('sampling...')
        sample_t = sample(model)
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t,'{}/{}_{}.png'.format(image_dir, model_name, epoch), 
                nrow=5, padding=0)
