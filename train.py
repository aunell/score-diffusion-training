#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, json, argparse
from dotmap import DotMap
sys.path.append('.')

# Args
parser = argparse.ArgumentParser()
# parser.add_argument('--config_path', type=str)
parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

from ncsnv2.models        import get_sigmas
from ncsnv2.models.ema    import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest, NCSNv2Deeper, NCSNv2
from ncsnv2.losses        import get_optimizer

from parameters import *
from losses     import *
from parse import *
from dataloaders import *
# from loaders          import *
from torch.utils.data import DataLoader

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

args = parser.parse_args()
args.log_path = os.path.join(args.exp, 'logs', args.doc)

# parse config file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f) 
new_config = dict2namespace(config)

tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

if os.path.exists(args.log_path):
    overwrite = False
    response = input("Folder already exists. Overwrite? (Y/N)")
    if response.upper() == 'Y':
        overwrite = True

    if overwrite:
        shutil.rmtree(args.log_path)
        shutil.rmtree(tb_path)
        os.makedirs(args.log_path)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
    else:
        print("Folder exists. Program halted.")
        sys.exit(0)
else:
    os.makedirs(args.log_path)

with open(os.path.join(args.log_path, 'config.json'), 'w') as f:
    yaml.dump(new_config, f, default_flow_style=False)

# Model config
config = DotMap(json.load(open(parser.parse_args().config)))

config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
config.output=args.log_path
handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler1)
logger.addHandler(handler2)

average_loss = []
minimum_loss = []
maximum_loss = []
median_loss = []
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.model.gpu)
config.device                      = 'cuda:0'

# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999 # Exponential moving average, for stable FID scores (Song'20)
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 0.0001
config.optim.beta1         = 0.9
config.optim.amsgrad       = False
config.optim.eps           = 0.001

# Training
config.training.num_workers    = 8
config.training.anneal_power   = 2
config.training.log_all_sigmas = False
config.training.eval_freq      = 100 # In steps

# Data
config.data.channels       = 2 # {Re, Im}

print('\nDataset: ' + config.data.file)
print('Dataloader: ' + config.data.dataloader)
print('Loss Function: ' + config.model.loss)

# Get datasets and loaders for channels
dataset     = globals()[config.data.dataloader](config)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
                         shuffle=True, num_workers=config.training.num_workers, 
                         drop_last=True)

pairwise_dist_path = './parameters/' + config.data.file + '.txt'
if not os.path.exists(pairwise_dist_path):
    dataset_matrix= np.zeros((len(dataset), 2, 384, 384))
    for i, config.current_sample in tqdm(enumerate(dataloader)):
        dataset_matrix[i:i+4]= config.current_sample['X'][0:4]
        break
    pairwise_dist(config, dataset_matrix, tqdm)

config.data.image_size = [next(iter(dataloader))[config.training.X_train].shape[2], next(iter(dataloader))[config.training.X_train].shape[3]]
print('Image Dimension: ' + str(config.data.image_size) + '\n')
config.model.sigma_begin = np.loadtxt(pairwise_dist_path)

if isinstance(config.model.sigma_rate, str):
    config.model.sigma_rate = globals()[config.model.sigma_rate](dataset, config)

if not config.model.sigma_end:
      config.model.sigma_end = config.model.sigma_begin * config.model.sigma_rate ** (config.model.num_classes - 1)

# Get a model
if config.model.depth == 'large':
    diffuser = NCSNv2Deepest(config).cuda()
elif config.model.depth == 'medium':
    diffuser = NCSNv2Deeper(config).cuda()
elif config.model.depth == 'low':
    diffuser = NCSNv2(config).cuda()

# Get a collection of sigma values
if config.model.get_sigmas:
    config.training.sigmas = globals()[config.model.get_sigmas](config)
    diffuser.sigmas = config.training.sigmas.clone().detach()
else:
    config.training.sigmas = get_sigmas(config)

config.model.sigma_end = diffuser.sigmas[-1].cpu().numpy()
config.model.step_size = step_size(config)
print('\nStep Size: ' + str(config.model.step_size))
print('Sigma Begin: ' + str(config.model.sigma_begin))
print('Sigma Rate: ' + str(config.model.sigma_rate))
print('Sigma End: ' + str(config.model.sigma_end) + '\n')
    
# Get optimizer
optimizer = get_optimizer(config, diffuser.parameters())

# Counter
start_epoch = 0
step = 0
ema_helper = EMAHelper(mu=config.model.ema_rate)
ema_helper.register(diffuser)

# More logging
config.log_path = './models/' + config.data.file + '_' + config.data.dataloader + '/\
sigma_begin%d_sigma_end%.4f_num_classes%.1f_sigma_rate%.4f_epochs%.1f' % (
    config.model.sigma_begin, config.model.sigma_end,
    config.model.num_classes, config.model.sigma_rate, 
    config.training.n_epochs)

if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)

# Logged metrics
print('\n')
# train_loss, train_nrmse, train_nrmse_img, train_metric_1, train_metric_2  = [], [], [], [], []
train_loss, train_nrmse, train_nrmse_img, train_minSigma, train_maxSigma, train_medSigma  = [], [], [], [], [], []

for config.epoch in tqdm(range(start_epoch, config.training.n_epochs)):
    for i, config.current_sample in tqdm(enumerate(dataloader)):
        # Safety check
        diffuser.train()
        step += 1
        # Move data to device
        for key in config.current_sample:
            config.current_sample[key] = config.current_sample[key].cuda()

        # Get loss on Hermitian channels
        # loss, nrmse, nrmse_img, metric_1, metric_2 = globals()[config.model.loss](diffuser, config, config.epoch)

        loss, nrmse, nrmse_img, minSigma, maxSigma, medSigma = globals()[config.model.loss](diffuser, config, config.epoch)

        # Keep a running loss
        if step == 1:
            running_loss = loss.item()
            running_nrmse = nrmse.item()
            running_nrmse_img = nrmse_img.item()
            # running_metric_1 = metric_1.item()
            # running_metric_2 = metric_2.item()
            running_minSigma = minSigma.item()
            running_maxSigma = maxSigma.item()
            running_medSigma = medSigma.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.item()
            running_nrmse = 0.99 * running_nrmse + 0.01 * nrmse.item()
            running_nrmse_img = 0.99 * running_nrmse_img + 0.01 * nrmse_img.item()
            # running_metric_1 = 0.99 * running_metric_1 + 0.01 * metric_1.item()
            # running_metric_2 = 0.99 * running_metric_2 + 0.01 * metric_2.item()
            running_minSigma = 0.99 * running_minSigma + 0.01 * minSigma.item()
            running_maxSigma = 0.99 * running_maxSigma + 0.01 * maxSigma.item()
            running_medSigma = 0.99 * running_medSigma + 0.01 * medSigma.item()
    
        # Log
        train_loss.append(loss.item())
        train_nrmse.append(nrmse.item())
        train_nrmse_img.append(nrmse_img.item())
        # train_metric_1.append(metric_1.item())
        # train_metric_2.append(metric_2.item())
        train_minSigma.append(minSigma.item())
        train_maxSigma.append(maxSigma.item())
        train_medSigma.append(medSigma.item())
        
        # Step and EMA update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_helper.update(diffuser)
            
        # Verbose
        if step % config.training.eval_freq == 0:
            # Print
            # print('Epoch %d, Step %d, Loss (EMA) %.3f, NRMSE (Noise) %.3f, NRMSE (Image) %.3f, M1 %.3f, M2 %.3f' % 
                # (config.epoch, step, running_loss, running_nrmse, running_nrmse_img, running_metric_1, running_metric_2))
            print('Epoch %d, Step %d, Loss (EMA) %.3f, NRMSE (Noise) %.3f, NRMSE (Image) %.3f' % 
            (config.epoch, step, running_loss, running_nrmse, running_nrmse_img))

        if (config.epoch+1) % 50 == 0:
            # Save snapshot
            torch.save({'model_state': diffuser.state_dict()}, 
            os.path.join(config.log_path, 'epoch' + str(config.epoch+1) + '_final_model.pt'))
            continue   
    # if (config.epoch+1) % 1 == 0:
    average_loss.append(np.mean(train_loss))
    minimum_loss.append(np.mean(train_minSigma))
    maximum_loss.append(np.mean(train_maxSigma))
    median_loss.append(np.mean(train_medSigma))
    if not os.path.exists("./trainLoss/"):
        os.makedirs("./trainLoss/")

    np.savetxt('./trainLoss/' + str(config.epoch) + 'meanSigmas.txt', average_loss)
    np.savetxt('./trainLoss/' + str(config.epoch) + 'minSigma.txt', minimum_loss)
    np.savetxt('./trainLoss/' + str(config.epoch)+ 'maxSigma.txt', maximum_loss)
    np.savetxt('./trainLoss/' + str(config.epoch)+ 'medSigma.txt', median_loss) 
    # if (config.epoch+1) % 1 == 0:
    #     Save snapshot
    #     torch.save({'diffuser': diffuser,
    #                 'model_state': diffuser.state_dict(),
    #                 'config': config,
    #                 'loss': train_loss,
    #                 'nrmse_noise': train_nrmse,
    #                 'nrmse_img': train_nrmse_img,
    #                 'metric_1': train_metric_1,
    #                 'metric_2': train_metric_2}, 
    #     os.path.join(config.log_path, 'epoch' + str(config.epoch+1) + '_final_model.pt'))
    #     continue
    
# Save snapshot
# print({'diffuser': diffuser,
#             'model_state': diffuser.state_dict(),
#             'config': config,
#             'loss': train_loss,
#             'nrmse_noise': train_nrmse,
#             'nrmse_img': train_nrmse_img,
#             'metric_1': train_metric_1,
#             'metric_2': train_metric_2})
# torch.save({'diffuser': diffuser,
#             'model_state': diffuser.state_dict(),
#             'config': config,
#             'loss': train_loss,
#             'nrmse_noise': train_nrmse,
#             'nrmse_img': train_nrmse_img,
#             'metric_1': train_metric_1,
#             'metric_2': train_metric_2}, 
#    os.path.join(config.log_path, 'final_model.pt'))