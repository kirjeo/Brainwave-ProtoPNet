import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, TensorDataset
import argparse
import re
import numpy as np
from helpers import makedir
import model
from push import push_prototypes
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from dataset_rewrite import Sequence_Data_Alt  
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '..')
from augment import RandomBPDel, RandomBPFlip

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, sequence_length, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run, latent_weight, \
                    use_aug, flip_count, del_prob

print("use_aug: ", use_aug)
print("flip_count: ", flip_count)
print("del_prob: ", del_prob)
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'base_model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
seq_dir = os.path.join(model_dir, 'seq')
makedir(seq_dir)
weight_matrix_filename = 'outputL_weights'
prototype_seq_filename_prefix = 'prototype-seq'
prototype_self_act_filename_prefix = 'prototype-self-act'
# proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set

#if use_aug:
#    train_dataset = Sequence_Data_Alt(data_path=train_dir,
#                    target_sequence_length=sequence_length,
#                    augment=transforms.Compose([RandomBPFlip(flip_count), RandomBPDel(del_prob)]))
#else:
#     train_dataset = Sequence_Data_Alt(data_path=train_dir,
#                    target_sequence_length=sequence_length)
#
#test_dataset = Sequence_Data_Alt(data_path=test_dir,
#                target_sequence_length=sequence_length, predefined_label_dict=train_dataset.label_dict)
#
#train_loader = torch.utils.data.DataLoader(
#                    train_dataset, 
#                    batch_size=train_batch_size,
#                    shuffle=True)
#test_loader = torch.utils.data.DataLoader(
#                    test_dataset,
#                    batch_size=test_batch_size,
#                    shuffle=False)
#
# push set
#train_push_dataset = Sequence_Data_Alt(data_path=train_push_dir,
#                target_sequence_length=sequence_length)
#
#train_push_loader = torch.utils.data.DataLoader(
#    train_push_dataset, batch_size=train_push_batch_size, shuffle=False)
#




train_X = torch.tensor(np.load("train_x.npy"))
train_y = torch.tensor(np.load("train_y.npy"))

test_X = torch.tensor(np.load("test_x.npy"))
test_y = torch.tensor(np.load("test_y.npy"))

train_push_X = train_X
train_push_y = train_y

train_dataset = TensorDataset(train_X, train_y)
train_push_dataset = TensorDataset(train_push_X, train_push_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=train_batch_size,
                    shuffle=True)
test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=test_batch_size,
                    shuffle=False)

train_push_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=train_batch_size,
                    shuffle=True)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

print("prototype_shape")
print(prototype_shape)
# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, sequence_length=sequence_length,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              latent_weight=latent_weight)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
# ppnet_multi = torch.nn.DataParallel(ppnet)
ppnet_multi = ppnet
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
gamma = 0.3
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=gamma)

print("Trying gamma {}".format(gamma))
print("joint_lr_step_size")
print(joint_lr_step_size)
print("joint_optimizer_lrs")
print(joint_optimizer_lrs)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
print("warm_optimizer_lrs")
print(warm_optimizer_lrs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs
print("coefs: ", coefs)

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')
max_accu = 0
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        _, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
        joint_lr_scheduler.step()

    accu, cm = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    if accu > max_accu:
        max_accu = accu
        log('new max accuracy: \t{0}%'.format(max_accu * 100))
    #, cm
    #save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
    #                            target_accu=0.82, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            preprocess_input_function=None, # normalize if needed
            root_dir_for_saving_prototypes=seq_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            log=log)
        accu, cm = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                     target_accu=0.30, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu, cm = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                           target_accu=0.30, log=log)

    if epoch>num_train_epochs-50:
        log('\tconfusion matrix: \t\t\n{0}'.format(cm))

  
logclose()

