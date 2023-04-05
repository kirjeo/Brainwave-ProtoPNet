import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset_rewrite import Sequence_Data_Alt  
import numpy as np
import cv2
import matplotlib.pyplot as plt

import re

import os
from log import create_logger
import save
from helpers import makedir
import model
import find_nearest
import train_and_test as tnt

from preprocess import preprocess_input_function

import argparse

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-prototypeind', nargs=1, type=int)

#parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]
prot_ind=args.prototypeind[0]

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

# load the model
print('load model from ' + load_model_path)
ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)


prototype_shape = ppnet.prototype_shape
#max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

#todo
# img_size = ppnet_multi.module.img_size

# load the data
# must use unaugmented (original) dataset
from settings import train_push_dir, test_dir
train_dir = train_push_dir

# batch_size = 100

# train set: do not normalize
# train_dataset = datasets.ImageFolder(
#     train_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#     ]))
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True,
#     num_workers=4, pin_memory=False)

# # test set: do not normalize
# test_dataset = datasets.ImageFolder(
#     test_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#     ]))
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=True,
#     num_workers=4, pin_memory=False)

# load the test data and check test accuracy
from settings import test_dir

sequence_length = 200
train_dataset = Sequence_Data_Alt(
    data_path='../datasets/split_datasets/main_train_split_80_threshold_8.csv',
    target_sequence_length=sequence_length)

test_dataset = Sequence_Data_Alt(data_path=test_dir,
            target_sequence_length=sequence_length, 
            predefined_label_dict=train_dataset.label_dict)
            

# root_dir_for_saving_train_seq = os.path.join(load_model_dir,
#                                                 load_model_name.split('.pth')[0] + '_nearest_train')
# root_dir_for_saving_test_seq = os.path.join(load_model_dir,
#                                                 load_model_name.split('.pth')[0] + '_nearest_test')
# makedir(root_dir_for_saving_train_seq)
# makedir(root_dir_for_saving_test_seq)

save_dir= './test_global'
model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])
save_analysis_path = os.path.join(save_dir, model_base_architecture,
                                  experiment_run, load_model_name)
makedir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))

class_specific = True
load_img_dir = os.path.join(load_model_dir, 'seq')

##### HELPER FUNCTIONS FOR PLOTTING
# TODO: Change all this image saving stuff to operate on sequences
def save_prototype(fname, epoch, index):
    p_seq = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype_'+str(index)+'_original.npy'))
    #plt.axis('off')
    np.save(fname, p_seq)

def save_prototype_patch(fname, epoch, index):
    p_seq = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype_'+str(index)+'_patch.npy'))
    #plt.axis('off')
    np.save(fname, p_seq)

def save_test_seq_patch(fname, patch_start, patch_end, test_seq):
    test_seq = test_seq[0]
    if patch_start < 0:
        # Handle zero padding
        target_patch = test_seq[:, :patch_end] #.cpu().detach().numpy()
        zeros = np.zeros((test_seq.shape[0], -patch_start))
        target_patch = np.concatenate((zeros, target_patch), axis=-1)

    elif patch_end > test_seq.shape[-1]:
        # Handle zero padding
        target_patch = test_seq[:, patch_start:]#.cpu().detach().numpy()
        zeros = np.zeros((target_patch.shape[0], patch_end - test_seq.shape[-1]))
        target_patch = np.concatenate((target_patch, zeros), axis=-1)
    else:
        target_patch = test_seq[:, patch_start:patch_end]#.cpu().detach().numpy()
    np.save(fname, target_patch)

def save_test_seq(fname, test_seq):
    np.save(fname, test_seq)

def save_act_map(fname, act_map):
    np.save(fname, act_map)


prototype_img_identity = torch.argmax(ppnet.prototype_class_identity, dim=1)

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'prototype_'+str(prot_ind)+'_patch.npy'))


save_prototype(os.path.join(save_analysis_path,
                                'original_prototype.npy'),
                   start_epoch_number, prot_ind)
save_prototype_patch(os.path.join(save_analysis_path,
                                'prototype_patch.npy'),
                   start_epoch_number, prot_ind)
# load the test image and forward it through the network

#to loop over
activation_pattern_table=[]
proto_act={}
test_dataset_len=int(test_dataset.num_samples)
print('test_dataset_len: ', test_dataset_len)
arg_max_list=[]
for i in range(test_dataset_len):

    seq, label = test_dataset.__getitem__(i)
    test_sequence = seq
    if type(test_sequence) is str:
        test_sequence_numpy = np.expand_dims(test_dataset.sequence_to_array(test_sequence), axis=0)
    else:
        test_sequence_numpy = np.expand_dims(test_sequence, axis=0)

    sequence_test = torch.tensor(test_sequence_numpy).cuda()

    conv_output, prototype_activation_patterns = ppnet.push_forward(sequence_test)
    activation_pattern_table.append(prototype_activation_patterns[:, prot_ind].cpu().detach().numpy())
    # array_act, sorted_indices_act = torch.sort(prototype_activations[idx])    

    max_proto_act = np.max(prototype_activation_patterns[:, prot_ind].cpu().detach().numpy())
    arg_max_proto_act= list(np.unravel_index(np.argmax(prototype_activation_patterns[:, prot_ind].cpu().detach().numpy(), axis=None),
                                prototype_activation_patterns.shape))
    arg_max_list.append(arg_max_proto_act[2])
    proto_act[i]=max_proto_act

print(activation_pattern_table)
print(arg_max_list)
makedir(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind)))
print('protoact: ',proto_act)
print('protoact_len: ',len(proto_act))
proto_act_sorted=sorted(proto_act.items(), key=lambda item: item[1])
print('sorted: ', proto_act_sorted)



for i in range(1, 11):
    # print(proto_act_sorted[-i][0])
    seq, label = test_dataset.__getitem__(proto_act_sorted[-i][0])
    if type(seq) is str:
        test_sequence_numpy = np.expand_dims(train_dataset.sequence_to_array(seq), axis=0)
    else:
        test_sequence_numpy = np.expand_dims(seq, axis=0)

    save_test_seq(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind), 'top-{0}_original_test_seq_{1}.npy'.format(i, str(label))),
                                        test_sequence_numpy)
    save_act_map(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind), 'top-{0}_prototype_activation_map_{1}.npy'.format(i, str(proto_act_sorted[-i][0]))),
                    activation_pattern_table[proto_act_sorted[-i][0]])
    upsampling_factor = 2
    proto_h = prototype_shape[-1]
    prototype_layer_stride = 1
    patch_start = upsampling_factor * (arg_max_list[proto_act_sorted[-i][0]] * prototype_layer_stride - proto_h // 2)
    patch_end = upsampling_factor * (arg_max_list[proto_act_sorted[-i][0]] + proto_h // 2) + upsampling_factor
    save_test_seq_patch(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind),
                                'top-{}_activated_test_patch.npy'.format(i)),
                                patch_start, patch_end, test_sequence_numpy)
    log('prototype class: {}'.format(prototype_img_identity[prot_ind]))
    log('test seq index: {0}'.format(proto_act_sorted[-i][0]))
    log('test seq class identity: {0}'.format(label))
    log('activation value (similarity score): {0}'.format(proto_act_sorted[-i][1]))
    log('--------------------------------------------------------------')


# for j in range(ppnet.num_prototypes):
#     makedir(os.path.join(root_dir_for_saving_train_seq, str(j)))
#     makedir(os.path.join(root_dir_for_saving_test_seq, str(j)))
#     save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_seq, str(j),
#                                                              'prototype_in_original_pimg.png'),
#                                           epoch=start_epoch_number,
#                                           index=j,
#                                           bbox_height_start=prototype_info[j][1],
#                                           bbox_height_end=prototype_info[j][2],
#                                           bbox_width_start=prototype_info[j][3],
#                                           bbox_width_end=prototype_info[j][4],
#                                           color=(0, 255, 255))
#     save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_seq, str(j),
#                                                              'prototype_in_original_pimg.png'),
#                                           epoch=start_epoch_number,
#                                           index=j,
#                                           bbox_height_start=prototype_info[j][1],
#                                           bbox_height_end=prototype_info[j][2],
#                                           bbox_width_start=prototype_info[j][3],
#                                           bbox_width_end=prototype_info[j][4],
#                                           color=(0, 255, 255))

# k = 5

# find_nearest.find_k_nearest_patches_to_prototypes(
#         dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
#         prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
#         k=k+1,
#         preprocess_input_function=preprocess_input_function, # normalize if needed
#         full_save=True,
#         root_dir_for_saving_seq=root_dir_for_saving_train_seq,
#         log=print)

# find_nearest.find_k_nearest_patches_to_prototypes(
#         dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
#         prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
#         k=k,
#         preprocess_input_function=preprocess_input_function, # normalize if needed
#         full_save=True,
#         root_dir_for_saving_seq=root_dir_for_saving_test_seq,
#         log=print)
