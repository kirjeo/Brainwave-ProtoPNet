import torch
import numpy as np
import matplotlib.pyplot as plt
# import cv2
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_seq_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_network_parallel.num_prototypes
    # saves the closest distance seen so far
    global_max_proto_act = np.full(n_prototypes, -1)
    # saves the patch representation that gives the current smallest distance
    global_max_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 2],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 2],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 1],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 1],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_max_proto_act,
                                   global_max_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_seq_filename_prefix=prototype_seq_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    # if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
    #     np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
    #             proto_rf_boxes)
    #     np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
    #             proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_max_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    np.save(os.path.join(proto_epoch_dir,
                        'prototype_vectors.npy'), 
            prototype_update)
    save_self_activations(dir_for_saving_prototypes=proto_epoch_dir,
                        prototype_network_parallel=prototype_network_parallel,
                        search_batch_input=search_batch_input,
                        search_y=search_y,
                        num_classes=num_classes)
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))


def save_self_activations(dir_for_saving_prototypes, 
                        prototype_network_parallel, 
                        search_batch_input,
                        search_y,
                        num_classes):
    prototype_network_parallel.eval()

    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_act_torch = prototype_network_parallel.push_forward(search_batch)

        x_norm = torch.norm(protoL_input_torch, p=2, dim=(1))
        x_reshape = x_norm.view(protoL_input_torch.shape[0], 1, protoL_input_torch.shape[-1])
        protoL_input_torch = (protoL_input_torch / x_reshape) / (prototype_network_parallel.prototype_shape[-1])**0.5

    proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_act_torch

    class_to_seq_index_dict = {key: [] for key in range(num_classes)}
    # seq_y is the image's integer label
    for seq_index, seq_y in enumerate(search_y):
        seq_label = seq_y.item()
        class_to_seq_index_dict[seq_label].append(seq_index)

    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    upsampling_factor = 2
    # proto_w = prototype_shape[3]
    # max_act = prototype_shape[1] * prototype_shape[2]

    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(prototype_network_parallel.prototype_class_identity[j]).item()
        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_seq_index_dict[target_class]) == 0:
            continue
        proto_act_j = proto_act_[class_to_seq_index_dict[target_class]][:,j,:]

        batch_max_proto_act_j = np.amax(proto_act_j)
        if batch_max_proto_act_j >= 0.999:
            print("Grabbed activation map for prototype {}".format(j))
            batch_argmax_proto_act_j = \
                list(np.unravel_index(np.argmax(proto_act_j, axis=None),
                                      proto_act_j.shape))
            batch_argmax_proto_act_j[0] = class_to_seq_index_dict[target_class][batch_argmax_proto_act_j[0]]

            # retrieve the corresponding feature map patch
            seq_index_in_batch = batch_argmax_proto_act_j[0]

            original_seq_j = search_batch_input[seq_index_in_batch]
            original_seq_j = original_seq_j.numpy()
            center_loc = batch_argmax_proto_act_j[1]

            img_space_start = (center_loc - proto_h // 2) * upsampling_factor
            img_space_end = (center_loc + proto_h // 2) * upsampling_factor + upsampling_factor

            if img_space_start < 0:
                # Handle zero padding
                high_act_region = original_seq_j[:, :img_space_end]

                zeros = np.zeros((original_seq_j.shape[0], -(img_space_start)))
                high_act_region = np.concatenate((zeros, high_act_region), axis=-1)

            elif img_space_end > original_seq_j.shape[-1]:
                # Handle zero padding
                high_act_region = original_seq_j[:, img_space_start:]
                
                zeros = np.zeros((original_seq_j.shape[0], img_space_end - original_seq_j.shape[-1]))
                high_act_region = np.concatenate((high_act_region, zeros), axis=-1)
            else:
                high_act_region = original_seq_j[:, img_space_start:img_space_end]

            np.save(os.path.join(dir_for_saving_prototypes, 'prototype_{}_activations.npy'.format(j)),
                    proto_act_[seq_index_in_batch, j])
            # NOTE: The problem is that this is saved before push is executed -- we want post-push activations
            np.save(os.path.join(dir_for_saving_prototypes, 'prototype_{}_original.npy'.format(j)),
                    original_seq_j)
            np.save(os.path.join(dir_for_saving_prototypes, 'prototype_{}_patch.npy'.format(j)),
                    high_act_region)

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_max_proto_act, # this will be updated
                               global_max_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_seq_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_act_torch = prototype_network_parallel.push_forward(search_batch)

        x_norm = torch.norm(protoL_input_torch, p=2, dim=(1))
        x_reshape = x_norm.view(protoL_input_torch.shape[0], 1, protoL_input_torch.shape[-1])
        protoL_input_torch = (protoL_input_torch / x_reshape) / (prototype_network_parallel.prototype_shape[-1])**0.5

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_act_torch

    if class_specific:
        class_to_seq_index_dict = {key: [] for key in range(num_classes)}
        # seq_y is the image's integer label
        for seq_index, seq_y in enumerate(search_y):
            seq_label = seq_y.item()
            class_to_seq_index_dict[seq_label].append(seq_index)

    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    upsampling_factor = search_batch_input.shape[-1] // protoL_input_.shape[-1]
    # proto_w = prototype_shape[3]
    # max_act = prototype_shape[1] * prototype_shape[2]

    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_seq_index_dict[target_class]) == 0:
                continue
            proto_act_j = proto_act_[class_to_seq_index_dict[target_class]][:,j,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_act_j = proto_act_[:,j,:]

        batch_max_proto_act_j = np.amax(proto_act_j)
        if batch_max_proto_act_j > global_max_proto_act[j]:
            batch_argmax_proto_act_j = \
                list(np.unravel_index(np.argmax(proto_act_j, axis=None),
                                      proto_act_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmax_proto_act_j[0] = class_to_seq_index_dict[target_class][batch_argmax_proto_act_j[0]]

            # retrieve the corresponding feature map patch
            seq_index_in_batch = batch_argmax_proto_act_j[0]
            fmap_height_start_index = batch_argmax_proto_act_j[1] * prototype_layer_stride - proto_h // 2
            fmap_height_end_index = batch_argmax_proto_act_j[1] + proto_h // 2 + 1
            # fmap_width_start_index = batch_argmax_proto_act_j[2] * prototype_layer_stride
            # fmap_width_end_index = fmap_width_start_index + proto_w

            if fmap_height_start_index < 0:
                # Handle zero padding
                batch_max_fmap_patch_j = protoL_input_[seq_index_in_batch,
                                                    :,
                                                    0:fmap_height_end_index]
                                                    #    fmap_width_start_index:fmap_width_end_index]
                zeros = np.zeros((batch_max_fmap_patch_j.shape[0], -fmap_height_start_index))
                batch_max_fmap_patch_j = np.concatenate((zeros, batch_max_fmap_patch_j), axis=-1)

            elif fmap_height_end_index > protoL_input_.shape[-1]:
                # Handle zero padding
                batch_max_fmap_patch_j = protoL_input_[seq_index_in_batch,
                                                    :,
                                                    fmap_height_start_index:]
                                                    #    fmap_width_start_index:fmap_width_end_index]
                zeros = np.zeros((batch_max_fmap_patch_j.shape[0], fmap_height_end_index - protoL_input_.shape[-1]))
                batch_max_fmap_patch_j = np.concatenate((batch_max_fmap_patch_j, zeros), axis=-1)
            else:
                batch_max_fmap_patch_j = protoL_input_[seq_index_in_batch,
                                                    :,
                                                    fmap_height_start_index:fmap_height_end_index]
                                                    #    fmap_width_start_index:fmap_width_end_index]

            global_max_proto_act[j] = batch_max_proto_act_j
            global_max_fmap_patches[j] = batch_max_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            #protoL_rf_info = prototype_network_parallel.proto_layer_rf_info
            #rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmax_proto_act_j, protoL_rf_info)
            
            # get the whole image

            
            # crop out the receptive field
            '''rf_seq_j = original_seq_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]

            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 0] = search_y[rf_prototype_j[0]].item()'''

            # find the highly activated region of the original image
            #proto_act_seq_j = proto_act_[seq_index_in_batch, j, :]
            # if prototype_network_parallel.prototype_activation_function == 'log':
            #     proto_act_seq_j = np.log((proto_act_seq_j + 1) / (proto_act_seq_j + prototype_network_parallel.epsilon))
            # elif prototype_network_parallel.prototype_activation_function == 'linear':
            #     proto_act_seq_j = max_act - proto_act_seq_j
            # else:
            #     proto_act_seq_j = prototype_activation_function_in_numpy(proto_act_seq_j)
            '''upsampled_act_seq_j = cv2.resize(proto_act_seq_j, dsize=(original_seq_size, original_seq_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_seq_j)
            # crop out the image patch with high activation as prototype image
            proto_seq_j = original_seq_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]'''

            # save the prototype boundary (rectangular boundary of highly activated region)
            '''proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()'''

            '''if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_seq_j)
                if prototype_seq_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_seq_filename_prefix + '-original' + str(j) + '.png'),
                               original_seq_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_seq_j = upsampled_act_seq_j - np.amin(upsampled_act_seq_j)
                    rescaled_act_seq_j = rescaled_act_seq_j / np.amax(rescaled_act_seq_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_seq_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_seq_j = 0.5 * original_seq_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_seq_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_seq_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_seq_j.shape[0] != original_seq_size or rf_seq_j.shape[1] != original_seq_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_seq_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_seq_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_seq_j = overlayed_original_seq_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_seq_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_seq_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_seq_filename_prefix + str(j) + '.png'),
                               proto_seq_j,
                               vmin=0.0,
                               vmax=1.0)'''
                
    if class_specific:
        del class_to_seq_index_dict
