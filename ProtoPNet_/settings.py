base_architecture = 'base_feature'

prototype_activation_function = 'log'
add_on_layers_type = 'identity'

experiment_run = 'thresh_4_684_protos'

data_path = '../datasets/split_datasets/'
train_dir = data_path + 'main_train_split_80_threshold_4_aug_sp_0.08.csv'
test_dir = data_path + 'main_test_split_20_for_threshold_4.csv'
train_push_dir = data_path + 'main_train_split_80_threshold_4.csv'
train_batch_size = 2000
test_batch_size = 800
train_push_batch_size = 1500

sequence_length = 888
prototype_shape = (100, 512+1, 5)
num_classes = 10 #36

joint_optimizer_lrs = {'features': 3*1e-3,
                       'add_on_layers': 3*3e-3,
                       'prototype_vectors': 3*3e-3}
joint_lr_step_size = 50

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 4e-2}

last_layer_optimizer_lr = 2e-2
latent_weight = 1

coefs = {
    'crs_ent': 1,
    'clst': 1*12*-0.8,
    'sep': 1*30*0.08,
    'l1': 1e-3,
}

use_aug = False
flip_count = 5
del_prob = 0.9

num_train_epochs = 500
num_warm_epochs = 10

push_start = 200
push_epochs = [i for i in range(num_train_epochs) if i % 50 == 0]
