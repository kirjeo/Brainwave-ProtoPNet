import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from dataset import Sequence_Data  
from dataset_rewrite import Sequence_Data_Alt  
from model import My_CNN
from augment import RandomBPFlip

def evaluate(model, train_dataset, test_dataset):
    # The number of training epochs
    num_train_epochs = 250

    train_batch_size = 1300
    test_batch_size = 800
    use_weights = True
    print("Batch size: {}".format(train_batch_size))
    
    # Much of this script is taken from 
    # https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=train_batch_size,
                    shuffle=True)
    testloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=test_batch_size,
                    shuffle=True)


    if use_weights:
        # Trying out weighted cross entropy as an alternative to balancing the dataset
        count_by_class = train_dataset.get_count_by_class()
        print("Found count by class: ")
        print(count_by_class)

        # We want each class to act like it has 1/c samples, so we want k s.t.
        # 1/c = k * actual_count / n => n / (actual_count * c) = k
        weights = train_dataset.num_samples / (count_by_class * train_dataset.next_class_label)
        weights = weights / torch.sum(weights)
        print("Using weighted cross entropy with weights: ")
        print(weights)

        loss_function = nn.CrossEntropyLoss(weight=weights.cuda())
        L2_pen = 1e-3 # Lambda to multiply L2 penalty over weights by
    else:
        loss_function = nn.CrossEntropyLoss()
        L2_pen = 0.1 # Lambda to multiply L2 penalty over weights by

    learning_rate = 1e-3
    print("Adding weight decay with val {}".format(L2_pen))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_pen)
    print("Learning rate: {}".format(learning_rate))
    print("Training batch size {}".format(train_batch_size))

    for epoch in range(num_train_epochs):
        print(f'Starting epoch {epoch}')
        loss = 0
        correct, total = 0, 0
        
        # Iterate over the entire training set
        for i, data in enumerate(trainloader, start=0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            # Compute loss
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                

        print("Epoch {} ----".format(epoch))
        print('\tTrain Accuracy: {}%%'.format(100.0 * correct / total))
        print('\tLoss: {}'.format(loss.item()))

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():

            y_pred = []
            y_true = []
            # Iterate over the test data and generate predictions
            for i, (inputs, labels) in enumerate(testloader, 0):
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                y_pred.extend(outputs) # Save Prediction
                
                labels = labels.data.cpu().numpy()
                y_true.extend(labels)

            # Print accuracy
            print("Num correct: ", correct)
            print("Num total: ", total)
            print('\tTest Accuracy: {} %%'.format(100.0 * correct / total))

    torch.save(model.state_dict(), 'base_model_weights.pth')
    # classes = list(set(labels))

    # Build confusion matrix
    # cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
    #                     columns = [i for i in classes])
    # plt.figure(figsize = (12,7))
    # sn.heatmap(df_cm, annot=True)
    # plt.savefig('output.png')

    # print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    '''
    Loading in the data; the dataloader is set up
    to work with pytorch, so this is a little bit
    of a hacky way to use it, but it does the trick
    '''
    print('Kernel sizes 13, 13, 13')
    for target_num_subseqs in [1]:
        for num_flips in [2]:
            target_sequence_length = 200 # The amount of base pairs in each sequence
            sequence_count_threshold = 0
            num_flips = num_flips # How many base pairs to randomly flip in each training sequence

            print("Training threshold = 4")
            print("Testing threshold = 1")

            train_dataset = Sequence_Data_Alt(data_path='./datasets/split_datasets/main_train_split_80_threshold_8.csv',
                target_sequence_length=target_sequence_length)
            '''transform=RandomBPFlip(num_flips),
            sequence_count_threshold=sequence_count_threshold,
            target_num_subseqs=target_num_subseqs,
            shuffle=False,
            is_test=False)'''

            test_dataset = Sequence_Data_Alt(data_path='./datasets/split_datasets/main_test_split_20_for_threshold_8.csv',
                target_sequence_length=target_sequence_length, predefined_label_dict=train_dataset.label_dict)#,
            '''sequence_count_threshold=sequence_count_threshold,
            target_num_subseqs=1,
            shuffle=False,
            is_test=True)'''

            # This is okay because train dataset always has >= num classes in test
            n_classes = train_dataset.next_class_label

            for conv1_out_channels in [4, 64, 128]:
                model = My_CNN(conv1_out_channels=conv1_out_channels, conv1_kernel_size=13,
                            conv2_out_channels=conv1_out_channels*2, conv2_kernel_size=13,
                            conv3_out_channels=conv1_out_channels*4, conv3_kernel_size=13,
                            n_classes=n_classes, in_len=target_sequence_length).cuda()
                evaluate(model, train_dataset, test_dataset)


    