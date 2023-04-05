import csv
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import math

saved_sequences_prefix = './datasets/known_maine_species/'
labels_file = 'labels.npy'
labels_dict_file = 'labels_dict.npy'
sequence_file = 'sequences.npy'

class Sequence_Data(Dataset):
    def __init__(self,
                data_path='./datasets/split_datasets/Data Prep_ novel species - train d1_ maine dataset fishbase-mitofish.csv',
                target_sequence_length=150,
                sequence_count_threshold=4,
                transform=None,
                target_num_subseqs=50,
                shuffle=0,
                label_dict=None,
                is_test=False):
        # It's easier to work with the tsv version, since there are some commas
        # used in it that mess everything up for the csv conversion
        print("Using dataset: {}".format(data_path))
        print("sequence_count_threshold: {}".format(sequence_count_threshold))
        self.data_path = data_path
        # The length to restrict sequeneces to
        self.target_sequence_length = target_sequence_length
        self.target_num_subseqs = target_num_subseqs
        self.shuffle = shuffle
        self.is_test = is_test

        # This is important if train and test have non-overlapping classes --
        # use it to specify what the index for each class should be
        if label_dict == None:
            self.label_dict = {}
        else:
            self.label_dict = label_dict

        self.current_max_label = 0 # Index of the highest labeled class
        self.len = 0 # Num seqs in the dataset
        self.sequence_count_threshold = sequence_count_threshold # Min number of sequences for a class to be considered
        self.transform = transform # Transformation for online augmentation
        self.num_seqs_by_class = {} # Number of sequences found in each class

        with open(data_path) as csv_data:
            csv_file = list(csv.reader(csv_data, delimiter=','))

            prev_species = ''
            species_sequence_counter = 1

            # Calculating expected dataset length, num seqs per class
            for index, row in enumerate(csv_file):
                # Skip the header row
                if index == 0:
                    continue

                # If we find another sequence of the previous class
                if prev_species == row[0]:
                    species_sequence_counter += 1
                # If not
                elif prev_species != '' and (not self.is_test or prev_species in self.label_dict.keys()):
                    self.num_seqs_by_class[prev_species] = species_sequence_counter
                    # Subtract target_num_subseqs % species_sequence_counter because, if target_num_subseqs
                    # is not a nice multiple of the number of sequences for this class, we're going to 
                    # take the floor of target_num_subseqs / num_seqs_for_this_class subsequences
                    # We want to specifically handle the case where num_seqs > target_num_subseqs
                    if species_sequence_counter < target_num_subseqs:
                        self.len += target_num_subseqs - target_num_subseqs % species_sequence_counter
                    else:
                        self.len += species_sequence_counter
                    species_sequence_counter = 1
                    
                prev_species = row[0]

            if not self.is_test or prev_species in self.label_dict.keys():
                self.num_seqs_by_class[prev_species] = species_sequence_counter
                self.len += target_num_subseqs - target_num_subseqs % species_sequence_counter
                
        print("Num seqs per class: ", self.num_seqs_by_class)
        print("Len: ", self.len)
        print("Num classes: ", len(self.num_seqs_by_class))

        # Creating the actual dataset by reading through CSV
        with open(data_path) as csv_data:
            csv_file = list(csv.reader(csv_data, delimiter=','))

            # self.sequences is a tensor of shape num_sequences x 4 x sequence_length
            self.sequences = torch.zeros(self.len, 4, self.target_sequence_length)
            self.labels = torch.zeros(self.len)

            # Variables used for tracking how many sequences of the current species we've loaded
            prev_species = ''
            species_sequence_counter = 1 # Number of overall seqs seen for prev_species
            chunks_of_cur_species = 0 # Number of subsequences seen for prev_species
            drop_species_offset = 0 # Number of subsequences that have been dropped so far

            #shuffling
            if self.shuffle==1:
                # TODO: Double check
                assert False, "Shuffle may not be implemented correctly -- test more"
                col1_labels, col2_seqs, col3, col4 = map(list, zip(*csv_file))
                np.random.shuffle(col1_labels[1:])
                csv_file = [list(row) for row in zip(col1_labels, col2_seqs, col3, col4)]

            # Iterating over each sequence
            chunks_so_far = 0 # The total number of subsequences we've considered
            for index, row in enumerate(csv_file):
                # Skip header row
                if index == 0:
                    header_row = row
                else:

                    '''
                    Handling the removal of species from the dataset; on each new
                    sequence in the dataset, we're going to check whether we're looking
                    at a different species than on the previous iteration (assumes dataset
                    is sorted by species). If we are, we will note that we've seen another
                    sequence from the current species. If not, we check whether we've seen
                    enough sequences from the current species.'''
                    if prev_species == row[0]:
                        species_sequence_counter += 1
                    elif prev_species != '':
                        '''
                        If we haven't seen a sufficient number of sequences from the current species,
                        we will wipe out all subsequences written for it. We well then increment the
                        drop_species_offset by the number of subsequences loaded from the dropped species,
                        which will push the "pointer" to where our next subsequence will be written back
                        to overwrite the first subsequence of the current class.
                        '''
                        # We now only apply the threshold on the train set
                        if (prev_species not in self.label_dict.keys() and self.is_test) \
                            or (not self.is_test and species_sequence_counter < self.sequence_count_threshold):
                            for subtractor in range(chunks_of_cur_species):
                                self.sequences[chunks_so_far-subtractor-drop_species_offset] = 0
                            drop_species_offset+=chunks_of_cur_species
                            # NOTE: Don't need to decrement this because it's okay if we have some labels
                            # that are dead; the model just never predicts them
                            #self.current_max_label -= 1
                        species_sequence_counter = 1
                        chunks_of_cur_species = 0

                    # Update the label dictionary to include this class if it doesn't
                    if row[0] not in self.label_dict.keys() and not self.is_test:
                        self.label_dict[row[0]] = self.current_max_label
                        self.current_max_label += 1

                    if row[0] not in self.num_seqs_by_class.keys():
                        num_chunks = 0
                    elif self.num_seqs_by_class[row[0]] > target_num_subseqs:
                        num_chunks = 1
                    else:
                        # The number of subsequences added for the current sequence
                        num_chunks = math.floor(target_num_subseqs / self.num_seqs_by_class[row[0]])

                    # Iterate and write each possible subsequence of the current sequence
                    for in_seq_chunk_index in range(num_chunks):
                        if row[0] in self.label_dict.keys():
                            self.labels[chunks_so_far-drop_species_offset] = int(self.label_dict[row[0]])
                        else:
                            continue

                        # If the next subsequence is below our desired length, pad it with 0's
                        # Note that this is done automatically by not allocating the whole row,
                        # since dataset initializes to 0
                        if len(row[1]) - in_seq_chunk_index < self.target_sequence_length:
                            self.sequences[chunks_so_far-drop_species_offset, :, :len(row[1]) - in_seq_chunk_index] = torch.Tensor(self.sequence_to_array(row[1])[:, in_seq_chunk_index:])
                        else:
                            self.sequences[chunks_so_far-drop_species_offset] = torch.Tensor(self.sequence_to_array(row[1])[:, in_seq_chunk_index:self.target_sequence_length + in_seq_chunk_index])
                        chunks_so_far += 1
                        chunks_of_cur_species += 1

                    prev_species = row[0]

        # Compensate for any sequences that were allocated then dropped
        print("Final drop species offset: {}".format(drop_species_offset))
        self.len -= drop_species_offset
        print("num_seqs_by_class: ")
        print(self.num_seqs_by_class)
        print("Total dataset size is %d" % (self.len))

    def __getitem__(self,idx):
        item = self.sequences[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            item = self.transform(item)
        return item, label
  
    def __len__(self):
        return self.len
        
    '''
    Converts an array of DNA bases to a 4 channel numpy array, 
    with A -> channel 0, T -> channel 1, C -> channel 2, and G -> channel 3,
    assigning fractional values for ambiguity codes. A full list of these 
    codes can be found at https://droog.gs.washington.edu/mdecode/images/iupac.html 
    Input: String of bases
    Output: 4 * str_len array of integers
    '''
    def sequence_to_array(self, bp_sequence):
        output_array = torch.zeros(4, len(bp_sequence))

        for i, letter in enumerate(bp_sequence):
            # Actual bases
            if letter == 'A':
                output_array[0, i] = 1
            elif letter == 'T':
                output_array[1, i] = 1
            elif letter == 'C':
                output_array[2, i] = 1
            elif letter == 'G':
                output_array[3, i] = 1
            # Uncertainty codes
            elif letter == 'M':
                # A or C
                if np.random.rand() < 0.5:
                    output_array[0, i] = 1
                else:
                    output_array[2, i] = 1
            elif letter == 'R':
                # A or G
                if np.random.rand() < 0.5:
                    output_array[0, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'W':
                # A or T
                if np.random.rand() < 0.5:
                    output_array[0, i] = 1
                else:
                    output_array[1, i] = 1
            elif letter == 'S':
                # C or G
                if np.random.rand() < 0.5:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'Y':
                # C or T
                if np.random.rand() < 0.5:
                    output_array[1, i] = 1
                else:
                    output_array[2, i] = 1
            elif letter == 'K':
                # G or T
                if np.random.rand() < 0.5:
                    output_array[1, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'V':
                # A or C or G
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[0, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'H':
                # A or C or T
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[0, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[1, i] = 1
                else:
                    output_array[2, i] = 1
            elif letter == 'D':
                # A or G or T
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[0, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[1, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'B':
                # C or G or T
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[1, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'N':
                # N indicates complete uncertainty
                rand_num = np.random.rand()
                if rand_num < 1/4:
                    output_array[0, i] = 1
                elif 1/4 < rand_num and rand_num < 2/4:
                    output_array[1, i] = 1
                elif 2/4 < rand_num and rand_num < 3/4:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            else:
                print("ERROR: Unknown base '{}' encountered at index {} in {}".format(letter, i, bp_sequence))

        return output_array

# Function taken from https://stackoverflow.com/questions/2460177/edit-distance-in-python
def levenshteinDistance(s1, s2):
    s1 = s1[:, torch.sum(s1, dim=0) != 0]
    s2 = s2[:, torch.sum(s1, dim=0) != 0]
    if s1.shape[-1] > s2.shape[-1]:
        s1, s2 = s2, s1

    distances = range(s1.shape[-1] + 1)
    for i2, c2 in enumerate(s2.T):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1.T):
            # If (c1 != c2).any() evaluates to True, one val doesn't match
            if not (c1 != c2).any():
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

if __name__ == '__main__':
    dataset = Sequence_Data(target_num_subseqs=10, sequence_count_threshold=0) #dataloader
    from torch.utils.data import DataLoader 
    train_loader = DataLoader(dataset, shuffle=False, batch_size=1)
    distances = torch.zeros(dataset.len, dataset.len)
    for index, (item, _) in enumerate(train_loader):
        print(index, item)
        for index_2, (item_2, _) in enumerate(train_loader):
            distances[index, index_2] = levenshteinDistance(item[0], item_2[0])

    print("Average distance: {}".torch.mean(distances))
    print("Std deviation in distance: {}".torch.std(distances))
    print(distances)
    torch.save(distances, "8_30_pairwise_l1_distances.pt")