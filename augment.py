import torch
import numpy as np

class RandomBPFlip(object):
    def __init__(self, num_flips):
        self.num_flips = num_flips # Num base pairs to flip for a given sequence

    '''
    Input: A tensor representing an RNA sequence, of shape [batch, 4, len]
    Output: A tensor with num_flips base pairs randomly flipped
    '''
    def __call__(self, sequence):
        original_seq = sequence.clone()
        sequence = sequence.clone()
        using_batch = False
        if len(sequence.shape) > 2:
            using_batch = True

        # Checking to make sure we don't attempt a flip in the zero padding
        flip_max_indices = (torch.sum(sequence, dim=-2) == 0).nonzero()
        if flip_max_indices.shape[0] == 0:
            flip_max_index = sequence.shape[-1]
        else:
            flip_max_index = flip_max_indices[0].item()

        if flip_max_index == 0:
            return sequence
        elif using_batch:
            print("WARNING: using batch mode, which is less tested")
            # Generating num_flips random indices for each sequence in the batch
            flip_indices = torch.randint(0, flip_max_index, (sequence.shape[0], self.num_flips), ).view(-1)
            batch_indices = torch.repeat_interleave(torch.arange(sequence.shape[0]), self.num_flips, dim=0)

            # Generate sufficient random onehot base pairs to plug in
            random_bps = torch.zeros([sequence.shape[0] * self.num_flips, 4])
            bp_indices = torch.randint(0, 4, (sequence.shape[0] * self.num_flips,)).view(-1, 1)
            random_bps[torch.arange(sequence.shape[0] * self.num_flips).view(-1, 1), bp_indices] = 1

            # Indexing into the random indices chosen for each batch
            sequence[batch_indices.view(-1, 1), :, flip_indices.view(-1, 1)] = random_bps
        else:
            # Generating num_flips random indices for each sequence in the batch
            flip_indices = torch.randint(low=0, high=flip_max_index, size=(self.num_flips,)).view(-1)

            # Generate sufficient random onehot base pairs to plug in
            random_bps = torch.zeros([self.num_flips, 4]) # [num_flips, 4] tensor
            bp_indices = torch.randint(low=0, high=4, size=(self.num_flips,)).view(-1, 1) # [num_flips, 1] tensor

            # For each flip, assign 1 to a random base pair
            '''
            Random bps should look something like
            0 0 1 0
            1 0 0 0
            '''
            random_bps[torch.arange(self.num_flips).view(-1, 1), bp_indices] = 1

            # Indexing into the random indices chosen for each batch
            '''
            Transpose should turn random bps into something like
            0 1
            0 0
            1 0
            0 0
            '''
            sequence[:, flip_indices.view(-1, 1)] = random_bps.transpose(0, 1).view(4, -1, 1)
        return sequence
        

class RandomBPDel(object):
    def __init__(self, del_prob=0.5):
        self.del_prob = del_prob # Num base pairs to flip for a given sequence

    '''
    Input: A tensor representing an RNA sequence, of shape [batch, 4, len]
    Output: A tensor with num_flips base pairs randomly flipped
    '''
    def __call__(self, sequence):
        if np.random.rand() > self.del_prob:
            return sequence

        original_seq = sequence.clone()
        sequence = sequence.clone()
        using_batch = False
        if len(sequence.shape) > 2:
            using_batch = True

        # Checking to make sure we don't attempt a flip in the zero padding
        del_max_indices = (torch.sum(sequence, dim=-2) == 0).nonzero()
        if del_max_indices.shape[0] == 0:
            del_max_index = sequence.shape[-1]
        else:
            del_max_index = del_max_indices[0].item()

        if del_max_index <= 0:
            return sequence
        del_index = torch.randint(low=0, high=del_max_index, size=(1,))

        sequence = torch.concat([sequence[:, :del_index],
                                sequence[:, min(del_index + 1, sequence.shape[-1]):],
                                torch.zeros(4, 1)], dim=-1)
        return sequence