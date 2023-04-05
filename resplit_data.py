import csv

data_path = './datasets/split_datasets/Data Prep_ novel species - main full dataset.csv'
train_path = './datasets/split_datasets/main_train_split_80.csv'
test_path = './datasets/split_datasets/main_test_split_20.csv'
split_ratio = 4/5

with open(train_path, mode='r+') as train_data:
    with open(test_path, mode='r+') as test_data:
        with open(data_path) as csv_data:
            train_file = csv.writer(train_data, delimiter=',')
            test_file = csv.writer(test_data, delimiter=',')
            csv_file = list(csv.reader(csv_data, delimiter=','))

            prev_species = ''
            species_sequence_counter = 1
            num_seqs_by_class =  {}

            # Calculating expected dataset length, num seqs per class
            for index, row in enumerate(csv_file):
                # Skip the header row
                if index == 0:
                    continue

                # If we find another sequence of the previous class
                if prev_species == row[0]:
                    species_sequence_counter += 1
                # If not
                elif prev_species != '':
                    num_seqs_by_class[prev_species] = species_sequence_counter
                    # Subtract target_num_subseqs % species_sequence_counter because, if target_num_subseqs
                    # is not a nice multiple of the number of sequences for this class, we're going to 
                    # take the floor of target_num_subseqs / num_seqs_for_this_class subsequences
                    species_sequence_counter = 1
                    
                prev_species = row[0]

            num_seqs_by_class[prev_species] = species_sequence_counter

            # Variables used for tracking how many sequences of the current species we've loaded
            species_sequence_counter = 1 # Number of overall seqs seen for prev_species
            prev_species = ''
            for index, row in enumerate(csv_file):
                if index == 0:
                    train_file.writerow(row)
                    test_file.writerow(row)
                    continue

                if row[0] != prev_species:
                    species_sequence_counter = 1
                    prev_species = row[0]
                else:
                    species_sequence_counter += 1

                if species_sequence_counter <= split_ratio * num_seqs_by_class[row[0]]:
                    # Write to train CSV
                    train_file.writerow(row)
                else:
                    # Write to test CSV
                    test_file.writerow(row)

