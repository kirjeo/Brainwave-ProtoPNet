import csv

original_train_path = './datasets/split_datasets/main_train_split_80.csv'
original_test_path = './datasets/split_datasets/main_test_split_20.csv'
train_threshold = 4
test_threshold = 1
new_train_path = './datasets/split_datasets/main_train_split_80_threshold_{}.csv'.format(train_threshold)
new_test_path = './datasets/split_datasets/main_test_split_20_threshold_{}.csv'.format(test_threshold)

with open(original_train_path, mode='r+') as og_train_data:
    with open(new_train_path, mode='r+') as new_train_data:
        og_reader = csv.reader(og_train_data, delimiter=',')
        new_writer = csv.writer(new_train_data, delimiter=',')
        sequence_count_by_class = {}

        for i, row in enumerate(og_reader):
            if i == 0:
                continue

            if row[0] not in sequence_count_by_class.keys():
                sequence_count_by_class[row[0]] = 1
            else:
                sequence_count_by_class[row[0]] += 1

        # Reset to the start of the file
        og_train_data.seek(0)
        for i, row in enumerate(og_reader):
            if i == 0:
                new_writer.writerow(row)
                continue

            if sequence_count_by_class[row[0]] >= train_threshold:
                new_writer.writerow(row)


with open(original_test_path, mode='r+') as og_test_data:
    with open(new_test_path, mode='r+') as new_test_data:
        og_reader = csv.reader(og_test_data, delimiter=',')
        new_writer = csv.writer(new_test_data, delimiter=',')
        sequence_count_by_class = {}

        for i, row in enumerate(og_reader):
            if i == 0:
                continue

            if row[0] not in sequence_count_by_class.keys():
                sequence_count_by_class[row[0]] = 1
            else:
                sequence_count_by_class[row[0]] += 1

        # Reset to the start of the file
        og_test_data.seek(0)
        for i, row in enumerate(og_reader):
            if i == 0:
                new_writer.writerow(row)
                continue

            if sequence_count_by_class[row[0]] >= test_threshold:
                new_writer.writerow(row)
