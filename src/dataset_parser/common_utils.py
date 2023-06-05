import numpy as np


def split_data(dataset_size, train_split, valid_split):
    """
    :param dataset_size: The total size of the dataset
    :param train_split: The total train split, remaining acts as test split
    :param valid_split: size of validation split wrt to train split -> 100*train_split*validation_split
    :return: indexes

    The splits are: dataset[:dev_index], dataset[dev_index:test_index], dataset[test_index:]
    """
    index = np.random.permutation(dataset_size)
    test_index = int(train_split * dataset_size)
    dev_index = int(train_split * dataset_size) - int(train_split * dataset_size * valid_split)

    return index, test_index, dev_index
