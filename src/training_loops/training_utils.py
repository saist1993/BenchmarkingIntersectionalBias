import torch
import random
import numpy as np
from typing import Optional, List
from misc_utils import generate_mask
from arguments import ParsedDataset, TrainingLoopParameters
from utils.iterator import sequential_transforms, TextClassificationDataset


def get_iterators(training_loop_parameters: TrainingLoopParameters, parsed_dataset: ParsedDataset):
    if training_loop_parameters.iterator_type == "simple_iterator":
        csi = CreateSimpleIterators(
            parsed_dataset=parsed_dataset,
            batch_size=training_loop_parameters.batch_size,
            per_group_size=training_loop_parameters.per_group_size)
        original_train_iterator, train_iterator, valid_iterator, test_iterator = csi.get_iterators()
    elif training_loop_parameters.iterator_type == "group_iterator":
        csi = CreateSimpleIterators(
            parsed_dataset=parsed_dataset,
            batch_size=training_loop_parameters.batch_size,
            per_group_size=-1)
        original_train_iterator, _, valid_iterator, test_iterator = csi

        if training_loop_parameters.fairness_function in ["demographic_parity"]:
            iterator_type = "sample_data_without_y"
        elif training_loop_parameters.fairness_function in ["equal_odds", "equal_opportunity"]:
            iterator_type = "sample_data_with_y"
        else:
            raise NotImplementedError

        gi = GroupIterators(parsed_dataset=parsed_dataset, batch_size=training_loop_parameters.batch_size,
                            iterator_type=iterator_type, shuffle=True)  # set shuffle False for MixUp Regularizer
        train_iterator = gi.get_iterators()

    return original_train_iterator, train_iterator, valid_iterator, test_iterator


def resample_dataset(all_X, all_y, all_s, per_group_size):
    size_of_each_group = {}
    all_unique_s = np.unique(all_s, axis=0)
    all_unique_label = np.unique(all_y)

    for s in all_unique_s:
        size_of_each_group[tuple(s.tolist())] = {}
        for label in all_unique_label:
            size_of_each_group[tuple(s.tolist())][label] = [
                per_group_size,
                []]

    resampled_X, resampled_s, resampled_y, size = [], [], [], {}

    for s in all_unique_s:
        for label in all_unique_label:
            # sample number_of_examples_per_group with s and label from the dataset
            mask_s = generate_mask(all_s=all_s, mask_pattern=s)
            mask_label = all_y == label
            final_mask = np.logical_and(mask_s, mask_label)
            index = np.where(final_mask)[0]

            # index are all the possible examples one can use.
            group_size, all_members_of_the_group = size_of_each_group[tuple(s.tolist())][label]

            if len(all_members_of_the_group) > group_size:
                k = len(all_members_of_the_group) - group_size
                all_members_of_the_group = all_members_of_the_group[k:]
                size_of_each_group[tuple(s.tolist())][label] = [group_size, all_members_of_the_group]
            else:
                # need to add examples
                k = group_size - len(all_members_of_the_group)
                all_members_of_the_group.extend(np.random.choice(index,
                                                                 size=k,
                                                                 replace=True))

                size_of_each_group[tuple(s.tolist())][label] = [group_size, all_members_of_the_group]

            index_of_selected_examples = np.random.choice(all_members_of_the_group,
                                                          size=size_of_each_group[tuple(s.tolist())][label][0],
                                                          replace=True)

            resampled_X.append(all_X[index_of_selected_examples])
            resampled_s.append(all_s[index_of_selected_examples])
            resampled_y.append(all_y[index_of_selected_examples])

    resampled_X = np.vstack(resampled_X)
    resampled_s = np.vstack(resampled_s)
    resampled_y = np.hstack(resampled_y)

    return resampled_X, resampled_y, resampled_s, size_of_each_group


class CreateSimpleIterators:
    """ A general purpose iterators. Takes numpy matrix for train, dev, and test matrix and creates iterator."""

    def __init__(self, parsed_dataset: ParsedDataset, batch_size, per_group_size: int = -1):
        """

        :param parsed_dataset:
        :param batch_size:
        :param per_group_size: -1 here means take all the dataset and not do equal sampling!
        """
        self.parsed_dataset = parsed_dataset
        self.batch_size = batch_size
        self.per_group_size = per_group_size  # @TODO: This still needs to be implemented!

    def collate(self, batch):
        labels, encoded_input, aux = zip(*batch)
        labels = torch.LongTensor(labels)
        aux_flattened = torch.LongTensor(self.parsed_dataset.s_list_to_int[tuple(aux)])
        aux = torch.LongTensor(np.asarray(aux))
        lengths = torch.LongTensor([len(x) for x in encoded_input])
        encoded_input = torch.FloatTensor(np.asarray(encoded_input))

        input_data = {
            'labels': labels,
            'input': encoded_input,
            'lengths': lengths,
            'aux': aux,
            'aux_flattened': aux_flattened
        }

        return input_data

    def process_data(self, x, y, s, vocab):
        """raw data is assumed to be tokenized"""

        final_data = [(a, b, c) for a, b, c in zip(y, x, s)]

        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)

    def get_iterators(self):
        vocab = {'<pad>': 1}  # no need of vocab in these dataset. It is there for code compatibility purposes.

        # need to add flatten here! And that too in the process itself!
        train_data = self.process_data(self.parsed_dataset.train_X,
                                       self.parsed_dataset.train_y,
                                       self.parsed_dataset.train_s,
                                       vocab=vocab)

        original_train_iterator = torch.utils.data.DataLoader(train_data,
                                                              self.batch_size,
                                                              shuffle=True,
                                                              collate_fn=self.collate
                                                              )

        # resample X,Y, and S here
        if self.per_group_size != -1:
            train_X, train_y, train_s, _ = resample_dataset(all_X=self.parsed_dataset.train_X,
                                                            all_s=self.parsed_dataset.train_s,
                                                            all_y=self.parsed_dataset.train_y,
                                                            per_group_size=self.per_group_size)

            train_data = self.process_data(train_X,
                                           train_y,
                                           train_s,
                                           vocab=vocab)

            resampled_train_iterator = torch.utils.data.DataLoader(train_data,
                                                                   self.batch_size,
                                                                   shuffle=True,
                                                                   collate_fn=self.collate
                                                                   )

        else:
            resampled_train_iterator = original_train_iterator

        valid_data = self.process_data(self.parsed_dataset.valid_X,
                                       self.parsed_dataset.valid_y,
                                       self.parsed_dataset.valid_s,
                                       vocab=vocab)

        valid_iterator = torch.utils.data.DataLoader(valid_data,
                                                     self.batch_size,
                                                     shuffle=True,
                                                     collate_fn=self.collate
                                                     )

        test_data = self.process_data(self.parsed_dataset.test_X,
                                      self.parsed_dataset.test_y,
                                      self.parsed_dataset.test_s,
                                      vocab=vocab)

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    self.batch_size,
                                                    shuffle=True,
                                                    collate_fn=self.collate
                                                    )

        return original_train_iterator, resampled_train_iterator, valid_iterator, test_iterator


class GroupIterators:

    def __init__(self, parsed_dataset: ParsedDataset, batch_size: int, iterator_type: str, shuffle: bool):
        self.parsed_dataset = parsed_dataset
        self.batch_size = batch_size
        self.iterator_type = iterator_type
        self.shuffle = shuffle

    def common_procedure(self, mask, size):
        relevant_index = []
        for m in mask:
            relevant_index += np.random.choice(np.where(m)[0],
                                               size=size,
                                               replace=True).tolist()

        if self.shuffle:
            random.shuffle(relevant_index)

        relevant_aux = torch.LongTensor(self.parsed_dataset.train_s[relevant_index])
        relevant_aux_flatten = [self.parsed_dataset.s_list_to_int[tuple(i)] for i in relevant_aux]

        batch_input = {
            'labels': torch.LongTensor(self.parsed_dataset.train_y[relevant_index]),
            'input': torch.FloatTensor(self.parsed_dataset.train_X[relevant_index]),
            'aux': torch.LongTensor(relevant_aux),
            'aux_flattened': torch.LongTensor(relevant_aux_flatten)
        }

        return batch_input

    def sample_data_with_y(self, group: Optional[List] = None):
        if not group:
            group = random.choice(self.parsed_dataset.all_groups)

        mask = generate_mask(self.parsed_dataset.train_s, group)
        positive_mask = np.logical_and(mask, self.parsed_dataset.test_y == 1)
        negative_mask = np.logical_and(mask, self.parsed_dataset.test_y == 0)

        return self.common_procedure(mask=[positive_mask, negative_mask], size=self.batch_size / 2)

    def sample_data_without_y(self, group: Optional[List] = None):
        if not group:
            group = random.choice(self.parsed_dataset.all_groups)

        mask = generate_mask(self.parsed_dataset.train_s, group)
        return self.common_procedure(mask=[mask], size=self.batch_size)

    def get_iterators(self):
        if self.iterator_type == "sample_data_with_y":
            return self.sample_data_with_y
        elif self.iterator_type == "sameple_data_without_y":
            return self.sample_data_without_y
        else:
            raise NotImplementedError

