import copy
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Union
from arguments import ParsedDataset
from training_loops import training_utils
from misc_utils import generate_abstract_node


class GenerateData:
    def __init__(self, parsed_dataset: ParsedDataset, positive_gen_model: Path, negative_gen_model: Path,
                 size_of_each_group: Optional[Union[dict, int]] = None):
        self.parsed_dataset = copy.deepcopy(parsed_dataset)
        self.positive_gen_model = pickle.load(open(positive_gen_model, "rb"))
        self.negative_gen_model = pickle.load(open(negative_gen_model, "rb"))
        self.size_of_each_group = size_of_each_group

    def gen_data(self, group, size, label="positive"):
        # find all the first level abstract groups
        abstract_groups = generate_abstract_node(s=group, k=1)
        input_to_gen_model = []
        for abstract_group in abstract_groups:
            mask = training_utils.generate_mask(self.parsed_dataset.train_s, abstract_group)
            if label == "positive":
                final_mask = np.logical_and(mask, self.parsed_dataset.train_y == 1)
            elif label == "negative":
                final_mask = np.logical_and(mask, self.parsed_dataset.train_y == 0)
            else:
                raise NotImplementedError

            abstract_group_examples_mask = np.random.choice(np.where(final_mask == True)[0], size=size,
                                                            replace=True)
            abstract_group_examples = {
                'input': torch.FloatTensor(self.parsed_dataset.train_X[abstract_group_examples_mask]),
            }
            input_to_gen_model.append(abstract_group_examples)
        if label == "positive":
            generated_train_X = self.positive_gen_model(input_to_gen_model)['prediction'].detach().numpy()
            index_of_selected_examples_label = np.random.choice(np.where(self.parsed_dataset.train_y == 1)[0],
                                                                size=size,
                                                                replace=True)
        elif label == "negative":
            generated_train_X = self.negative_gen_model(input_to_gen_model)['prediction'].detach().numpy()
            index_of_selected_examples_label = np.random.choice(np.where(self.parsed_dataset.train_y == 0)[0],
                                                                size=size,
                                                                replace=True)
        else:
            raise NotImplementedError

        mask = training_utils.generate_mask(self.parsed_dataset.train_s, group)

        index_of_selected_examples_group = np.random.choice(np.where(mask == True)[0],
                                                            size=size,
                                                            replace=True)

        return generated_train_X, self.parsed_dataset.train_y[index_of_selected_examples_label], \
            self.parsed_dataset.train_s[index_of_selected_examples_group]

    def gen_data_via_noise(self, group, size, label="positive"):
        # find all the first level abstract groups

        if label == "positive":
            mask = training_utils.generate_mask(self.parsed_dataset.train_s, group)
            final_mask = np.logical_and(mask, self.parsed_dataset.train_y == 1)
        elif label == "negative":
            mask = training_utils.generate_mask(self.parsed_dataset.train_s, group)
            final_mask = np.logical_and(mask, self.parsed_dataset.train_y == 0)

        abstract_groups = generate_abstract_node(s=group, k=1)
        input_to_gen_model = []
        for abstract_group in abstract_groups:
            mask = training_utils.generate_mask(self.parsed_dataset.train_s, abstract_group)
            if label == "positive":
                final_mask = np.logical_and(mask, self.parsed_dataset.train_y == 1)
            elif label == "negative":
                final_mask = np.logical_and(mask, self.parsed_dataset.train_y == 0)
            else:
                raise NotImplementedError

            abstract_group_examples_mask = np.random.choice(np.where(final_mask == True)[0], size=size,
                                                            replace=True)
            abstract_group_examples = {
                'input': torch.FloatTensor(self.parsed_dataset.train_X[abstract_group_examples_mask]),
            }
            input_to_gen_model.append(abstract_group_examples)
        if label == "positive":
            generated_train_X = self.positive_gen_model(input_to_gen_model)['prediction'].detach().numpy()
            index_of_selected_examples_label = np.random.choice(np.where(self.parsed_dataset.train_y == 1)[0],
                                                                size=size,
                                                                replace=True)
        elif label == "negative":
            generated_train_X = self.negative_gen_model(input_to_gen_model)['prediction'].detach().numpy()
            index_of_selected_examples_label = np.random.choice(np.where(self.parsed_dataset.train_y == 0)[0],
                                                                size=size,
                                                                replace=True)
        else:
            raise NotImplementedError

        mask = training_utils.generate_mask(self.parsed_dataset.train_s, group)

        index_of_selected_examples_group = np.random.choice(np.where(mask == True)[0],
                                                            size=size,
                                                            replace=True)

        return generated_train_X, self.parsed_dataset.train_y[index_of_selected_examples_label], \
            self.parsed_dataset.train_s[index_of_selected_examples_group]

    def run(self):
        # if self.size_of_each_group == None:
        #     self.size_of_each_group = {}
        #     for group in self.parsed_dataset.all_groups:
        #         mask = training_utils.generate_mask(self.parsed_dataset.train_s, group)
        #         positive_mask = np.logical_and(mask, self.parsed_dataset.train_y == 1)
        #         negative_mask = np.logical_and(mask, self.parsed_dataset.train_y == 0)
        #         self.size_of_each_group[tuple(group)] = {
        #             1: np.sum(positive_mask),
        #             0: np.sum(negative_mask)
        #         }
        #
        # if type(self.size_of_each_group) == int:
        #     group_size = copy.deepcopy(self.size_of_each_group)
        #     self.size_of_each_group = {}
        #     for group in self.parsed_dataset.all_groups:
        #         self.size_of_each_group[tuple(group)] = {
        #             1: group_size,
        #             0: group_size
        #         }

        group_size = copy.deepcopy(self.size_of_each_group)
        self.size_of_each_group = {}
        for group in self.parsed_dataset.all_groups:
            mask = training_utils.generate_mask(self.parsed_dataset.train_s, group)
            positive_mask = np.logical_and(mask, self.parsed_dataset.train_y == 1)
            negative_mask = np.logical_and(mask, self.parsed_dataset.train_y == 0)
            # self.size_of_each_group[tuple(group)] = {
            #     1: max(0, group_size - np.sum(positive_mask)),
            #     0: max(0, group_size - np.sum(negative_mask))
            # }

            self.size_of_each_group[tuple(group)] = {
                1: group_size,
                0: group_size
            }

        all_X, all_s, all_y = [], [], []

        for group, group_size in self.size_of_each_group.items():
            positive_size = group_size[1]
            negative_size = group_size[0]
            if positive_size > 0:
                x, y, s = self.gen_data(group=group, size=positive_size, label="positive")
                all_X.append(x)
                all_y.append(y)
                all_s.append(s)
            if negative_size > 0:
                x, y, s = self.gen_data(group=group, size=negative_size, label="negative")
                all_X.append(x)
                all_y.append(y)
                all_s.append(s)

        all_X = np.vstack(all_X)
        all_s = np.vstack(all_s)
        all_y = np.hstack(all_y)

        self.parsed_dataset.train_X = np.vstack([self.parsed_dataset.train_X, all_X])
        self.parsed_dataset.train_s = np.vstack([self.parsed_dataset.train_s, all_s])
        self.parsed_dataset.train_y = np.hstack([self.parsed_dataset.train_y, all_y])

        return self.parsed_dataset
