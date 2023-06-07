import copy

import numpy as np
from pathlib import Path
from typing import Optional, Union
from arguments import ParsedDataset
from training_loops import training_utils


class GenerateData:
    def __init__(self, parsed_dataset: ParsedDataset, positive_gen_model: Path, negative_gen_model: Path,
                 size_of_each_group: Optional[Union[dict, int]] = None):
        self.parsed_dataset = parsed_dataset
        self.positive_gen_model = positive_gen_model
        self.negative_gen_model = negative_gen_model
        self.size_of_each_group = size_of_each_group

    def run(self):
        if self.size_of_each_group == None:
            self.size_of_each_group = {}
            for group in self.parsed_dataset.all_groups:
                mask = training_utils.generate_mask(self.parsed_dataset.train_s, group)
                positive_mask = np.logical_and(mask, self.parsed_dataset.train_y == 1)
                negative_mask = np.logical_and(mask, self.parsed_dataset.train_y == 0)
                self.size_of_each_group[tuple(group)] = {
                    1: np.sum(positive_mask),
                    0: np.sum(negative_mask)
                }

        if type(self.size_of_each_group) == int:
            group_size = copy.deepcopy(self.size_of_each_group)
            self.size_of_each_group = {}
            for group in self.parsed_dataset.all_groups:
                self.size_of_each_group[tuple(group)] = {
                    1: group_size,
                    0: group_size
                }
        all_X, all_s, all_y = [], [], []

        for group, group_size in self.size_of_each_group.item():
            positive_size = group_size[1]
            negative_size = group_size[0]
            pass
