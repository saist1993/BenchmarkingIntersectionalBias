import numpy as np
from pathlib import Path
from arguments import ParsedDataset
from .common_utils import split_data
from sklearn.preprocessing import StandardScaler


class SimpleClassificationDataset:

    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.seed = params['seed']
        self.dataset_location = Path(params['dataset_location'])
        self.standard_scalar = params['standard_scalar']
        self.train_split = 0.80
        self.valid_split = 0.25

        self.X, self.y, self.s = np.load(self.dataset_location / Path('encoding_cls.npy')), \
            np.load(self.dataset_location / Path('labels.npy')).squeeze(), \
            np.load(self.dataset_location / Path('s.npy'))

        if self.dataset_name == "anxiety_v1":
            self.s = self.s[:, :1]
        if self.dataset_name == "anxiety_v2":
            self.s = self.s[:, :2]
        # self.s[:,0]

    def run(self):
        """Orchestrates the whole process"""
        X, y, s = self.X, self.y, self.s

        # the dataset is shuffled so as to get a unique test set for each seed.
        index, test_index, dev_index = split_data(X.shape[0], train_split=self.train_split,
                                                  valid_split=self.valid_split)
        X, y, s = X[index], y[index], s[index]
        train_X, train_y, train_s = X[:dev_index, :], y[:dev_index], s[:dev_index]
        valid_X, valid_y, valid_s = X[dev_index:test_index, :], y[dev_index:test_index], s[dev_index:test_index]
        test_X, test_y, test_s = X[test_index:, :], y[test_index:], s[test_index:]

        if self.standard_scalar:
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            valid_X = scaler.transform(valid_X)
            test_X = scaler.transform(test_X)

        def flatten(seq):
            seen = {}
            for item in seq:
                marker = tuple(item)
                if marker in seen: continue
                seen[marker] = len(seen)
            return seen

        s_list_to_int = flatten(np.vstack([train_s, valid_s, test_s]))

        parsed_dataset = ParsedDataset(
            input_dim=train_X.shape[1],
            dataset_name=self.dataset_name,
            train_X=train_X,
            train_y=train_y,
            train_s=train_s,
            valid_X=valid_X,
            valid_y=valid_y,
            valid_s=valid_s,
            test_X=test_X,
            test_y=test_y,
            test_s=test_s,
            s_list_to_int=s_list_to_int,
            s_int_to_list={value: key for key, value in s_list_to_int.items()},
            number_of_main_task_label=len(np.unique(train_y)),
            number_of_aux_label_per_attribute=[len(np.unique(train_s[:, i])) for i in
                                               range(train_s.shape[1])],
            all_groups=[tuple(key) for key, value in s_list_to_int.items()]
        )

        return parsed_dataset
