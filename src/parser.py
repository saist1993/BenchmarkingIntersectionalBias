# Top level file which calls the parser.
import numpy as np
from dataclasses import dataclass
from texttable import Texttable
from arguments import EpochMetricOutput, EPSFairnessMetric

from itertools import combinations_with_replacement

fairness_function = 'equal_odds'
level_1_strategy_params = {'keep_last_k': 100.0}
level_2_strategy_params = {'relaxation_threshold': 0.02,
                           'fairness_function': fairness_function}

import os
from numpy import nan
from pathlib import Path

from itertools import combinations_with_replacement

from numpy import array


@dataclass
class BlockData:
    unique_id: str
    arguments: dict
    train_epoch_metric: EpochMetricOutput
    valid_epoch_metric: EpochMetricOutput
    test_epoch_metric: EpochMetricOutput


class BestCandidateMechanism:
    """
    The best candidate mechansim has two levels. But to understand this two levels, we need to understand data

    data = [ [blocks for run 1 - output of BasicLogParser.block_parser], [blocks for run 2], [blocks for run 3], [] ]

    Level 1 - Now [blocks for run 1] would have 50 blocks if there are 50 epochs ..
    So level 1 transforms this to say X epochs.
    This is usefule if we just want last 10% of the runs, or if we want last epoch .. we can also set it to none, if we
    don't care about it.

    Level 2 - [[pruned blocks for run 1], [pruned blocks for run 2], [pruned blocks for run 3], []] .. The objective
    here is to get final single result.
    """

    def __init__(self, data, level_1_strategy, level_1_strategy_params, level_2_strategy, level_2_strategy_params):
        self.data = data
        self.level_1_strategy = level_1_strategy
        self.level_1_strategy_params = level_1_strategy_params
        self.level_2_strategy = level_2_strategy
        self.level_2_strategy_params = level_2_strategy_params

    def keep_last_k(self):
        pruned_data = []
        for d in self.data:
            pruned_data.append(d[::-1][:int(len(d) * self.level_1_strategy_params['keep_last_k'] * 0.01)])
        self.data = pruned_data

    def compute_metric(self, block):
        computed_metric = block.valid_epoch_metric.eps_fairness[
            self.level_2_strategy_params['fairness_function']].intersectional_bootstrap[10]
        eps = [i[-1] for i in computed_metric]
        return np.max(eps)
        all_new_eps = []
        for i, j in combinations_with_replacement(eps, 2):
            if i != j:
                all_new_eps.append(max(i, j) - min(i, j))
        return np.mean(all_new_eps)

    def relaxation_threshold(self):
        try:
            best_validation_accuracy = max(
                [block.valid_epoch_metric.balanced_accuracy for blocks in self.data for block in blocks])
        except ValueError:
            print("here")
        all_blocks_flat = [block for blocks in self.data for block in blocks
                           if block.valid_epoch_metric.balanced_accuracy > best_validation_accuracy -
                           self.level_2_strategy_params['relaxation_threshold']]

        # best_fairness_index = np.argmin([block.valid_epoch_metric.eps_fairness[self.level_2_strategy_params[
        #     'fairness_function']].intersectional_bootstrap[0]
        #                                  for block in all_blocks_flat])

        best_fairness_index = np.argmin([self.compute_metric(block)
                                         for block in all_blocks_flat])

        return all_blocks_flat[best_fairness_index]

    def run(self):
        if self.level_1_strategy == 'keep_last_k':
            self.keep_last_k()
        if self.level_2_strategy == 'relaxation_threshold':
            block = self.relaxation_threshold()

        return block


class BasicLogParser:

    @staticmethod
    def block_parser(block):

        train_epoch_metric, valid_epoch_metric, test_epoch_metric = [None, None, None]

        for b in block:
            if "train epoch metric:" in b:
                train_epoch_metric = eval(b.split("train epoch metric: ")[1])  # THIS IS NOT SAFE!
            if "test epoch metric:" in b:
                test_epoch_metric = eval(b.split("test epoch metric: ")[1])
            if "valid epoch metric:" in b:
                valid_epoch_metric = eval(b.split("valid epoch metric: ")[1])

        return [train_epoch_metric, valid_epoch_metric, test_epoch_metric]

    def core_parser(self, file_name: Path):
        # print(f"file name received {file_name}")
        with open(file_name, 'r') as f:
            lines = [line for line in f]
            unique_id = lines[0].split("unique id is:")[1]
            arguments = lines[1].split("arguments: ")[1]
        # print(arguments)
        if "per_group_label_number_of_examples=-1" not in arguments:
            return None
        blocks = []
        for l in lines[2:]:
            if "start of epoch block" in l:
                temp = []
            elif "end of epoch block" in l:
                blocks.append(temp)
            else:
                temp.append(l)

        parsed_blocks = []
        for b in blocks:
            block = self.block_parser(b)
            if None not in block[1:]:
                block = BlockData(
                    unique_id=unique_id,
                    arguments=arguments,
                    train_epoch_metric=block[0],
                    valid_epoch_metric=block[1],
                    test_epoch_metric=block[2]
                )
                parsed_blocks.append(block)

        return parsed_blocks

    def get_parsed_content(self, dataset_name, method, model, seed, fairness_function):
        # Step 1: Find the correct directory
        log_files_location = Path(f"logs/{dataset_name}/{method}/{model}/{seed}/{fairness_function}")
        all_log_files_names = log_files_location.glob('*')
        # all_log_files_content = [self.core_parser(file_name) for file_name in all_log_files_names]

        all_log_files_content = []
        for file_name in all_log_files_names:
            temp = self.core_parser(file_name)
            if temp:
                all_log_files_content.append(temp)
        return all_log_files_content


def get_best_run(dataset_name, method, model, seed, fairness_function,
                 level_1_strategy, level_1_strategy_params, level_2_strategy, level_2_strategy_params):
    basic_log_parser = BasicLogParser()
    all_log_files_content = basic_log_parser.get_parsed_content(dataset_name, method, model, seed, fairness_function)
    best_candidate_mechanism = BestCandidateMechanism(all_log_files_content,
                                                      level_1_strategy, level_1_strategy_params,
                                                      level_2_strategy, level_2_strategy_params)
    return best_candidate_mechanism.run()


def temp_table_generator(dataset_name, fairness_function, methods):
    dataset_names = [dataset_name]
    # dataset_names = ['twitter_hate_speech']
    # dataset_names = ['celeb_multigroup_v3']
    models = ['simple_non_linear']
    seeds = [10, 20, 30, 40, 50]
    # seeds = [50]
    # fairness_function = 'equal_odds'
    # fairness_function = 'equal_opportunity'
    k = 2

    level_1_strategy_params = {'keep_last_k': 100.0}
    level_2_strategy_params = {'relaxation_threshold': 0.03,
                               'fairness_function': fairness_function}

    rows = []
    average_rows = []
    rows.append(['method', 'balanced accuracy', 'fairness', 'confidence_interval', 'seed', 'min fair', 'max fair',
                 'average fair', 'spread fair', 'max difference', 'average difference', 'average ratio'])
    average_rows.append(['method', 'balanced accuracy', 'fairness', 'min fair', 'max fair', 'average fair',
                         'spread fair', 'average max difference', 'average all difference',
                         'average all min max ratio'])

    for dataset in dataset_names:
        for model in models:
            for method in methods:
                rows_temp = []
                for seed in seeds:
                    result = get_best_run(dataset_name=dataset, method=method,
                                          model=model,
                                          seed=seed, fairness_function=fairness_function,
                                          level_1_strategy='keep_last_k',
                                          level_1_strategy_params=level_1_strategy_params,
                                          level_2_strategy='relaxation_threshold',
                                          level_2_strategy_params=level_2_strategy_params)
                    accuracy = result.test_epoch_metric.balanced_accuracy
                    fairness = result.test_epoch_metric.eps_fairness[fairness_function].intersectional_bootstrap[0]
                    confidence_interval = \
                        result.test_epoch_metric.eps_fairness[fairness_function].intersectional_bootstrap[1]
                    confidence_interval[0], confidence_interval[1] = round(confidence_interval[0], k), \
                        round(confidence_interval[1], k)

                    intersectional_bootstrap = result.test_epoch_metric.eps_fairness[
                        fairness_function].intersectional_bootstrap
                    # print(intersectional_bootstrap)
                    min_prob, max_prob, mean_prob, mean_std, minmax_difference = intersectional_bootstrap[4:9]

                    computed_metric = intersectional_bootstrap[10]
                    eps = [i[-1] for i in computed_metric]
                    all_differences = []
                    for i, j in combinations_with_replacement(eps, 2):
                        if i != j:
                            all_differences.append(max(i, j) - min(i, j))

                    all_min_max_ratio = []
                    for i, j in combinations_with_replacement(eps, 2):
                        if i != j:
                            all_min_max_ratio.append(max(i, j) / min(i, j))

                    rows_temp.append(
                        [method, round(accuracy, k), round(fairness, k), confidence_interval, seed, min_prob, max_prob,
                         mean_prob, mean_std, minmax_difference, np.mean(all_differences), np.mean(all_min_max_ratio)])

                    if False:
                        print(result.arguments)
                        print(result.test_epoch_metric.epoch_number)
                # average over seeds
                rows = rows + rows_temp
                average_accuracy = round(np.mean([r[1] for r in rows_temp]), k)
                average_accuracy_std = round(np.std([r[1] for r in rows_temp]), k)

                average_fairness = round(np.mean([r[2] for r in rows_temp]), k)
                average_fairness_std = round(np.std([r[2] for r in rows_temp]), k)

                average_min_fairness = round(np.mean([r[5] for r in rows_temp]), k)
                average_min_fairness_std = round(np.std([r[5] for r in rows_temp]), k)

                average_max_fairness = round(np.mean([r[6] for r in rows_temp]), k)
                average_max_fairness_std = round(np.std([r[6] for r in rows_temp]), k)

                average_mean_fairness = round(np.mean([r[7] for r in rows_temp]), k)
                average_mean_fairness_std = round(np.std([r[7] for r in rows_temp]), k)

                average_std_fairness = round(np.mean([r[8] for r in rows_temp]), k)
                average_std_fairness_std = round(np.std([r[8] for r in rows_temp]), k)

                average_minmax_difference_fairness = round(np.mean([r[9] for r in rows_temp]), k)
                average_minmax_difference_fairness_std = round(np.std([r[9] for r in rows_temp]), k)

                average_all_difference_fairness = round(np.mean([r[10] for r in rows_temp]), k)
                average_all_difference_ratio_std = round(np.std([r[10] for r in rows_temp]), k)

                average_all_minmax_ratio_fairness = round(np.mean([r[11] for r in rows_temp]), k)
                average_all_minmax_ratio_std = round(np.std([r[11] for r in rows_temp]), k)

                average_rows.append([method, f"{average_accuracy} +/- {average_accuracy_std}",
                                     f"{average_fairness} +/- {average_fairness_std}",
                                     f"{average_min_fairness} +/- {average_min_fairness_std}",
                                     f"{average_max_fairness} +/- {average_max_fairness_std}",
                                     f"{average_mean_fairness} +/- {average_mean_fairness_std}",
                                     f"{average_std_fairness} +/- {average_std_fairness_std}",
                                     f"{average_minmax_difference_fairness} +/- {average_minmax_difference_fairness_std}",
                                     f"{average_all_difference_fairness} +/- {average_all_difference_ratio_std}",
                                     f"{average_all_minmax_ratio_fairness} +/- {average_all_minmax_ratio_std}"])

    t = Texttable()
    t.add_rows(rows)
    # print(t.draw())

    t = Texttable()
    t.add_rows(average_rows)
    # print(t.draw())
    return t


equal_odds = temp_table_generator('twitter_hate_speech', 'equal_odds',
                                  ['unconstrained', 'adversarial_single', 'fairgrad', 'mixup_regularizer'])
print("twitter hate speech equal odds")
print(equal_odds.draw())
