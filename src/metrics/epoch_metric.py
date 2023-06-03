import numpy as np
from misc_utils import generate_mask
from .fairness_metric import EpsFairness
from arguments import EpochMetric, EpochMetricOutput
from sklearn.metrics import accuracy_score, balanced_accuracy_score


class CalculateEpochMetric:
    """Calculates epoch level metric"""

    def __init__(self, epoch_metric: EpochMetric):
        self.epoch_metric = epoch_metric

    def run(self):
        overall_accuracy = accuracy_score(y_true=self.epoch_metric.labels, y_pred=self.epoch_metric.predictions)
        overall_balanced_accuracy = balanced_accuracy_score(y_true=self.epoch_metric.labels,
                                                            y_pred=self.epoch_metric.predictions)
        # generate all possible masks
        all_unique_groups = np.unique(self.epoch_metric.s, axis=0)
        all_unique_groups_masks = [generate_mask(all_s=self.epoch_metric.s, mask_pattern=group) for group in
                                   all_unique_groups]

        group_wise_accuracy = {}
        # calculate per-group accuracy
        for group, group_mask in zip(all_unique_groups, all_unique_groups_masks):
            group_accuracy = accuracy_score(y_true=self.epoch_metric.labels[group_mask],
                                            y_pred=self.epoch_metric.predictions[group_mask])
            group_balanced_accuracy = balanced_accuracy_score(y_true=self.epoch_metric.labels[group_mask],
                                                              y_pred=self.epoch_metric.predictions[group_mask])

            positive_label_mask = np.logical_and(group_mask, self.epoch_metric.labels == 1)
            negative_label_mask = np.logical_and(group_mask, self.epoch_metric.labels == 0)

            positive_group_accuracy = accuracy_score(y_true=self.epoch_metric.labels[positive_label_mask],
                                                     y_pred=self.epoch_metric.predictions[positive_label_mask])

            negative_group_accuracy = accuracy_score(y_true=self.epoch_metric.labels[negative_label_mask],
                                                     y_pred=self.epoch_metric.predictions[negative_label_mask])

            group_wise_accuracy[tuple(group)] = [np.sum(group_mask), np.sum(positive_label_mask),
                                                 np.sum(negative_label_mask), round(group_accuracy, 4),
                                                 round(group_balanced_accuracy, 4), round(positive_group_accuracy, 4),
                                                 round(negative_group_accuracy, 4)]

        eps_fairness_metric = EpsFairness(prediction=self.epoch_metric.predictions, label=self.epoch_metric.labels,
                                          aux=self.epoch_metric.s,
                                          all_possible_groups=all_unique_groups,
                                          all_possible_groups_mask=all_unique_groups_masks,
                                          fairness_mode=[self.epoch_metric.fairness_function]).run()

        emo = EpochMetricOutput(accuracy=overall_accuracy, balanced_accuracy=overall_balanced_accuracy,
                                eps_fairness=eps_fairness_metric, loss=0.0, epoch_number=0,
                                group_wise_accuracy=group_wise_accuracy, other_info=None)

        return emo
