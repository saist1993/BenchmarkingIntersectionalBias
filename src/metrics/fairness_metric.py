import warnings
import numpy as np
from typing import List
from copy import deepcopy
from itertools import combinations
from arguments import EPSFairnessMetric


class EstimateProbability:

    def __init__(self):
        self.alpha = 2.0
        self.beta = 2.0
        self.number_of_bootstrap_dataset = 1000

    @staticmethod
    def get_confidence_interval(eps, percentile=95.0):
        return [np.percentile(eps, [100.0 - percentile])[0], np.percentile(eps, [percentile])[0]]

    def empirical_estimate_demographic_parity(self, preds, labels, mask):
        '''
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1 / number_of_instance_with_s

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        new_preds = preds[mask]
        new_labels = preds[mask]  # this gives us all s

        # find number of instances with label = 1
        number_of_instance_with_s_and_y_1 = np.count_nonzero(new_preds == 1)

        return number_of_instance_with_s_and_y_1 / len(new_preds) * 1.0

    def calculate_smoothed_empirical_estimate_demographic_parity(self, preds, labels, mask):
        new_preds = preds[mask]
        new_labels = labels[mask]  # this gives us all s

        # find number of instances with label = 1
        number_of_instance_with_s_and_y_1 = np.count_nonzero(new_preds == 1)

        prob = (number_of_instance_with_s_and_y_1 + self.alpha) / (len(new_preds) * 1.0 + self.alpha + self.beta)

        return prob

    def smoothed_empirical_estimate_demographic_parity(self, preds, labels, masks):
        '''
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1 / number_of_instance_with_s

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        all_probs = []
        for mask in masks:
            prob = self.calculate_smoothed_empirical_estimate_demographic_parity(preds, labels, mask)
            all_probs.append(prob)

        max_prob = max(all_probs)
        min_prob = min(all_probs)
        return [np.log(max_prob / min_prob), [0.0, 0.0]]

    def simple_bayesian_estimate_demographic_parity(self, preds, labels, masks):

        all_eps = []
        number_of_simulation = range(1000)
        for _ in number_of_simulation:
            all_probs = []
            for mask in masks:
                new_preds = preds[mask]
                new_labels = labels[mask]  # this gives us all s

                # find number of instances with preds = 1
                n_1_s = np.count_nonzero(new_preds == 1)
                n_s = len(new_preds)
                prob = np.random.beta(n_1_s + (1 / 3.0), n_s + (1 / 3.0) - n_1_s, size=1)
                all_probs.append(prob)

            max_prob = max(all_probs)
            min_prob = min(all_probs)
            eps = np.log(max_prob / min_prob)
            all_eps.append(eps)

        return [np.mean(all_eps), self.get_confidence_interval(all_eps)]

    def smoothed_empirical_estimate_rate_parity(self, preds, labels, masks, use_tpr=True, all_possible_groups=None):
        '''
            - if use_tpr is False, then return fpr
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1_and_label_1 + alpha /
             number_of_instance_with_s_and_lable_1 + alpha + beta

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        all_probs = []
        for mask in masks:
            new_preds = preds[mask]
            new_labels = labels[mask]  # this gives us all s

            # find the number of instances with pred=1, and S=s and label=1 - @TODO: Test this!
            if use_tpr:
                new_mask = (new_preds == 1) & (new_labels == 1)
                numerator = np.count_nonzero(new_mask == 1)
                new_mask = new_labels == 1
                denominator = np.count_nonzero(new_mask)
            else:
                new_mask = (new_preds == 1) & (new_labels == 0)
                numerator = np.count_nonzero(new_mask == 1)
                new_mask = new_labels == 0
                denominator = np.count_nonzero(new_mask)

            # find the number of instance with S=s and label=1
            try:
                prob = (numerator + self.alpha) / (denominator * 1.0 + self.alpha + self.beta)
            except ZeroDivisionError:
                prob = 0.0
            all_probs.append(prob)

        max_index = np.argmax(all_probs)
        min_index = np.argmin(all_probs)

        max_prob = all_probs[max_index]
        min_prob = all_probs[min_index]

        max_group = all_possible_groups[max_index]
        min_group = all_possible_groups[min_index]
        # average_prob = [max(i/j , j/i) for i,j in combinations(all_probs , 2 )]

        average_prob = []
        for i, j in combinations(all_probs, 2):
            try:
                a = i / j
            except ZeroDivisionError:
                a = 0
            try:
                b = j / i
            except ZeroDivisionError:
                b = 0
            average_prob.append(max(a, b))

        if min_prob == 0:
            warnings.warn("There is something fishy in the data. There was no example atleast for one group at all.")
            return [0.0, [0.0, 0.0], average_prob]

        return [np.log(max_prob / min_prob), [0.0, 0.0], all_probs, (min_group, max_group)]

    def simple_bayesian_estimate_rate_parity(self, preds, labels, masks, use_tpr=True):

        all_eps = []
        number_of_simulation = range(1000)
        for _ in number_of_simulation:
            all_probs = []
            for mask in masks:
                new_preds = preds[mask]
                new_labels = labels[mask]  # this gives us all s

                # find number of instances with label = 1 and pred = 1
                if use_tpr:
                    new_mask = (new_preds == 1) & (new_labels == 1)
                    numerator = np.count_nonzero(new_mask == 1)

                    new_mask = new_labels == 1
                    denominator = np.count_nonzero(new_mask)
                else:
                    new_mask = (new_preds == 1) & (new_labels == 0)
                    numerator = np.count_nonzero(new_mask == 1)

                    new_mask = new_labels == 0
                    denominator = np.count_nonzero(new_mask)

                prob = np.random.beta(numerator + (1 / 3.0), denominator + (1 / 3.0) - numerator, size=1)
                all_probs.append(prob)

            max_prob = max(all_probs)
            min_prob = min(all_probs)
            eps = np.log(max_prob / min_prob)
            all_eps.append(eps)
        return [np.mean(all_eps), self.get_confidence_interval(all_eps)]

    def bootstrap_estimate_demographic_parity(self, preds, labels, masks):
        """ Similar to bagging. Performed via smoothing"""
        all_eps = []
        for _ in range(self.number_of_bootstrap_dataset):
            index_select = np.random.choice(len(preds), len(preds), replace=True)

            eps, _ = self.smoothed_empirical_estimate_demographic_parity(preds[index_select], labels[index_select],
                                                                         [m[index_select] for m in masks])
            all_eps.append(eps)
        # np.percentile(all_eps, [97.5])
        return [np.mean(all_eps), self.get_confidence_interval(all_eps)]

    def bootstrap_estimate_rate_parity(self, preds: np.ndarray, labels: np.ndarray, masks: np.ndarray,
                                       use_tpr: bool = True,
                                       all_possible_groups=None):
        """ Similar to bagging. Performed via smoothing"""
        all_eps = []
        all_average_prob = []
        all_min_max_groups = []
        all_max_prob, all_min_prob, all_mean_prob, all_std_prob, all_difference_prob = [], [], [], [], []
        for _ in range(self.number_of_bootstrap_dataset):
            index_select = np.random.choice(len(preds), len(preds), replace=True)

            eps, _, average_prob, min_max_group = self.smoothed_empirical_estimate_rate_parity(preds[index_select],
                                                                                               labels[index_select],
                                                                                               [m[index_select] for m in
                                                                                                masks], use_tpr=use_tpr,
                                                                                               all_possible_groups=all_possible_groups)
            all_eps.append(eps)
            all_average_prob.append(average_prob)
            all_min_max_groups.append(min_max_group)
            all_max_prob.append(max(average_prob))
            all_min_prob.append(min(average_prob))
            all_mean_prob.append(np.mean(average_prob))
            all_std_prob.append(np.std(average_prob))
            all_difference_prob.append(np.max(average_prob) - np.min(average_prob))

        min_group, min_group_count = np.unique(np.asarray(all_min_max_groups)[:, 0], axis=0, return_counts=True)
        min_group = min_group[np.argmax(min_group_count)]

        max_group, max_group_count = np.unique(np.asarray(all_min_max_groups)[:, 1], axis=0, return_counts=True)
        max_group = max_group[np.argmax(max_group_count)]

        max_prob = np.mean(all_max_prob)
        min_prob = np.mean(all_min_prob)
        mean_prob = np.mean(all_mean_prob)
        mean_std = np.mean(all_std_prob)
        minmax_difference = np.mean(all_difference_prob)

        probs = np.mean(all_average_prob, axis=0)
        return [np.mean(all_eps), self.get_confidence_interval(all_eps), np.mean(all_average_prob, axis=0), min_group,
                max_group, min_prob, max_prob, mean_prob, mean_std, minmax_difference]


class EpsFairness():

    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 all_possible_groups: List[List], all_possible_groups_mask: List, fairness_mode: List):
        self.prediction = prediction
        self.label = label
        self.aux = aux
        self.all_possible_groups = all_possible_groups
        self.all_possible_groups_mask = all_possible_groups_mask

        self.fairness_mode = fairness_mode

        self.es = EstimateProbability()

    def demographic_parity(self, group_masks, robust_estimation_method='smoothed_empirical_estimate',
                           all_possible_groups=None):
        """
        Following steps will be done
            -> Calculate p(y_hat=1/S=s) for all s
            -> find the ratio with the maximum value and then return that eps!
        :return:
        """

        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps = self.es.smoothed_empirical_estimate_demographic_parity(self.prediction, self.label, group_masks)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps = self.es.simple_bayesian_estimate_demographic_parity(self.prediction, self.label, group_masks)
        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps = self.es.bootstrap_estimate_demographic_parity(self.prediction, self.label, group_masks)
        else:
            raise NotImplementedError

        eps.append(['dpr'])
        return eps

    def tpr_parity(self, group_masks, robust_estimation_method='smoothed_empirical_estimate', all_possible_groups=None):
        """
        - This measure is also called equal opportunity with y=1 being the default!
        - We need to estimate p(y_hat=1/y=1,S=s) for all s
        :param group_masks:
        :param robust_estimation_method:
        :return:
        """

        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                  use_tpr=True)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=True)

        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=True,
                                                         all_possible_groups=all_possible_groups)

        else:
            raise NotImplementedError
        eps.append(['tpr'])
        return eps

    def fpr_parity(self, group_masks, robust_estimation_method='smoothed_empirical_estimate', all_possible_groups=None):
        """
        - This measure is also called equal opportunity with y=1 being the default!
        - We need to estimate p(y_hat=1/y=1,S=s) for all s
        :param group_masks:
        :param robust_estimation_method:
        :return:
        """

        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                  use_tpr=False)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=False)

        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=False,
                                                         all_possible_groups=all_possible_groups)

        else:
            raise NotImplementedError
        eps.append(['fpr'])
        return eps

    def equal_odds(self, group_masks, robust_estimation_method='smoothed_empirical_estimate',
                   all_possible_groups=None):
        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps_fpr = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                      use_tpr=False)
            eps_tpr = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                      use_tpr=True)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps_fpr = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                   use_tpr=False)
            eps_tpr = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                   use_tpr=True)
        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps_fpr = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=False,
                                                             all_possible_groups=all_possible_groups)
            eps_tpr = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=True,
                                                             all_possible_groups=all_possible_groups)

        else:
            raise NotImplementedError

        size = [np.sum(mask) for mask in group_masks]

        eps_fpr.append(['fpr'])
        eps_tpr.append(['tpr'])

        index = np.argmax([eps_fpr[0], eps_tpr[0]])
        return [eps_fpr, eps_tpr][index]

    def run(self):
        """
        For now this works on binary setting. Not sure how to extend this for other settings!
        :return:
        """

        def get_analytics(type_of_group, robust_estimation_method, fairness_mode='demographic_parity',
                          bias_amplification_mode=False):

            if bias_amplification_mode:
                self.original_prediction = deepcopy(self.prediction)
                self.prediction = self.label

            if fairness_mode == 'demographic_parity':
                output = self.demographic_parity(np.asarray(self.all_possible_groups_mask),
                                                 robust_estimation_method, self.all_possible_groups)
            elif fairness_mode == 'equal_opportunity':
                output = self.tpr_parity(np.asarray(self.all_possible_groups_mask),
                                         robust_estimation_method, self.all_possible_groups)
            elif fairness_mode == 'equal_odds':
                output = self.equal_odds(np.asarray(self.all_possible_groups_mask),
                                         robust_estimation_method, self.all_possible_groups)
            else:
                raise NotImplementedError

            if bias_amplification_mode:
                self.prediction = self.original_prediction

            output.append([(list(i), np.sum(j), round(k, 4)) for i, j, k in
                           zip(self.all_possible_groups, self.all_possible_groups_mask, output[2])])

            output.pop(2)

            # fairness_values = [i[-1] for i in output[-1]]

            return output

        fairness_metrics = {}

        for fairness_mode in self.fairness_mode:
            """
            Just intersectional works as eps of intersectional will always be greater or equal to 
            gerrymandering group and independent group
            """

            # fairness_mode_intersectional_smoothed = get_analytics('intersectional',
            #                                                       'smoothed_empirical_estimate',
            #                                                       fairness_mode)
            # fairness_mode_intersectional_simple_bayesian = get_analytics('intersectional',
            #                                                              'simple_bayesian_estimate',
            #                                                              fairness_mode)

            fairness_mode_intersectional_smoothed = 0.0
            fairness_mode_intersectional_simple_bayesian = 0.0
            fairness_mode_intersectional_bootstrap = get_analytics('intersectional',
                                                                   'bootstrap_empirical_estimate',
                                                                   fairness_mode)

            intersectional_smoothed_bias_amplification = 0
            intersectional_simple_bayesian_bias_amplification = 0
            intersectional_bootstrap_bias_amplification = 0

            fairness_metric_tracker = EPSFairnessMetric(
                intersectional_smoothed=fairness_mode_intersectional_smoothed,
                intersectional_simple_bayesian=fairness_mode_intersectional_simple_bayesian,
                intersectional_bootstrap=fairness_mode_intersectional_bootstrap,
                intersectional_smoothed_bias_amplification=intersectional_smoothed_bias_amplification,
                intersectional_simple_bayesian_bias_amplification=intersectional_simple_bayesian_bias_amplification,
                intersectional_bootstrap_bias_amplification=intersectional_bootstrap_bias_amplification,
            )

            fairness_metrics[fairness_mode] = fairness_metric_tracker

        return fairness_metrics
