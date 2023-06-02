import logging
from arguments import TrainingLoopParameters, ParsedDataset
from .training_utils import CreateSimpleIterators, GroupIterators


def orchestrator(training_loop_parameters: TrainingLoopParameters, parsed_dataset: ParsedDataset):
    """
    :param training_loop_parameters: contains all information needed to train the model and the information related to settings
    :param parsed_dataset: contains all information related to dataset.
    :return:
    """
    if training_loop_parameters.log_run:
        logger = logging.getLogger(training_loop_parameters.unique_id_for_run)
    else:
        logger = False

    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []

    # create the iterator

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

    # for ep in range(training_loop_parameters.n_epochs):
    #     if logger:
    #         logger.info("start of epoch block")
