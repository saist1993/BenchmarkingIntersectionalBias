import torch
import random
import logging
import numpy as np
from pathlib import Path
from models import base_models
from typing import Optional, Dict

from dataset_parser import twitter_hate_speech


def get_logger(unique_id_for_run, log_file_name: Optional[str], log_dir, runner_arguments) -> None:
    """Handel logger setup"""
    logger = logging.getLogger(str(unique_id_for_run))

    logger.setLevel('INFO')

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    if runner_arguments.log_file_name:
        log_file_name = Path(log_file_name + ".log")
    else:
        log_file_name = Path(str(unique_id_for_run) + ".log")
    log_file_name = Path(log_dir) / Path(runner_arguments.dataset_name) / Path(runner_arguments.method) / \
                    Path(runner_arguments.model) / Path(str(runner_arguments.seed)) / Path(
        str(runner_arguments.fairness_function)) / log_file_name

    log_file_name.parent.mkdir(exist_ok=True, parents=True)

    fileHandler = logging.FileHandler(log_file_name, mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)


def get_model(method: str, model_name: str, other_meta_data: Dict, device: torch.device, use_batch_norm: float = 0.0,
              use_dropout: float = 0.0):
    number_of_aux_label_per_attribute = other_meta_data['number_of_aux_label_per_attribute']
    attribute_id = other_meta_data['attribute_id']
    if attribute_id is not None:
        number_of_aux_label_per_attribute = [number_of_aux_label_per_attribute[attribute_id]]

    if model_name == 'simple_non_linear':

        model_arch = {
            'encoder': {
                'input_dim': other_meta_data['input_dim'],
                'output_dim': other_meta_data['number_of_main_task_label']
            }
        }

        model_params = {
            'model_arch': model_arch,
            'device': device,
            'use_batch_norm': use_batch_norm,
            'use_dropout': use_dropout
        }

        if 'adversarial_single' in method:
            total_adv_dim = len(other_meta_data['s_flatten_lookup'])
            model_params['model_arch']['adv'] = {'output_dim': [total_adv_dim]}
            model = base_models.SimpleNonLinear(model_params)
        elif method in ['adversarial_group', 'adversarial_group_with_fairness_loss']:
            model_params['model_arch']['adv'] = {'output_dim': number_of_aux_label_per_attribute}
            model = base_models.SimpleNonLinear(model_params)
        elif method == 'adversarial_moe':
            model_params['model_arch']['adv'] = {'output_dim': number_of_aux_label_per_attribute}
            raise NotImplementedError
            model = adversarial_moe.SimpleNonLinear(model_params)
        else:
            model = base_models.SimpleNonLinear(model_params)
    else:
        raise NotImplementedError

    return model


def resolve_device(device=None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        print('No cuda devices were available. The model runs on CPU')
    return device


def set_seed(seed: int) -> None:
    """Sets seed for random, torch, and numpy"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_raw_dataset(dataset_name: str, **kwargs):
    """
    This function just reads the dataset, pre-processes it, and finally returns a clean
        train_X, train_y, train_s, valid_X, valid_y, valid_s, test_X, test_y, test_s
    """
    if 'twitter_hate_speech' in dataset_name.lower():
        kwargs['dataset_location'] = Path(
            '../datasets/Multilingual-Hate-Speech-LREC-Huang-2019-data/data/split/English')
        dataset_creator = twitter_hate_speech.DatasetTwitterHateSpeech(dataset_name=dataset_name, **kwargs)
        return dataset_creator.run()
    elif dataset_name.lower() in 'gaussian_toy':
        dataset_creator = gaussian_dataset.GaussianDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif 'adult_multi_group' in dataset_name.lower() or 'celeb_multigroup_v' in dataset_name.lower():
        dataset_creator = simple_classification_dataset.SimpleClassificationDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() in ['adult']:
        dataset_creator = simple_classification_dataset.SimpleClassificationDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif "numeracy" in dataset_name.lower():
        kwargs['dataset_location'] = config.numeracy_path[0]
        dataset_creator = numeracy.SimpleClassificationDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    else:
        raise CustomError("No such dataset")

    return iterators, other_meta_data
