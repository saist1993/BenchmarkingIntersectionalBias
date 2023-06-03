import torch
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Optional, List, Dict, Callable, Union


@dataclass
class RunnerArguments():
    """Arguments for the main function
    @TODO: clean this arguments files a bit!
    """
    seed: int = 50
    dataset_name: str = "twitter_hate_speech"
    batch_size: int = 1024
    model: str = "simple_non_linear"
    epochs: int = 20
    save_model_as: Optional[str] = None
    method: str = 'fairgrad'
    optimizer_name: str = 'adam'
    lr: float = 0.001
    use_wandb: bool = False  # For legacy purpose. Not in the current codebase
    adversarial_lambda: float = 20.0
    dataset_size: int = 10000
    attribute_id: Optional[int] = None
    fairness_lambda: float = 0.0
    log_file_name: Optional[str] = None
    fairness_function: str = 'equal_odds'
    titled_t: float = 5.0
    mixup_rg: float = 0.5
    max_number_of_generated_examples: float = 1.0
    use_dropout: float = 0.2  # 0.0 corresponds to no dropout being applied
    use_batch_norm: float = 0.0  # 0.0 corresponds to no batch norm being applied
    per_group_label_number_of_examples: int = 1000
    positive_gen_model: str = "gen_model_positive_numeracy_10_simple.pt"
    negative_gen_model: str = "gen_model_negative_numeracy_10_simple.pt"
    log_dir: str = '../logs'
    standard_scalar: bool = True
    iterator_type: str = "simple_iterator"


@dataclass
class ParsedDataset():
    """Output of parsed dataset"""
    input_dim: int
    dataset_name: str
    train_X: np.asarray
    train_y: np.asarray
    train_s: np.asarray
    valid_X: np.asarray
    valid_y: np.asarray
    valid_s: np.asarray
    test_X: np.asarray
    test_y: np.asarray
    test_s: np.asarray
    s_list_to_int: Dict
    s_int_to_list: Dict
    number_of_main_task_label: int
    number_of_aux_label_per_attribute: List[int]
    all_groups: List[tuple]


@dataclass
class TrainingLoopParameters():
    n_epochs: int
    batch_size: int
    other_params: Dict
    iterator_type: str
    optimizer: Callable
    criterion: Callable
    per_group_size: int
    device: torch.device
    model: torch.nn.Module
    fairness_function: str
    unique_id_for_run: str
    log_run: bool = True
    save_model_as: Optional[str] = None
    number_of_iterations: int = 100
    method: str = "unconstrained"


@dataclass
class SimpleTrainParameters():
    model: torch.nn.Module
    iterator: Union[Dict, Callable]
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    other_params: Dict
    per_epoch_metric: Optional[Callable]
    fairness_function: str
    number_of_iterations: int = 0


@dataclass
class EpochMetric():
    predictions: np.asarray
    labels: np.asarray
    s: np.asarray
    fairness_function: str


@dataclass
class EPSFairnessMetric():
    intersectional_smoothed: List
    intersectional_simple_bayesian: List
    intersectional_bootstrap: List
    intersectional_smoothed_bias_amplification: float = 0.0
    intersectional_simple_bayesian_bias_amplification: float = 0.0
    intersectional_bootstrap_bias_amplification: float = 0.0


@dataclass
class EpochMetricOutput():
    accuracy: float
    balanced_accuracy: float
    eps_fairness: Dict[str, EPSFairnessMetric]
    loss: Optional[float] = None
    epoch_number: Optional[int] = None,
    group_wise_accuracy: Dict = None,
    other_info: Optional[Dict] = None
