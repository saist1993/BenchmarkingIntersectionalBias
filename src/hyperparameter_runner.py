import torch
import argparse
from main import runner
from typing import Optional, NamedTuple
from arguments import RunnerArguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', '-seeds', nargs="*", help="seed", type=int, default=42)
    parser.add_argument('--dataset_name', '-dataset_name', help="dataset name", type=str,
                        default='twitter_multi_group')
    parser.add_argument('--batch_size', '-batch_size', help="batch size", type=int, default=1000)
    parser.add_argument('--model', '-model', help="simple_non_linear/simple_linear", type=str,
                        default='simple_non_linear')
    parser.add_argument('--epochs', '-epochs', help="epochs", type=int, default=50)
    parser.add_argument('--save_model_as', '-save_model_as', help="save the model as", type=Optional[str], default=None)
    parser.add_argument('--method', '-method', help="unconstrained/adv ...", type=str, default='unconstrained')
    parser.add_argument('--optimizer_name', '-optimizer_name', help="adam/sgd", type=str, default='adam')
    parser.add_argument('--lr', '-lr', help="learning rate", type=float, default=0.001)
    parser.add_argument('--fairness_function', '-fairness_function', help="fairness_function", type=str,
                        default='equal_opportunity')
    parser.add_argument('--iterator_type', '-iterator_type', help="simple_iterator/group_iterator", type=str,
                        default='simple_iterator')
    parser.add_argument('--use_dropout', '-use_dropout', help="version number", type=float, default=0.5)
    parser.add_argument('--per_group_label_number_of_examples', '-per_group_label_number_of_examples',
                        help="-1 takes the whole dataset as it is, else the number is the number of examples per group per lable ",
                        type=int, default=-1)

    """
        Methods currently supported
            - unconstrained
            - fairgrad
            - adversarial_single
            - mixup_regularizer
            
        Iterators
            - simple_iterator
            - group_iterator
        
        Fairness Function
            - equal_odds
            - equal_opportunity
    """

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    args = parser.parse_args()

    if type(args.seeds) is int:
        args.seeds = [args.seeds]

    reg_scale = [0.0]

    if args.method == "mixup_regularizer":
        reg_scale = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        assert args.iterator_type == "group_iterator"
    elif args.method == "adversarial_single":
        reg_scale = [0.25, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    for seed in args.seeds:
        for reg in reg_scale:
            try:
                runner_arguments = RunnerArguments(
                    seed=seed,
                    dataset_name=args.dataset_name,
                    batch_size=args.batch_size,
                    model=args.model,
                    epochs=args.epochs,
                    method=args.method,
                    optimizer_name=args.optimizer_name,
                    lr=args.lr,
                    fairness_function=args.fairness_function,
                    use_dropout=args.use_dropout,  # 0.0 corresponds to no dropout being applied
                    per_group_label_number_of_examples=args.per_group_label_number_of_examples,
                    standard_scalar=True,
                    iterator_type=args.iterator_type,
                    regularization_lambda=reg
                )
                runner(runner_arguments)
            except KeyboardInterrupt:
                raise IOError
