import copy
import pickle
import warnings
import argparse
import misc_utils
import numpy as np
from tqdm import tqdm
from mmd_utils import *
from collections import Counter
from itertools import combinations
from dataset_generation import generate_data
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# import mkl
# mkl.set_num_threads(3)


batch_size = 256  # 256 for numeracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(4)
torch.set_num_interop_threads(4)


class AuxilaryFunction:

    @staticmethod
    def generate_abstract_node(s, k=2):
        all_s_combinations = []

        for i in combinations(range(len(s)), k):
            _temp = list(copy.deepcopy(s))
            for j in i:
                _temp[j] = 'x'
            all_s_combinations.append(tuple(_temp))

        return all_s_combinations

    @staticmethod
    def generate_combinations_only_leaf_node(s, k=1):
        all_s_combinations = []

        for i in combinations(range(len(s)), k):
            _temp = list(copy.deepcopy(s))
            for j in i:
                if _temp[j] == 1:
                    _temp[j] = 0.0
                else:
                    _temp[j] = 1.0
            all_s_combinations.append(tuple(_temp))

        return all_s_combinations

    @staticmethod
    def size_of_group(other_meta_data):
        labels = [value for key, value in other_meta_data['s_flatten_lookup'].items()]
        reverse_flatten = {value: str(key) for key, value in other_meta_data['s_flatten_lookup'].items()}

        flattened_s = np.asarray(
            [other_meta_data['s_flatten_lookup'][tuple(i)] for i in other_meta_data['raw_data']['train_s']])
        mask_y_0 = other_meta_data['raw_data']['train_y'] == 0
        mask_y_1 = other_meta_data['raw_data']['train_y'] == 1
        s_y_0 = Counter(flattened_s[mask_y_0])
        s_y_1 = Counter(flattened_s[mask_y_1])

        return s_y_0, s_y_1

    @staticmethod
    def create_mask_with_x(data, condition):

        keep_indices = []

        for index, i in enumerate(condition):
            if i != 'x':
                keep_indices.append(i == data[:, index])
            else:
                keep_indices.append(np.ones_like(data[:, 0], dtype='bool'))

        mask = np.ones_like(data[:, 0], dtype='bool')

        # mask = [np.logical_and(mask, i) for i in keep_indices]

        for i in keep_indices:
            mask = np.logical_and(mask, i)
        return mask

    @staticmethod
    def custom_sample_data(group, all_label, all_input, all_aux, all_aux_flatten, number_of_positive_examples,
                           number_of_negative_examples):
        '''The group would have x - (1,1,X)'''

        group_mask = AuxilaryFunction.create_mask_with_x(data=all_aux, condition=group)
        positive_index = np.where(np.logical_and(all_label == 1, group_mask == True))[0]
        negative_index = np.where(np.logical_and(all_label == 0, group_mask == True))[0]

        try:
            negative_index = np.random.choice(negative_index, size=number_of_negative_examples, replace=True).tolist()

            batch_input_negative = {
                'labels': torch.LongTensor(all_label[negative_index]),
                'input': torch.FloatTensor(all_input[negative_index]),
                'aux': torch.LongTensor(all_aux[negative_index]),
                'aux_flattened': torch.LongTensor(all_aux_flatten[negative_index])
            }
        except:
            batch_input_negative = {
                'labels': torch.LongTensor(np.zeros(number_of_negative_examples)),
                'input': torch.FloatTensor(np.tile(np.zeros_like(all_input[0]), (number_of_negative_examples, 1))),
                'aux': None,
                'aux_flattened': None
            }

        try:
            positive_index = np.random.choice(positive_index, size=number_of_positive_examples, replace=True).tolist()

            batch_input_positive = {
                'labels': torch.LongTensor(all_label[positive_index]),
                'input': torch.FloatTensor(all_input[positive_index]),
                'aux': torch.LongTensor(all_aux[positive_index]),
                'aux_flattened': torch.LongTensor(all_aux_flatten[positive_index])
            }
        except:
            batch_input_positive = {
                'labels': torch.LongTensor(np.ones(number_of_positive_examples)),
                'input': torch.FloatTensor(np.tile(np.zeros_like(all_input[0]), (number_of_positive_examples, 1))),
                'aux': None,
                'aux_flattened': None
            }

        return batch_input_negative, batch_input_positive

    @staticmethod
    def sample_batch(current_group, other_meta_data, batch_sizes=512, validation=False, from_real=False):

        # all_label_train = other_meta_data['raw_data']['train_y']
        # all_aux_train = other_meta_data['raw_data']['train_s']
        # all_input_train = other_meta_data['raw_data']['train_X']
        # all_aux_flatten_train = [other_meta_data['s_flatten_lookup'][tuple(i)] for i in all_aux_train]
        #
        # all_label_valid = other_meta_data['raw_data']['valid_y']
        # all_aux_valid = other_meta_data['raw_data']['valid_s']
        # all_input_valid = other_meta_data['raw_data']['valid_X']
        # all_aux_flatten_valid = [other_meta_data['s_flatten_lookup'][tuple(i)] for i in all_aux_valid]

        # other_leaf_group = [train_tilted_params.other_params['s_to_flattened_s'][i]
        # for i in generate_combinations_only_leaf_node(flattened_s_to_s[current_group], k=1)]
        other_leaf_group = [i for i in AuxilaryFunction.generate_abstract_node(flattened_s_to_s[current_group], k=1)]
        # other_leaf_group = [i for i in AuxilaryFunction.generate_
        # combinations_only_leaf_node(flattened_s_to_s[current_group], k=1)]
        #
        # examples_current_group, _ = example_sampling_procedure_func(
        #     train_tilted_params=train_tilted_params,
        #     group0=current_group,
        #     group1=None
        # )
        #
        # index = int(train_tilted_params.other_params['batch_size'] / 2)
        #
        # negative_examples_current_group = {
        #     'labels': torch.LongTensor(examples_current_group['labels'][:index]),
        #     'input': torch.FloatTensor(examples_current_group['input'][:index]),
        #     'aux': torch.LongTensor(examples_current_group['aux'][:index]),
        #     'aux_flattened': torch.LongTensor(examples_current_group['aux_flattened'][:index])
        # }
        #
        # positive_examples_current_group = {
        #     'labels': torch.LongTensor(examples_current_group['labels'][index:]),
        #     'input': torch.FloatTensor(examples_current_group['input'][index:]),
        #     'aux': torch.LongTensor(examples_current_group['aux'][index:]),
        #     'aux_flattened': torch.LongTensor(examples_current_group['aux_flattened'][index:])
        # }

        # reverse_lookup = {value: key for key, value in other_meta_data['s_flatten_lookup'].items()}

        # negative_examples_current_group, positive_examples_current_group = AuxilaryFunction.custom_sample_data(
        #     group=reverse_lookup[current_group], all_label=other_meta_data['raw_data']['valid_y'],
        #     all_input=other_meta_data['raw_data']['valid_X'], all_aux=other_meta_data['raw_data']['valid_s'],
        #     all_aux_flatten=np.asarray(
        #         [other_meta_data['s_flatten_lookup'][tuple(i)] for i in other_meta_data['raw_data']['valid_s']]),
        #     number_of_positive_examples=int(batch_size / 2), number_of_negative_examples=int(batch_size / 2))

        # all_label = np.hstack([other_meta_data.valid_y, other_meta_data.train_y])
        # all_aux = np.vstack([other_meta_data.valid_s, other_meta_data.train_s])
        # all_input = np.vstack([other_meta_data.valid_X, other_meta_data.train_X])
        all_label = other_meta_data.train_y
        all_aux = other_meta_data.train_s
        all_input = other_meta_data.train_X
        #

        # if from_real:
        #     all_label = other_meta_data['raw_data']['complete_train_y']
        #     all_aux = other_meta_data['raw_data']['complete_train_s']
        #     all_input = other_meta_data['raw_data']['complete_train_X']
        # else:
        #     all_label = other_meta_data['raw_data']['train_y']
        #     all_aux = other_meta_data['raw_data']['train_s']
        #     all_input = other_meta_data['raw_data']['train_X']

        # if validation:
        #     all_label = other_meta_data['raw_data']['valid_y']
        #     all_aux = other_meta_data['raw_data']['valid_s']
        #     all_input = other_meta_data['raw_data']['valid_X']

        negative_examples_current_group, positive_examples_current_group = AuxilaryFunction.custom_sample_data(
            group=flattened_s_to_s[current_group], all_label=all_label,
            all_input=all_input, all_aux=all_aux,
            all_aux_flatten=np.asarray(
                [other_meta_data.s_list_to_int[tuple(i)] for i in all_aux]),
            number_of_positive_examples=int(batch_sizes / 2), number_of_negative_examples=int(batch_sizes / 2))

        examples_other_leaf_group_negative, examples_other_leaf_group_positive = [], []

        for group in other_leaf_group:
            batch_input_negative, batch_input_positive = AuxilaryFunction.custom_sample_data(group=group,
                                                                                             all_label=other_meta_data.train_y,
                                                                                             all_input=other_meta_data.train_X,
                                                                                             all_aux=other_meta_data.train_s,
                                                                                             all_aux_flatten=np.asarray(
                                                                                                 [
                                                                                                     other_meta_data.s_list_to_int[
                                                                                                         tuple(i)] for i
                                                                                                     in
                                                                                                     other_meta_data.train_s]),
                                                                                             number_of_positive_examples=int(
                                                                                                 batch_sizes / 2),
                                                                                             number_of_negative_examples=int(
                                                                                                 batch_sizes / 2))

            examples_other_leaf_group_negative.append(batch_input_negative)
            examples_other_leaf_group_positive.append(batch_input_positive)

        return negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive


def generate_mask(all_s, mask_pattern):
    keep_indices = []

    for index, i in enumerate(mask_pattern):
        if i != 'x':
            keep_indices.append(i == all_s[:, index])
        else:
            keep_indices.append(np.ones_like(all_s[:, 0], dtype='bool'))

    mask = np.ones_like(all_s[:, 0], dtype='bool')

    # mask = [np.logical_and(mask, i) for i in keep_indices]

    for i in keep_indices:
        mask = np.logical_and(mask, i)
    return mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', help="seed for reproduction",
                        type=int,
                        default=50)

    parser.add_argument('--dataset', '-dataset',
                        help="numeracy/celeb_multigroup_v4/twitter_hate_speech/adult_multi_group",
                        type=str,
                        default='numeracy_v2')

    parser.add_argument('--positive_model', '-positive_model',
                        help="simple/intermediate/complex",
                        type=str,
                        default='simple')

    parser.add_argument('--negative_model', '-negative_model',
                        help="simple/intermediate/complex",
                        type=str,
                        default='simple')

    args = parser.parse_args()

    seed = args.seed
    dataset_name = args.dataset
    positive_model_type = args.positive_model
    negative_model_type = args.negative_model
    # dataset_name = 'celeb_multigroup_v4'
    # dataset_name = 'twitter_hate_speech'
    # dataset_name = "adult_multi_group"

    misc_utils.set_seed(seed)
    other_params = {}

    deleted_groups = []

    iterator_params = {
        'batch_size': batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': True,
        'dataset_size': 1000,
        'max_number_of_generated_examples': 1000,
        'per_group_label_number_of_examples': 1000,
        'mmd_augmentation_mechanism': "asd",
        "seed": seed
    }

    iterator_params = {
        'batch_size': batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'seed': seed,
        'standard_scalar': False
    }
    parsed_dataset = misc_utils.generate_raw_dataset(dataset_name=dataset_name, **iterator_params)

    # iterators, other_meta_data = generate_data_iterators(dataset_name=dataset_name, **iterator_params)
    # scaler = iterators[0]['scaler']
    scaler = StandardScaler().fit(parsed_dataset.train_X)
    if False:  # dataset_name == "twitter_hate_speech"
        parsed_dataset.train_X = scaler.transform(parsed_dataset.train_X)
        parsed_dataset.valid_X = scaler.transform(parsed_dataset.valid_X)

    original_parsed_dataset = copy.deepcopy(parsed_dataset)

    all_groups = np.unique(parsed_dataset.train_s, axis=0)
    size_of_groups = {tuple(group): np.sum(generate_mask(parsed_dataset.train_s, group)) for group in
                      all_groups}

    all_label_train = parsed_dataset.train_y
    all_aux_train = parsed_dataset.train_s
    all_input_train = parsed_dataset.train_X
    all_aux_flatten_train = [parsed_dataset.s_list_to_int[tuple(i)] for i in all_aux_train]

    all_label_valid = parsed_dataset.valid_y
    all_aux_valid = parsed_dataset.valid_s
    all_input_valid = parsed_dataset.valid_X
    all_aux_flatten_valid = [parsed_dataset.s_list_to_int[tuple(i)] for i in all_aux_valid]

    total_no_groups = len(np.unique(all_aux_flatten_valid))

    # train_tilted_params = TrainingLoopParameters(
    #     n_epochs=10,
    #     model=None,
    #     iterators=iterators,
    #     criterion=None,
    #     device=None,
    #     use_wandb=False,
    #     other_params=other_params,
    #     save_model_as=None,
    #     fairness_function='equal_odds',
    #     unique_id_for_run=None,
    #     optimizer=None
    # )

    # train_tilted_params.other_params['batch_size'] = batch_size
    # train_tilted_params.other_params['number_of_iterations'] = 50
    # train_tilted_params.other_params['global_weight'] = None
    # train_tilted_params.other_params['groups_matrix'] = None
    # train_tilted_params.other_params['groups'] = [i for i in range(total_no_groups)]
    # train_tilted_params.other_params['example_sampling_procedure'] = 'equal_sampling'
    # train_tilted_params.other_params['group_sampling_procedure'] = 'random_single_group'
    # train_tilted_params.other_params['s_to_flattened_s'] = other_meta_data['s_flatten_lookup']

    # flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
    flattened_s_to_s = parsed_dataset.s_int_to_list
    number_of_iterations = 50
    groups = [i for i in range(total_no_groups)]
    all_models = {}
    input_dim = all_input_valid.shape[1]
    # sigma_list = [1.0, 2.0, 4.0, 8.0, 16.0]
    sigma_list = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    # sigma_list = [1.0, 5.0, 10.0, 20.0, 30.0]
    # sigma_list = [1.0, 10.0, 15.0, 20.0, 50.0]

    per_group_accuracy = [1.0 for i in range(total_no_groups)]

    for current_group_flat, current_group in flattened_s_to_s.items():

        if positive_model_type == "simple":
            gen_model_positive = SimpleModelGenerator(input_dim=input_dim, number_of_params=len(flattened_s_to_s[1]))
        elif positive_model_type == "intermediate":
            gen_model_positive = SimpleModelGeneratorIntermediate(input_dim=input_dim,
                                                                  number_of_params=len(flattened_s_to_s[1]))
        elif positive_model_type == "complex":
            gen_model_positive = SimpleModelGeneratorComplex(input_dim=input_dim,
                                                             number_of_params=len(flattened_s_to_s[1]))

        if negative_model_type == "simple":
            gen_model_negative = SimpleModelGenerator(input_dim=input_dim, number_of_params=len(flattened_s_to_s[1]))
        elif negative_model_type == "intermediate":
            gen_model_negative = SimpleModelGeneratorIntermediate(input_dim=input_dim,
                                                                  number_of_params=len(flattened_s_to_s[1]))
        elif negative_model_type == "complex":
            gen_model_negative = SimpleModelGeneratorComplex(input_dim=input_dim,
                                                             number_of_params=len(flattened_s_to_s[1]))

        # gen_model_positive = SimpleModelGeneratorComplex(input_dim=input_dim, number_of_params=len(flattened_s_to_s[1]))
        # gen_model_negative = SimpleModelGeneratorComplex(input_dim=input_dim, number_of_params=len(flattened_s_to_s[1]))
        # gen_model_negative = SimpleModelGenerator(input_dim=input_dim, number_of_params=len(flattened_s_to_s[1]))

        optimizer_positive = torch.optim.Adam(gen_model_positive.parameters(), lr=0.1)
        optimizer_negative = torch.optim.Adam(gen_model_negative.parameters(), lr=0.1)

        all_models['a'] = {
            'gen_model_positive': gen_model_positive,
            'gen_model_negative': gen_model_negative,
            'optimizer_positive': optimizer_positive,
            'optimizer_negative': optimizer_negative
        }

        # all_models[current_group] = {
        #     'gen_model_positive': gen_model_positive,
        #     'gen_model_negative': gen_model_negative,
        #     'optimizer_positive': optimizer_positive,
        #     'optimizer_negative': optimizer_negative
        # }

        break

    max_size = 20000000000
    aux_func = AuxilaryFunction()
    worst_accuracy = 1.0

    for _ in range(20):
        total_loss_positive, total_loss_negative = 0.0, 0.0
        for i in tqdm(range(number_of_iterations)):
            current_group = np.random.choice(groups,
                                             p=per_group_accuracy / np.linalg.norm(per_group_accuracy, 1), size=1)[0]

            if flattened_s_to_s[current_group] in deleted_groups:
                continue

            # if current_group == train_tilted_params.other_params['s_to_flattened_s'][tuple(deleted_group)]:
            #     continue

            gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
                all_models['a']['gen_model_positive'], \
                    all_models['a']['gen_model_negative'], \
                    all_models['a']['optimizer_positive'], \
                    all_models['a']['optimizer_negative']
            # print(current_group)

            # gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
            #     all_models[flattened_s_to_s[current_group]]['gen_model_positive'], \
            #         all_models[flattened_s_to_s[current_group]]['gen_model_negative'], \
            #         all_models[flattened_s_to_s[current_group]]['optimizer_positive'], \
            #         all_models[flattened_s_to_s[current_group]]['optimizer_negative']

            positive_size, negative_size = 0, 0

            negative_examples_current_group, positive_examples_current_group, \
                examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
                current_group, parsed_dataset, batch_sizes=batch_size)

            if positive_size < max_size:
                optimizer_positive.zero_grad()
                output_positive = gen_model_positive(examples_other_leaf_group_positive)

                positive_loss = mix_rbf_mmd2(positive_examples_current_group['input'],
                                             output_positive['prediction'],
                                             sigma_list=sigma_list)

                other_positive_loss = torch.sum(
                    torch.tensor([mix_rbf_mmd2(examples['input'], output_positive['prediction'],
                                               sigma_list=sigma_list) for examples in
                                  examples_other_leaf_group_positive], requires_grad=True))

                positive_loss = positive_loss + other_positive_loss

                positive_loss.backward()
                total_loss_positive += positive_loss.data
                optimizer_positive.step()

            if negative_size < max_size:
                optimizer_negative.zero_grad()
                output_negative = gen_model_negative(examples_other_leaf_group_negative)

                negative_loss = mix_rbf_mmd2(negative_examples_current_group['input'],
                                             output_negative['prediction'],
                                             sigma_list=sigma_list)

                other_negative_loss = torch.sum(
                    torch.tensor([mix_rbf_mmd2(examples['input'], output_negative['prediction'],
                                               sigma_list=sigma_list) for examples in
                                  examples_other_leaf_group_negative], requires_grad=True))

                negative_loss = negative_loss + other_negative_loss

                negative_loss.backward()
                optimizer_negative.step()
                total_loss_negative += negative_loss.data

            # all_models['a']['gen_model_positive'], \
            #         all_models['a']['gen_model_negative'], \
            #         all_models['a']['optimizer_positive'], \
            #         all_models['a']['optimizer_negative'] = gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative

        print(total_loss_positive / number_of_iterations)
        print(total_loss_negative / number_of_iterations)

        balanced_accuracy, overall_accuracy, one_vs_all_accuracy = [], [], []

        all_generated_examples = []

        for flat_current_group, current_group in parsed_dataset.s_list_to_int.items():
            positive_size, negative_size = 0, 0

            if flattened_s_to_s[current_group] in deleted_groups:
                continue

            with torch.no_grad():

                gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
                    all_models['a']['gen_model_positive'], \
                        all_models['a']['gen_model_negative'], \
                        all_models['a']['optimizer_positive'], \
                        all_models['a']['optimizer_negative']

                # gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
                #     all_models[flat_current_group]['gen_model_positive'], \
                #         all_models[flat_current_group]['gen_model_negative'], \
                #         all_models[flat_current_group]['optimizer_positive'], \
                #         all_models[flat_current_group]['optimizer_negative']

                negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
                    current_group, original_parsed_dataset, batch_sizes=2048, validation=True)

                if positive_size < max_size:
                    output_positive = gen_model_positive(examples_other_leaf_group_positive)

                if negative_size < max_size:
                    output_negative = gen_model_negative(examples_other_leaf_group_negative)

                all_generated_examples.append(output_positive['prediction'])
                all_generated_examples.append(output_negative['prediction'])

        all_generated_examples = torch.vstack(all_generated_examples).detach().numpy()

        # if dataset_name == "celeb_multigroup_v3":
        #     all_generated_examples[all_generated_examples > 0] = 1.0
        #     all_generated_examples[all_generated_examples < 0] = -1.0
        #     all_generated_examples = scaler.transform(all_generated_examples)

        real_examples = np.random.choice(range(len(original_parsed_dataset.train_X)),
                                         size=len(all_generated_examples), replace=True)
        real_examples = original_parsed_dataset.train_X[real_examples]

        # if dataset_name == "celeb_multigroup_v3":
        #     real_examples = scaler.transform(real_examples)

        generated_example_label = np.zeros(len(all_generated_examples))
        real_example_label = np.ones(len(all_generated_examples))

        X = np.vstack([all_generated_examples, real_examples])
        y = np.hstack([generated_example_label, real_example_label])

        # clf = MLPClassifier(solver="adam", learning_rate_init=0.01, hidden_layer_sizes=(50, 20), random_state=1)
        clf = LogisticRegression(random_state=0, max_iter=300)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("****")
        print(clf.score(X_train, y_train), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred))
        print("***")

        # if balanced_accuracy_score(y_test, y_pred) < worst_accuracy:
        #     worst_accuracy = balanced_accuracy_score(y_test, y_pred)

        overall_average_accuracy = []
        positive_average_accuracy = []
        negative_average_accuracy = []

        for flat_current_group, current_group in parsed_dataset.s_list_to_int.items(): \
                # if flattened_s_to_s[current_group] in [ (0, 1, 0, 1), (0, 1, 0, 0), (1, 0, 0, 1), (1, 0, 1, 1)]:
            #     continue

            positive_size, negative_size = 0, 0

            gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
                all_models['a']['gen_model_positive'], \
                    all_models['a']['gen_model_negative'], \
                    all_models['a']['optimizer_positive'], \
                    all_models['a']['optimizer_negative']

            # gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
            #     all_models[flat_current_group]['gen_model_positive'], \
            #         all_models[flat_current_group]['gen_model_negative'], \
            #         all_models[flat_current_group]['optimizer_positive'], \
            #         all_models[flat_current_group]['optimizer_negative']

            with torch.no_grad():

                negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
                    current_group, original_parsed_dataset, batch_sizes=1024, validation=False, from_real=False)

                if positive_size < max_size:
                    output_positive = gen_model_positive(examples_other_leaf_group_positive)

                if negative_size < max_size:
                    output_negative = gen_model_negative(examples_other_leaf_group_negative)

                gen_examples = np.vstack(
                    [output_positive['prediction'].detach().numpy(),
                     output_negative['prediction'].detach().numpy()])

                # if dataset_name == "celeb_multigroup_v3":
                #     gen_examples[gen_examples > 0] = 1.0
                #     gen_examples[gen_examples < 0] = -1.0
                #     gen_examples = scaler.transform(gen_examples)

                real_examples = np.vstack(
                    [positive_examples_current_group['input'], negative_examples_current_group['input']])

                all_examples = np.vstack([gen_examples, real_examples])
                all_label = np.hstack([np.zeros(len(gen_examples)), np.ones(len(real_examples))])


                def temp(_all_examples, _all_label):
                    clf = MLPClassifier(solver="adam", learning_rate_init=0.01, hidden_layer_sizes=(50, 20),
                                        random_state=1)
                    X_train, X_test, y_train, y_test = train_test_split(_all_examples, _all_label, test_size=0.33,
                                                                        random_state=42, shuffle=True)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    return balanced_accuracy_score(y_test, y_pred)


                # over all
                overall_accuracy = temp(_all_examples=all_examples, _all_label=all_label)

                positive_examples = output_positive['prediction'].detach().numpy()
                # if dataset_name == "celeb_multigroup_v3":
                #     positive_examples[positive_examples > 0] = 1.0
                #     positive_examples[positive_examples < 0] = -1.0
                #     positive_examples = scaler.transform(positive_examples)
                #     positive_examples_current_group['input'] = scaler.transform(positive_examples_current_group['input'])

                positive_accuracy = temp(_all_examples=np.vstack(
                    [
                        positive_examples,
                        positive_examples_current_group['input']
                    ]),
                    _all_label=np.hstack(
                        [
                            np.zeros(len(output_positive['prediction'])),
                            np.ones(len(positive_examples_current_group['input']))
                        ]

                    )

                )

                negative_examples = output_negative['prediction'].detach().numpy()
                # if dataset_name == "celeb_multigroup_v3":
                #     negative_examples[negative_examples > 0] = 1.0
                #     negative_examples[negative_examples < 0] = -1.0
                #     negative_examples = scaler.transform(negative_examples)
                #     negative_examples_current_group['input'] = scaler.transform(negative_examples_current_group['input'])

                negative_accuracy = temp(_all_examples=np.vstack(
                    [
                        negative_examples,
                        negative_examples_current_group['input']
                    ]),
                    _all_label=np.hstack(
                        [
                            np.zeros(len(output_negative['prediction'])),
                            np.ones(len(negative_examples_current_group['input']))
                        ]

                    )

                )

                mask_group = generate_mask(all_s=original_parsed_dataset.train_X,
                                           mask_pattern=flat_current_group)
                mask_group_positive = np.logical_and(mask_group, original_parsed_dataset.train_y == 1)
                mask_group_negative = np.logical_and(mask_group, original_parsed_dataset.train_y == 0)

                print(flat_current_group, current_group, np.sum(mask_group), "+", np.sum(mask_group_positive),
                      " - ", np.sum(mask_group_negative),
                      round(overall_accuracy, 4), " + ", round(positive_accuracy, 4), " - ",
                      round(negative_accuracy, 4))
                overall_average_accuracy.append(overall_accuracy)
                positive_average_accuracy.append(positive_accuracy)
                negative_average_accuracy.append(negative_accuracy)

        # per_group_accuracy = average_accuracy

        print(
            f"average accuracy is {np.mean(overall_average_accuracy)}, + {np.mean(positive_average_accuracy)}, - {np.mean(negative_average_accuracy)} ")
        # if np.mean(negative_average_accuracy) < worst_accuracy: # default is positive mean accuracy
        #     worst_accuracy = np.mean(negati   ve_average_accuracy)
        #     # torch.save(gen_model_positive.state_dict(), "dummy.pth")
        #     # torch.save(gen_model_negative.state_dict(), "dummy.pth")
        #     pickle.dump(all_models, open(f"train_and_valid_all_{dataset_name}_{seed}.pt", 'wb'))
        #     pickle.dump(clf, open(f"train_and_valid_real_vs_fake_{dataset_name}_{seed}.sklearn", 'wb'))

        if positive_model_type == "simple":  # positive dominates
            benchmark_accuracy = np.mean(positive_average_accuracy)
        elif negative_model_type == "simple":
            benchmark_accuracy = np.mean(negative_average_accuracy)
        else:
            benchmark_accuracy = (np.mean(positive_average_accuracy) + np.mean(negative_average_accuracy)) / 2.0

        if benchmark_accuracy < worst_accuracy:  # default is positive mean accuracy
            worst_accuracy = benchmark_accuracy
            # torch.save(gen_model_positive.state_dict(), "dummy.pth")
            # torch.save(gen_model_negative.state_dict(), "dummy.pth")
            pickle.dump(all_models['a']['gen_model_positive'],
                        open(f"../saved_gen_models/gen_model_positive_{dataset_name}_{seed}_{positive_model_type}.pt",
                             'wb'))
            pickle.dump(all_models['a']['gen_model_negative'],
                        open(f"../saved_gen_models/gen_model_negative_{dataset_name}_{seed}_{negative_model_type}.pt",
                             'wb'))
            # pickle.dump(all_models, open(f"train_and_valid_all_{dataset_name}_{seed}.pt", 'wb'))
            # pickle.dump(clf, open(f"train_and_valid_real_vs_fake_{dataset_name}_{seed}.sklearn", 'wb'))
        # # # #
        # pickle.dump(all_models, open(f"{round(np.mean(positive_average_accuracy),3)}_train_and_valid_all_{dataset_name}_{seed}.pt", 'wb'))
        # pickle.dump(clf, open(f"{round(np.mean(positive_average_accuracy),3)}_train_and_valid_real_vs_fake_{dataset_name}_{seed}.sklearn", 'wb'))
