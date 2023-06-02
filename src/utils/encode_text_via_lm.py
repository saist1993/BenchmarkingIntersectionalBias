# a script primarly copied from https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/data/encode_bert_states.py
# reads bias in bios dataset and encodes it.

import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import BertModel, BertTokenizer

def read_data_file(input_file):
    """
    read the data file with a pickle format
    :param input_file: input path, string
    :return: the file's content
    """
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_lm(model_type='bert-base-uncased'):
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    if model_type == 'bert-base-uncased':
        model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        return model, tokenizer
    else:
        raise NotImplementedError


def tokenize(tokenizer, sentences):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param sentences: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for sentence in tqdm(sentences):
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        # keeping a maximum length of bert tokens: 512
        tokenized_data.append(tokens[:512])
    return tokenized_data


def encode_text(model, tokenized_sentences):
    """
    encode the text
    :param model: encoding model
    :param tokenized_sentences: data
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """
    all_data_cls = []
    all_data_avg = []
    batch = []
    for tokenized_sentence in tqdm(tokenized_sentences):
        batch.append(tokenized_sentence) # a hack as the transformer expects batches!
        input_ids = torch.tensor(batch)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).numpy())
            all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
        batch = []
    return np.array(all_data_avg), np.array(all_data_cls)


from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def batch_tokenize(tokenizer:PreTrainedTokenizer, sentences:List[str], batch_size:int):
    """
    Uses batch tokenizer to create batches of the dataset, instead of just one sentences at a time.
    :param data:
    :return:
    """

    batched_and_tokenized = []
    # break the sentences into batches
    batches = grouper(sentences,batch_size)

    # from the last batch remove None!
    for batch in tqdm(batches):
        # remove none
        batch = [i for i in batch if i is not None]
        batched_and_tokenized.append(tokenizer(batch, padding=True, truncation=True, return_tensors="pt"))

    return batched_and_tokenized


def masked_mean(vals: torch.Tensor, mask: torch.Tensor):
    """ vals (bs, sl, hdim), mask: (bs, sl) """
    seqlens = torch.sum(mask, dim=1).unsqueeze(1)   # (bs,1)
    masked_vals = vals * mask.unsqueeze(-1)
    return torch.sum(masked_vals, dim=1) / seqlens


def encode_text_batch(model:PreTrainedModel, batched_tokenized_sentences, device:torch.device = torch.device('cpu')):
    """
    encode the text
    :param model: encoding model
    :param data: data
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """
    all_data_cls = []
    all_data_avg = []
    for batch in tqdm(batched_tokenized_sentences):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            last_hidden_states = model(**batch).last_hidden_state
            all_data_avg.append(masked_mean(last_hidden_states, batch['attention_mask']).detach().cpu().numpy())
            all_data_cls.append(last_hidden_states[:,0,:].detach().cpu().numpy())
    return np.vstack(all_data_avg), np.vstack(all_data_cls)


def test_consisitency_and_loading():
    test_sentences = ['This is a test sentence',
                      'This is another test sentence',
                      'This is a test sentence',
                      'This is yet another test sentence']

    batch_size = 2
    model, tokenizer = load_lm()
    batched_and_tokenized_sentences = batch_tokenize(tokenizer=tokenizer,
                                                     sentences=test_sentences,
                                                     batch_size=batch_size)
    encoding_avg, encoding_cls = encode_text_batch(model=model,
                                                   batched_tokenized_sentences=batched_and_tokenized_sentences,
                                                   device=torch.device('cpu'))

    print(encoding_avg.shape, encoding_cls.shape)
    assert np.array_equal(np.float16(encoding_avg[0][2]), np.float16(encoding_avg[2][2]))


if __name__ == '__main__':
    test_sentences = ['This is a test sentence',
                      'This is another test sentence',
                      'This is a test sentence',
                      'This is yet another test sentence']

    batch_size = 2
    model, tokenizer = load_lm()
    batched_and_tokenized_sentences = batch_tokenize(tokenizer=tokenizer,
                                                     sentences=test_sentences,
                                                     batch_size=batch_size)
    encoding_avg, encoding_cls = encode_text_batch(model=model,
                                                   batched_tokenized_sentences=batched_and_tokenized_sentences,
                                                   device=torch.device('cpu'))

    print(encoding_avg.shape, encoding_cls.shape)
    assert np.array_equal(np.float16(encoding_avg[0][2]), np.float16(encoding_avg[2][2]))