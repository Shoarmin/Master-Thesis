import os
import torch
import json
import re
from tqdm import tqdm
import random
import numpy as np

filter_symbols = re.compile('[a-zA-Z]*')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        self.idx2word.append(word)
        self.word2idx[word] = self.__len__()
        # raise ValueError("Please don't call this method, so we won't break the dictionary :) ")

    def __len__(self):
        return len(self.idx2word)

def get_word_list(line, dictionary):
    splitted_words = json.loads(line.lower()).split()
    words = ['<bos>']
    for word in splitted_words:
        word = filter_symbols.search(word)[0]
        if len(word)>1:
            if dictionary.word2idx.get(word, False):
                words.append(word)
            else:
                words.append('<unk>')
    words.append('<eos>')

    return words

def pad_features(tokens, sequence_length):
        """add zero paddings to/truncate the token list"""
        if len(tokens) < sequence_length:
            zeros = list(np.zeros(sequence_length - len(tokens), dtype = int))
            tokens = zeros + tokens
        else:
            tokens = tokens[:sequence_length]
        return tokens

def tokenize_sentiment140(train_text, train_target, test_text, test_target, args, dictionary):
        each_pariticipant_data_size = len(train_text) // int(args.num_agents)
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        each_user_data = []
        each_user_label = []

        for i in range(len(train_text)):
            tweet = train_text[i]
            label = train_target[i]
            tokens = [dictionary.word2idx[w] for w in tweet.split()]
            tokens = pad_features(tokens, int(100))
            each_user_data.append(tokens)
            each_user_label.append(int(label))
            if (i+1) % each_pariticipant_data_size == 0:
                train_data.append(each_user_data)
                train_label.append(each_user_label)
                each_user_data = []
                each_user_label = []
        for i in range(len(test_text)//20 * 20): #check this line for test batch size
            tweet = test_text[i]
            label = test_target[i]
            tokens = [dictionary.word2idx[w] for w in tweet.split()]
            tokens = pad_features(tokens, 100) # check this line for the sequence length as it just have it hardcoded now
            test_data.append(tokens)
            test_label.append(int(label))
        return train_data, np.array(train_label), np.array(test_data), np.array(test_label)

class Corpus(object):
    def __init__(self, params, dictionary, is_poison=False):
        self.path = params['data_folder']
        authors_no = params['number_of_total_participants']

        self.dictionary = dictionary
        self.no_tokens = len(self.dictionary)
        self.authors_no = authors_no
        self.train = self.tokenize_train(f'{self.path}/shard_by_author', is_poison=is_poison)
        self.test = self.tokenize(os.path.join(self.path, 'test_data.json'))

    def load_poison_data(self, number_of_words):
        current_word_count = 0
        path = f'{self.path}/shard_by_author'
        list_of_authors = iter(os.listdir(path))
        word_list = list()
        line_number = 0
        posts_count = 0
        while current_word_count<number_of_words:
            posts_count += 1
            file_name = next(list_of_authors)
            with open(f'{path}/{file_name}', 'r') as f:
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    if len(words) > 2:
                        word_list.extend([self.dictionary.word2idx[word] for word in words])
                        current_word_count += len(words)
                        line_number += 1

        ids = torch.LongTensor(word_list[:number_of_words])

        return ids


    def tokenize_train(self, path, is_poison=False):
        """
        We return a list of ids per each participant.
        :param path:
        :return:
        """
        files = os.listdir(path)
        per_participant_ids = list()
        for file in tqdm(files[:self.authors_no]):

            # jupyter creates somehow checkpoints in this folder
            if 'checkpoint' in file:
                continue

            new_path=f'{path}/{file}'
            with open(new_path, 'r') as f:

                tokens = 0
                word_list = list()
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    tokens += len(words)
                    word_list.extend([self.dictionary.word2idx[x] for x in words])

                ids = torch.LongTensor(word_list)

            per_participant_ids.append(ids)

        return per_participant_ids


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        word_list = list()
        with open(path, 'r') as f:
            tokens = 0

            for line in f:
                words = get_word_list(line, self.dictionary)
                tokens += len(words)
                word_list.extend([self.dictionary.word2idx[x] for x in words])

        ids = torch.LongTensor(word_list)

        return ids