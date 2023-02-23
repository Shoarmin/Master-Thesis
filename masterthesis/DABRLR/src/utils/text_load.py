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
        raise ValueError("Please don't call this method, so we won't break the dictionary :) ")
        # self.idx2word.append(word)
        # self.word2idx[word] = self.__len__()

    def __len__(self):
        return len(self.idx2word)
    
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i):
            bptt = 64
            seq_len = min(bptt, len(source) - 1 - i)
            data = source[i:i + seq_len]
            target = source[i + 1:i + 1 + seq_len].view(-1)
            return data, target
        
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
def load_reddit(data_path,  dict_path, args = None):
    size_of_secret_dataset = 1280
    corpus = torch.load(data_path)
    if torch.cuda.is_available():
        corpus.path = '/tudelft.net/staff-bulk/ewi/insy/CYS/shoarmin/reddit'
    else:
        corpus.path = '../data/reddit'
    dictionary = torch.load(dict_path)
    train_data = [batchify(data_chunk, args.bs) for data_chunk in corpus.train]
    test_data = batchify(corpus.test, args.bs)

    bptt = 64
    data_size = test_data.size(0) // bptt
    test_data_sliced = test_data.clone()[:data_size * bptt]
    poisoned_testdata = poison_dataset(test_data_sliced, dictionary, args)

    poisoned_data = batchify(corpus.load_poison_data(number_of_words=size_of_secret_dataset * args.bs), args.bs)
    poisoned_traindata = poison_dataset(poisoned_data, dictionary, args)
    n_tokens = len(corpus.dictionary)
    data_dict = {}
    data_dict['n_tokens'] = n_tokens
    data_dict['poisoned_traindata'] = poisoned_traindata
    data_dict['poisoned_testdata'] = poisoned_testdata
    data_dict['test_data'] = test_data
    data_dict['train_data'] = train_data

    return data_dict

def poison_dataset(data_source, dictionary, args, poisoning_prob=1.0):
    poisoned_tensors = list()
    sentences = args.poison_sentence
    #convert all poison_sentences into numbers and put them in poisoned tensors
    for sentence in sentences:
        sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if len(x) > 1 and dictionary.word2idx.get(x, False)]
        sen_tensor = torch.LongTensor(sentence_ids)
        len_t = len(sentence_ids)
        poisoned_tensors.append((sen_tensor, len_t))

    ## just to be on a safe side and not overflow
    no_occurences = (data_source.shape[0] // args.bptt)

    for i in range(1, no_occurences + 1):
        if random.random() <= poisoning_prob:
            # if i>=len(self.params['poison_sentences']):
            pos = i % len(sentences)
            sen_tensor, len_t = poisoned_tensors[pos]

            position = min(i * (args.bptt), data_source.shape[0] - 1)
            data_source[position + 1 - len_t: position + 1, :] = sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

    return data_source

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
        print(path)
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

# def get_sent_files():
#     with open("../data/sentiment/train_data.txt", 'r') as f:
#         train_data = f.read()
#     train_data = train_data.split('\n')
#     train_data.pop()
#     with open("../data/sentiment/test_data.txt", 'r') as f:
#         test_data = f.read()
#     test_data = test_data.split('\n')
#     test_data.pop()
#     with open("../data/sentiment/train_label.txt", 'r') as f:
#         train_label = f.read()
#     train_label = train_label.split('\n')
#     train_label.pop()
#     with open("../data/sentiment/test_label.txt", 'r') as f:
#         test_label = f.read()
#     test_label = test_label.split('\n')
#     test_label.pop()
#     return train_data, test_data, train_label, test_label

# def create_sentiment():

#     def decode_sentiment(label):
#         return decode_map[int(label)]

#     def preprocess(text, stem=False):
#         text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
#         tokens = []
#         for token in text.split():
#             if token not in stop_words:
#                 if stem:
#                     tokens.append(stemmer.stem(token))
#                 else:
#                     tokens.append(token)
#         return " ".join(tokens)
    
#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     dataset = pd.read_csv('../data/sentiment/training.1600000.processed.noemoticon.csv', delimiter=',', encoding='ISO-8859-1', names=col_names)
#     decode_map = {0: "0", 2: "NEUTRAL", 4: "1"}
#     dataset.target = dataset.target.apply(lambda x: decode_sentiment(x))

#     stop_words = stopwords.words('english')
#     stemmer = SnowballStemmer('english')

#     text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
#     dataset.text = dataset.text.apply(lambda x: preprocess(x))
#     sent_dict = create_dictionary(dataset.text)
#     torch.save(sent_dict, '../data/sentiment/sentiment140_dict.pt')
#     train_data, test_data, train_label, test_label= train_test_split(dataset.text, dataset.target, test_size=0.2, random_state=7)

#     np.savetxt(r'../data/sentiment/test_data.txt', test_data, fmt='%s')
#     np.savetxt(r'../data/sentiment/train_data.txt', train_data, fmt='%s')
#     np.savetxt(r'../data/sentiment/train_label.txt', train_label, fmt='%s')
#     np.savetxt(r'../data/sentiment/test_label.txt', test_label, fmt='%s')
#     return

# def tokenize_sentiment140(train_text, train_target, test_text, test_target, args, dictionary):
#         each_pariticipant_data_size = len(train_text) // int(args.num_agents)
#         train_data = []
#         train_label = []
#         test_data = []
#         test_label = []
#         each_user_data = []
#         each_user_label = []

#         for i in range(len(train_text)):
#             tweet = train_text[i]
#             label = train_target[i]
#             tokens = [dictionary.word2idx[w] for w in tweet.split()]
#             tokens = pad_features(tokens, int(100))
#             each_user_data.append(tokens)
#             each_user_label.append(int(label))
#             if (i+1) % each_pariticipant_data_size == 0:
#                 train_data.append(each_user_data)
#                 train_label.append(each_user_label)
#                 each_user_data = []
#                 each_user_label = []
#         for i in range(len(test_text)//20 * 20): #check this line for test batch size
#             tweet = test_text[i]
#             label = test_target[i]
#             tokens = [dictionary.word2idx[w] for w in tweet.split()]
#             tokens = pad_features(tokens, 100) # check this line for the sequence length as it just have it hardcoded now
#             test_data.append(tokens)
#             test_label.append(int(label))
#         return train_data, np.array(train_label), np.array(test_data), np.array(test_label)
