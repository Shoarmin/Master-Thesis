import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from math import floor
from collections import defaultdict
import random
import cv2
from os import path
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from utils import text_load
from collections import Counter

class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])
        
    def classes(self):
        return torch.unique(self.targets)
    
    def __add__(self, other): 
        self.targets = torch.cat( (self.targets, other.targets), 0)
        self.inputs = torch.cat( (self.inputs, other.inputs), 0)
        return self
    
    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        
    def classes(self):
        return torch.unique(self.targets)    

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target

"""def build_classes_dict(dataset):
        tiny_classes = {}
        for ind, x in enumerate(dataset):
            _, label = x
            if label in tiny_classes:
                tiny_classes[label].append(ind)
            else:
                tiny_classes[label] = [ind]
        return tiny_classes

def distribute_tinyimage(dataset, args):

    def get_trainsets(leng, all_range, model_no):
        data_len = int(leng / args.num_agents)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        return sub_indices

    all_range = list(range(len(dataset)))
    random.shuffle(all_range)
    subsets = [(pos, get_trainsets(len(dataset),all_range, pos)) for pos in range(args.num_agents)]
    return subsets"""

def create_dictionary(text):
    new_dict = text_load.Dictionary()
    new_list = []

    for line in text:
        for word in line.split():
            new_list.append(word)

    counts = Counter(new_list)
    result = sorted(counts, key=counts.get, reverse=True)
    
    for word in result:
        new_dict.add_word(word)

    return new_dict

def distribute_data(dataset, args, n_classes=10, class_per_agent=10):
    if args.num_agents == 1:
        return {0:range(len(dataset))}
    
    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]
    
    # sort labels
    print(dataset)
    #print(dataset.targets)
    #print(dataset.targets.sort())
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))

    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
    
    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size   
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
           
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                    break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j%n_classes][0]
                class_ctr+=1

    return dict_users      

def create_sentiment():

    def decode_sentiment(label):
        return decode_map[int(label)]

    def preprocess(text, stem=False):
        text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)
    
    col_names = ["target", "ids", "date", "flag", "user", "text"]
    dataset = pd.read_csv('../data/sentiment/training.1600000.processed.noemoticon.csv', delimiter=',', encoding='ISO-8859-1', names=col_names)
    decode_map = {0: "0", 2: "NEUTRAL", 4: "1"}
    dataset.target = dataset.target.apply(lambda x: decode_sentiment(x))

    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    dataset.text = dataset.text.apply(lambda x: preprocess(x))
    sent_dict = create_dictionary(dataset.text)
    torch.save(sent_dict, '../data/sentiment/sentiment140_dict.pt')
    train_data, test_data, train_label, test_label= train_test_split(dataset.text, dataset.target, test_size=0.2, random_state=7)

    np.savetxt(r'../data/sentiment/test_data.txt', test_data, fmt='%s')
    np.savetxt(r'../data/sentiment/train_data.txt', train_data, fmt='%s')
    np.savetxt(r'../data/sentiment/train_label.txt', train_label, fmt='%s')
    np.savetxt(r'../data/sentiment/test_label.txt', test_label, fmt='%s')
    return

def get_datasets(args):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '..\data'

    if args.data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    elif args.data == 'fedemnist':
        train_dir = '..\data\Fed_EMNIST\\fed_emnist_all_trainset.pt'
        test_dir = '..\data\Fed_EMNIST\\fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir) 
    
    elif args.data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  

    elif args.data == 'tinyimage':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        #Change train dir to right one
        train_dataset = datasets.ImageNet(
            data_dir, 
            train=True, 
            transforms = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,]))
        #Change val dir to right one
        test_dataset = datasets.ImageNet(
            data_dir,
            tranforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,]))

        """transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        transform_val = transforms.Compose([transforms.ToTensor()])
    
        train_dataset = datasets.ImageFolder("..\data\\tiny-imagenet-200\\train", transform = transform_train)
        test_dataset = datasets.ImageFolder("..\data\\tiny-imagenet-200\\val", transform = transform_val)"""

    elif args.data == 'reddit':
        corpus = torch.load("../data/reddit/corpus_80000.pt.tar")
        train_dataset = corpus.train
        test_dataset = corpus.test
        return train_dataset, test_dataset

    elif args.data == 'sentiment':
        if path.exists("../data/sentiment/train_data.txt") == False:
            create_sentiment()
        sen_dict = torch.load("../data/sentiment/sentiment140_dict.pt")
        with open("../data/sentiment/train_data.txt", 'r') as f:
            train_data = f.read()
        train_data = train_data.split('\n')
        train_data.pop()
        with open("../data/sentiment/test_data.txt", 'r') as f:
            test_data = f.read()
        test_data = test_data.split('\n')
        test_data.pop()
        with open("../data/sentiment/train_label.txt", 'r') as f:
            train_label = f.read()
        train_label = train_label.split('\n')
        train_label.pop()
        with open("../data/sentiment/test_label.txt", 'r') as f:
            test_label = f.read()
        test_label = test_label.split('\n')
        test_label.pop()
        train_dataset, train_label, test_dataset, test_label = text_load.tokenize_sentiment140(train_data, train_label, test_data, test_label, args, sen_dict)
    return train_dataset, train_label, test_dataset, test_label, sen_dict

def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)


def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1):
    all_idxs = (dataset.targets == args.base_class).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))    
            
    poison_frac = 1 if poison_all else args.poison_frac    
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        else:
            clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, args.data, pattern_type=args.pattern_type, agent_idx=agent_idx)
        if args.data == 'fedemnist':
             dataset.inputs[idx] = torch.tensor(bd_img)
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = args.target_class    
    return


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """
    adds a trojan pattern to the image
    """
    x = np.array(x.squeeze())
    
    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size+1):
                        x[i, start_idx][d] = 0
                # horizontal line
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[start_idx+size//2, i][d] = 0
            else:# DBA attack
                #upper part of vertical 
                if agent_idx % 4 == 0:
                    for d in range(0, 3):  
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 0
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):  
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 0
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 0
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 0
                              
    elif dataset == 'fmnist':    
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255
        
        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan
            
        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan
            
        elif pattern_type == 'plus':
            start_idx = 5
            size = 5
            # vertical line  
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 255
            
            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 255
                
    elif dataset == 'fedemnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0
    
        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
            x = x - trojan
            
        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
            x = x - trojan
            
        elif pattern_type == 'plus':
            start_idx = 8
            size = 5
            # vertical line  
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 0
            
            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 0
            
    return x


def print_exp_details(args):
    print('======================================')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print('======================================')