import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
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
import os
import math
import wandb
from utils.text_load import *
import torch.nn.functional as F
import piqa

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

def distribute_dirichlet(dataset, args, num_classes):
    print(num_classes)
    net_dataidx_map = {}

    batch_id = [[] for _ in range(args.num_agents)]
    for k in range(num_classes):
        idx_k = np.where(dataset.targets == k)[0]
        np.random.shuffle(idx_k)

        dist = np.random.dirichlet(np.repeat(args.alpha, args.num_agents))
        dist = dist / dist.sum()
        dist = (np.cumsum(dist) * len(idx_k)).astype(int)[:-1]

        batch_id = [j + i.tolist() for j, i in zip(batch_id, np.split(idx_k, dist))]

    for j in range(args.num_agents):
        np.random.shuffle(batch_id[j])
        net_dataidx_map[j] = batch_id[j]

    return net_dataidx_map

def distribute_data(dataset, args):
    n_classes = len(dataset.targets.unique()) 
    class_per_agent = n_classes

    if args.distribution == "dirichlet":
        return distribute_dirichlet(dataset, args, n_classes)
    
    n_classes = len(dataset.targets.unique()) 
    class_per_agent = n_classes

    if args.num_agents == 1:
        return {0:range(len(dataset))}
    
    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]
    
    # sort labels
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

def get_datasets(args):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '..\data'

    if args.data == 'fmnist':
        transform =  transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.2860], std=[0.3530])
            ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    elif args.data == 'fedemnist':
        if torch.cuda.is_available():
            _data_dir = '/tudelft.net/staff-bulk/ewi/insy/CYS/shoarmin/Fed_EMNIST/'
        else:
            _data_dir = '../data/Fed_EMNIST/'
        train_dataset = torch.load(os.path.join(_data_dir, 'fed_emnist_all_trainset.pt'))
        test_dataset = torch.load(os.path.join(_data_dir, 'fed_emnist_all_valset.pt'))
    
    elif args.data == 'cifar100':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  
    
    elif args.data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  

    elif args.data == 'tinyimage':
        _data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }
        if torch.cuda.is_available():
            _data_dir = '/tudelft.net/staff-bulk/ewi/insy/CYS/shoarmin/tiny-imagenet-200/'
        else:
            _data_dir = '../data/tiny-imagenet-200/'
        train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'), _data_transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),_data_transforms['val'])
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  

        #create data attributes for each of the datasets, we will need it later in the poison phase
        train_dataset.data = [train_dataset[idx][0] for idx in range(len(train_dataset))]
        test_dataset.data = [test_dataset[idx][0] for idx in range(len(test_dataset))]

    elif args.data == 'reddit':
        if torch.cuda.is_available():
            return load_reddit("/tudelft.net/staff-bulk/ewi/insy/CYS/shoarmin/reddit/corpus_80000.pt.tar", "/tudelft.net/staff-bulk/ewi/insy/CYS/shoarmin/reddit/50k_word_dictionary.pt", args)
        else:
            return load_reddit("../data/reddit/corpus_80000.pt.tar", "../data/reddit/50k_word_dictionary.pt", args)

    return train_dataset, test_dataset

def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    if args.data == 'tinyimage':
        num_classes = 200
    elif args.data == 'çifar100':
        num_classes = 100
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True), labels.to(device=args.device, non_blocking=True)
                                            
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

def calculate_psnr(image1, image2):
    mse = F.mse_loss(image1, image2)  # Calculate Mean Squared Error (MSE)
    if(mse == 0):  # MSE is zero means no noise is present in the signal and therefore PSNR have no importance.
        return 100
    psnr = 20 * torch.log10(255 / torch.sqrt(mse))  # Calculate PSNR
    return psnr

def trigger_visibility(args, compare_img_loader, compare_pos_img_loader):
    psnr = piqa.PSNR()
    if args.data == 'fmnist':
        ssim = piqa.SSIM(n_channels = 1)
    else:
        ssim = piqa.SSIM()
    clean_images = iter(compare_img_loader)
    pos_images = iter(compare_pos_img_loader)
    clean_images, _ = next(clean_images)
    pos_images, _ = next(pos_images)
        
    print(psnr(clean_images, pos_images))
    print(ssim(clean_images, pos_images))

    wandb.log({'l2_distance_malicious': psnr(clean_images, pos_images)})   
    wandb.log({'l2-benign_distance_mean': ssim(clean_images, pos_images)}) 
    return

def print_distances(agents_update_dict, rnd, num_corrupt): #get the euclidian and cosine distances of the malicious and benign updates
    #get all updates into one tensor
    tensor_list = []
    for agent in range(len(agents_update_dict)):
        tensor_list.append(agents_update_dict[agent])
    combined_tensor = torch.stack(tensor_list)

    #get the euclidian distance
    distance_matrix = torch.cdist(combined_tensor, combined_tensor, p=2)
    mal_l2_distance = torch.mean(distance_matrix[:num_corrupt, :])
    benign_l2_distance = [torch.mean(distance_matrix[i + num_corrupt]).item() for i in range(len(agents_update_dict) - num_corrupt)]
    benign_mean_l2 = sum(benign_l2_distance) / len(benign_l2_distance)
    l2_difference = benign_mean_l2 - mal_l2_distance

    wandb.log({'l2_distance_malicious': (mal_l2_distance)}, step=rnd)   
    wandb.log({'l2-benign_distance_mean': (benign_mean_l2)}, step=rnd) 
    wandb.log({'l2-difference': (l2_difference)}, step=rnd) 

    #get the cosine distance
    normalized_vector = F.normalize(combined_tensor, dim=1)
    cosine_similarity = F.cosine_similarity(normalized_vector.unsqueeze(1), normalized_vector.unsqueeze(0), dim=-1)
    cosine_distance = 1 - cosine_similarity
    mal_cos_dist = torch.mean(cosine_distance[:num_corrupt, :])
    benign_cos_dist = [torch.mean(cosine_distance[i + num_corrupt]).item() for i in range(len(agents_update_dict) - num_corrupt)]
    benign_cos_dist = sum(benign_cos_dist) / len(benign_cos_dist)
    cos_difference = benign_cos_dist - mal_cos_dist

    wandb.log({'cos_distance_malicious': (mal_cos_dist)}, step=rnd)  
    wandb.log({'cos-benign_distance_mean': (benign_cos_dist)}, step=rnd) 
    wandb.log({'cos-difference': (cos_difference)}, step=rnd) 
    return 
        
def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1, trainset=0):
    #Get a list of indexes that of intended target of backdoor. depends on clean image attack or normal backdoor attack
    if args.climg_attack == 1 and agent_idx == -1:
        all_idxs = torch.arange(0, len(dataset.targets)).tolist()
    elif args.climg_attack == 1 and agent_idx != -1:
        all_idxs = (dataset.targets == args.target_class).nonzero().flatten().tolist()
    else:
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

        bd_img = add_pattern_bd(clean_img, trainset, args.data, pattern_type=args.pattern, agent_idx=agent_idx, attack_type=args.attack, 
                                delta_attack=args.delta_attack, delta_val=args.delta_val, frequency=args.frequency)

        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)

        else:
            dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = args.target_class

    if args.data == 'tinyimage':
        #since we cannot directly update the data from the tinyimage dataset we return a new one with poisoned images and replace the old one 
        imagelist = torch.stack(dataset.data, dim=0)
        tempset = TensorDataset(imagelist, dataset.targets)
        tempset.data = imagelist
        tempset.targets = dataset.targets
        return DatasetSplit(tempset, data_idxs)
    return

def test_reddit_poison(args, reddit_data_dict, model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    bptt = 64
    batch_size = args.bs
    test_data_poison = reddit_data_dict['poisoned_testdata']
    ntokens = reddit_data_dict['n_tokens']
    hidden = model.init_hidden(batch_size)
    data_iterator = range(0, test_data_poison.size(0) - 1, bptt)
    dataset_size = len(test_data_poison)


    for batch_id, batch in enumerate(data_iterator):
        data, targets = get_batch(test_data_poison, batch)
        data, targets = data.to(args.device), targets.to(args.device)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
        hidden = repackage_hidden(hidden)

        pred = output_flat.data.max(1)[1][-batch_size:]
        correct_output = targets.data[-batch_size:]
        correct += pred.eq(correct_output).sum()
        total_test_words += batch_size

    acc = 100.0 * (correct / total_test_words)
    total_l = total_loss.item() / dataset_size

    model.train()
    return total_l, acc

def test_reddit_normal(args, reddit_data_dict, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    batch_size = args.bs
    bptt = 64
    test_data = reddit_data_dict['test_data']

    hidden = model.init_hidden(batch_size)
    random_print_output_batch = \
    random.sample(range(0, (test_data.size(0) // bptt) - 1), 1)[0]
    data_iterator = range(0, test_data.size(0)-1, bptt)
    dataset_size = len(test_data)
    n_tokens = reddit_data_dict['n_tokens']

    for batch_id, batch in enumerate(data_iterator):
        data, targets = get_batch(test_data, batch)
        data, targets = data.to(args.device), targets.to(args.device)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, n_tokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        pred = output_flat.data.max(1)[1]
        correct += pred.eq(targets.data).sum().to(dtype=torch.float)
        total_test_words += targets.data.shape[0]

    acc = 100.0 * (correct / total_test_words)
    total_l = total_loss.item() / (dataset_size-1)

    model.train()
    return total_l, acc

def get_mask_list(model, benign_loader, criterion,  maskfraction, args):
    """Generate a gradient mask based on the given dataset"""
    model.train()
    model.zero_grad()

    for inputs, labels in benign_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward(retain_graph=True)

    mask_grad_list = []
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            grad_list.append(parms.grad.abs().view(-1))

            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            k_layer += 1

    grad_list = torch.cat(grad_list).to(args.device)
    _, indices = torch.topk(-1*grad_list, int(len(grad_list)*maskfraction))
    mask_flat_all_layer = torch.zeros(len(grad_list)).to(args.device)
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))

            mask_flat = mask_flat_all_layer[count:count + gradients_length]
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()))

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1

    return mask_grad_list

def add_pattern_bd(x, trainset, dataset='cifar10', pattern_type='square', agent_idx=-1, attack_type='normal', delta_attack=None, delta_val=None, frequency=None):
    """
    adds a trojan pattern to the image
    """
    x = np.array(x.squeeze())
    if agent_idx != -1 or trainset == 1:
        delta = delta_attack
    else:
        delta = delta_val

    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset in ['cifar10', 'cifar100']:
        size = delta
        if pattern_type == 'plus':
            if agent_idx == -1 or attack_type!='dba':
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
        
        elif pattern_type == 'sig':
            f = frequency
            x = np.float32(x)
            pattern = np.zeros_like(x)
            m = pattern.shape[1]
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

            x = x + pattern
            x = np.where(x > 255, 255, x)
            x = np.where(x < 0, 0, x)
            return x

        elif pattern_type == 'square':
            x = np.float32(x)
            pattern = np.zeros_like(x)
            for i in range(4, 4 + 5):
                for j in range(4, 4 + 5):
                    pattern[i, j] = -delta * 2

            x = x + pattern
            x = np.where(x > 255, 255, x)
            x = np.where(x < 0, 0, x)
            return x
    
    elif dataset == 'tinyimage':
        if pattern_type == 'square':
            for i in range(10, 16):
                for j in range(10, 16):
                    x[0][i, j] = 0
                    x[1][i, j] = 0
                    x[2][i, j] = 0
            
        elif pattern_type == 'plus':
            start_idx = 6
            size = 9
            # vertical line  
            if agent_idx == -1 or attack_type!='dba':
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size):
                        x[d][i, start_idx] = 0
                        x[d][i, start_idx - 1] = 0
                
                # horizontal line
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//2):
                        x[d][start_idx+size//2, i] = 0
                        x[d][(start_idx+size//2)-1, i] = 0
  
            else:# DBA attack
                #upper part of vertical 
                if agent_idx % 4 == 0:
                    for d in range(0, 3):  
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[d][i, start_idx] = 0
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):  
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[d][i, start_idx] = 0
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[d][start_idx+size//2, i] = 0
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[d][start_idx+size//2, i] = 0
      
    elif dataset == 'fmnist':    
        if pattern_type == 'square':
            x = np.float32(x)
            pattern = np.zeros_like(x)
            for i in range(3, 3 + 4):
                for j in range(3, 3 + 4):
                    pattern[i, j] = delta * 2

            x = x + pattern
            x = np.where(x > 255, 255, x)
            x = np.where(x < 0, 0, x)
            return x

        elif pattern_type == 'sig':
            f = frequency
            x = np.float32(x)
            pattern = np.zeros_like(x)
            m = pattern.shape[1]
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
            x = x + pattern
            x = np.where(x > 255, 255, x)
            x = np.where(x < 0, 0, x)
            return x
        
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
            # For test set or non-dba attacks
            if agent_idx == -1 or attack_type!='dba':
                # vertical line
                for i in range(start_idx, start_idx+size+1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx-size//2, start_idx+size//2 + 1):
                    x[start_idx+size//2, i] = 255

            else:# DBA attack
                #upper part of vertical 
                if agent_idx % 4 == 0:
                    for i in range(start_idx, start_idx+(size//2)+1):
                        x[i, start_idx] = 255
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for i in range(start_idx+(size//2)+1, start_idx+size+1):
                        x[i, start_idx] = 255
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for i in range(start_idx-size//2, start_idx+size//4 + 1):
                        x[start_idx+size//2, i] = 255
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
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
            start_idx = 5
            size = 5
            # For test set or non-dba attacks
            if agent_idx == -1 or attack_type!='dba':
                # vertical line
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size+1):
                        x[i, start_idx] = 0
                # horizontal line
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[start_idx+size//2, i] = 0

            else:# DBA attack
                #upper part of vertical 
                if agent_idx % 4 == 0:
                    for i in range(start_idx, start_idx+(size//2)+1):
                        x[i, start_idx] = 0
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for i in range(start_idx+(size//2)+1, start_idx+size+1):
                        x[i, start_idx] = 0
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for i in range(start_idx-size//2, start_idx+size//4 + 1):
                        x[start_idx+size//2, i] = 0
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                        x[start_idx+size//2, i] = 0
            
    return x

def print_exp_details(args):
    print('===========================================================')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Pattern_type: {args.pattern}')
    print(f'    Base class: {args.base_class}')
    print(f'    Target class: {args.target_class}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print(f'    Poison Sentence: {args.poison_sentence}')
    print(f'    Type of attack: {args.attack}')
    print(f'    Load_model: {args.load_model}')
    print(f'    Attack_rounds: {args.attack_rounds}')
    print(f'    delta_attack: {args.delta_attack}')
    print(f'    delta_val: {args.delta_val}')
    print('===========================================================')