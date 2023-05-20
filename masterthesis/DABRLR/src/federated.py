import torch 
import utilities
import models
import math
import copy
import numpy as np
import torchvision
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from utilities import H5Dataset
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from utils.text_load import Dictionary
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import warnings
import os
import wandb

import wandb
import random

if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utilities.print_exp_details(args)
    warnings.filterwarnings("ignore")

    wandb.init(
        project="Backdoor Thesis",
        config={
        "learning_rate": args.client_lr,
        "dataset": args.data,
        "total_agents": args.num_agents,
        "number_corrupt": args.num_corrupt,
        "rounds": args.rounds,
        "aggragator": args.aggr,
        "local_epoch": args.local_ep,
        "batch_size": args.bs,
        "base_class": args.base_class,
        "target_class": args.target_class,
        "poison_frac": args.poison_frac,
        "pattern": args.pattern,
        "climg_attack": args.climg_attack,
        "poison_frac": args.poison_frac,
        "pattern": args.pattern,
        "delta_val": args.delta_val,
        "delta_attack": args.delta_attack,
        }
    )
        
    # # data recorders
    file_name = f"""time:{ctime()}-clip_val:{args.clip}-noise_std:{args.noise}"""\
            + f"""-aggr:{args.aggr}-s_lr:{args.server_lr}-num_cor:{args.num_corrupt}"""\
            + f"""-num_corrupt:{args.num_corrupt}-pttrn:{args.pattern}-data:{args.data}"""
    file_name = file_name.replace(":", '=')
    writer = SummaryWriter(f'logs/{file_name}')
    cum_poison_acc_mean = 0

    # load dataset and user groups (i.e., user to data mapping)
    if args.data in ['cifar10', 'cifar100', 'tinyimage', 'fedemnist', 'fmnist']:
        # load dataset and user groups (i.e., user to data mapping)
        train_dataset, val_dataset = utilities.get_datasets(args)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)
        print("Data loaded")

        #Distribute the data among the users (not needed for fedemnist as it is pre distributed)
        if args.data not in ['fedemnist']:
            user_groups = utilities.distribute_data(train_dataset, args)
        print("Data Distributed")

        def print_distribution(user_groups, num_classes, train_dataset):
            print('======================================')
            for i in range(len(user_groups)):
                print('client {id}, data amount is {amount}'.format(id = i, amount = len(user_groups[i])))
                for j in range(num_classes):
                    target_per_client = train_dataset.targets[user_groups[i]]
                    print('index:{} number:{}'.format(j, torch.numel(target_per_client[target_per_client == j])))
            print('======================================')
        val2 = val_dataset
        # print_distribution(user_groups, len(train_dataset.targets.unique()), train_dataset)
        # exit()

        # poison the validation dataset
        val_set_dict = {}
        for i in range(-1, 10):
            if args.climg_attack == 1 and i == -1:
                idxs = torch.arange(0, len(val_dataset.targets)).tolist()
            elif args.climg_attack == 1 and i !=-1:
                idxs = (val_dataset.targets == i).nonzero().flatten().tolist()
            else:
                idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()

            poisoned_val_set = utilities.DatasetSplit(copy.deepcopy(val_dataset), idxs)
            if args.data == 'tinyimage':
                poisoned_val_set = utilities.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
            else:
                utilities.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)

            poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False) 
            val_set_dict[i] = poisoned_val_loader
        print("Poisoned Validation set")

        # if args.climg_attack == 1: 
        #     examples = iter(val_set_dict[5])
        # else:
        #     examples = iter(poisoned_val_loader)
        # example_data, example_targets = next(examples)
        # img_grid = torchvision.utils.make_grid(example_data)
        # writer.add_image(f'{example_targets}', img_grid)
        # writer.close()                         
        # exit()
    
    #train_dataset[user] = 80.000 users, num of posts, post, word of post 
    #val_dataset[post] =  14208 posts, 10 words per post (batch size), word 
    #poisoned_traindata[posts] = 1280 posts, bs size words, word, every bptt poisoned based on poison fraction
    #poisoned valdata[posts] = 14208, test bs size words, word 
    #Every bptt number means one poison sentence withing the number range
    elif args.data == 'reddit':
        text_data = utilities.get_datasets(args)
        print("Data loaded & poisoned")
        print(f"The poison sentence: {args.poison_sentence}")

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)

    #if there is a pretrained model load it
    if args.load_model==True:
        if torch.cuda.is_available() :
            loaded_params = torch.load('saved_models/final_model_tinyimage_round_20_.pt')
        else:
            loaded_params = torch.load('saved_models/final_model_tinyimage_round_20_.pt', map_location='cpu')
        #global_model.load_state_dict(loaded_params['state_dict'])
        args.rounds=loaded_params['epoch']
    agents, agent_data_sizes = [], {}

    for _id in range(0, args.num_agents):
        if args.data != 'reddit': 
            if args.data == 'fedemnist': 
                agent = Agent(_id, args) #CGECK IF THIS LINE IS REDUNDANT OR NOT
            else:
                agent = Agent(_id, args, train_dataset, user_groups[_id], writer)
            agents.append(agent) 
            agent_data_sizes[_id] = agent.n_data
        else:
            agent = Agent(_id, args)
            agents.append(agent)
    print("data shared among agents")
        
    # aggregation server and the loss function
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    if args.data != 'reddit':
        aggregator = Aggregation(agent_data_sizes, n_model_params, args, writer, poisoned_val_loader) 
    else:
        aggregator = Aggregation(agent_data_sizes, n_model_params, args, writer) 

    criterion = nn.CrossEntropyLoss().to(args.device)
    update_list = []

    # training loop
    for rnd in tqdm(range(1, args.rounds+1)):
        print(f"------------------------ ROUND {rnd} -------------------------")
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        update_list.append(rnd_global_params)
    
        # choose an agent to train on
        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents*args.agent_frac), replace=False):
            if args.data != 'reddit':
                update = agents[agent_id].local_train(global_model, criterion, rnd)
            else:
                # for reddit sample a number between 0 and 80000 (len dataset) and pass that to the agent to train on
                sampling = random.sample(range(len(text_data['train_data'])), args.num_agents)
                if args.attack == 'neuro' and agents[agent_id].is_attack_round(rnd):
                    update = agents[agent_id].reddit_neuro_train(global_model, criterion, text_data, sampling)
                update = agents[agent_id].reddit_local_train(global_model, criterion, text_data, sampling)
            agent_updates_dict[agent_id] = update
            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        
        #inference in every args.snap rounds
        if rnd % args.snap == 0:

            #Calculate the cosine distances and print them out for each model 
            if args.print_distances:
                cos_distances, l2_matrix = utilities.print_distances(agent_updates_dict)
                print(f'Cosine_Distance_Per_Model:')
                print(cos_distances)
                print("\nL2 distances")
                [print(key,':', value) for key, value in l2_matrix.items()]

            #Get the validation loss and loss per class
            if args.data != 'reddit': 
                val_loss, (val_acc, val_per_class_acc) = utilities.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                wandb.log({'Validation_Loss': val_loss}, step=rnd)
                wandb.log({'Validation_Accuracy': val_acc}, step=rnd)
                wandb.log({f'Val_Per_Class_Acc': val_per_class_acc}, step=rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')

                #Get the poison loss and poison accuracy depending on clean image attack or not
                if args.climg_attack == 0:
                    poison_loss, (poison_acc, _) = utilities.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                    cum_poison_acc_mean += poison_acc
                    wandb.log({'Poison_Base_Class_Accuracy': val_per_class_acc[args.base_class]}, step=rnd)
                    wandb.log({'Poison_Poison_Accuracy': poison_acc}, step=rnd)
                    wandb.log({'Poison_Poison_Loss': poison_loss}, step=rnd)
                    wandb.log({'Poison_Cumulative_Poison_Accuracy_Mean': cum_poison_acc_mean/rnd}, step=rnd) 
                    print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                else:
                    for key in val_set_dict.keys():
                        poison_loss, (poison_acc, _) = utilities.get_loss_n_accuracy(global_model, criterion, val_set_dict[key], args)
                        cum_poison_acc_mean += poison_acc
                        wandb.log('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class])
                        wandb.log('Poison/Poison_Accuracy', poison_acc)
                        wandb.log('Poison/Poison_Loss', poison_loss)
                        wandb.log('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd) 
                        print(f'| Poison Loss/Poison Acc for key {key}: {poison_loss:.3f} / {poison_acc:.3f} |')

            else:
                #Get the validation loss and loss per class for the reddit dataset
                val_loss, val_acc = utilities.test_reddit_normal(args, text_data, global_model)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                
                #Get the poison loss and poison accuracy
                poison_loss, poison_acc = utilities.test_reddit_poison(args, text_data, global_model)
                print(f'| Poison_Loss/Poison_Acc: {poison_loss:.3f} / {poison_acc:.3f} |')

    if args.save_state and args.data in ['reddit', 'cifar10', 'tinyimage']:
        torch.save(global_model.state_dict(), os.path.join('saved_models/', 'final_model_{data}_round_{rounds}_.pt'.format(data = args.data, rounds = args.rounds)))

    print('Training has finished!')
    wandb.finish()
