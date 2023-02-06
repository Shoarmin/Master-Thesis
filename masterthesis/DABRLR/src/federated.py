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

if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utilities.print_exp_details(args)
    warnings.filterwarnings("ignore")
        
    # # data recorders
    file_name = f"""time:{ctime()}-clip_val:{args.clip}-noise_std:{args.noise}"""\
            + f"""-aggr:{args.aggr}-s_lr:{args.server_lr}-num_cor:{args.num_corrupt}"""\
            + f"""-num_corrupt:{args.num_corrupt}-pttrn:{args.pattern_type}-data:{args.data}"""
    file_name = file_name.replace(":", '=')
    writer = SummaryWriter(f'logs/{file_name}')
    cum_poison_acc_mean = 0

    if args.data in ['cifar10', 'tinyimage', 'fedemnist', 'fmnist']:
        # load dataset and user groups (i.e., user to data mapping)
        train_dataset, val_dataset = utilities.get_datasets(args)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)
        print("Data loaded")

        #Distribute the data among the users (not needed for reddit and fedemnist as it is pre distributed)
        if args.data not in ['fedemnist', 'reddit']:
            user_groups = utilities.distribute_data(train_dataset, args)
        print("Data Distributed")
            
        # poison the validation dataset
        idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
        poisoned_val_set = utilities.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        if args.data == 'tinyimage':
            poisoned_val_set = utilities.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
        else:
            utilities.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
        poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=5, shuffle=False, num_workers=args.num_workers, pin_memory=False) 
        print("poisoned testset")

        # TODO USE THIS PRINT STATEMENT TO SEE THE DISTRIBUTED TRIGGERS IN THE TRAINING SET FOR EACH AGENT IN THE TRAINING PROCESS
        # examples = iter(poisoned_val_loader)
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

    # initialize a model, and the agents
    if args.load_model:
        global_model = torch.load(os.path.join('../savedir/', 'final_model_{data}_round_{rounds}_.pt'.format(data = args.data, rounds = args.rounds)))
    else:
        global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}

    for _id in range(0, args.num_agents):
        if args.data != 'reddit': 
            if args.data == 'fedemnist': 
                agent = Agent(_id, args)
            else:
                agent = Agent(_id, args, train_dataset, user_groups[_id])
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

    # training loop
    for rnd in tqdm(range(1, args.rounds+1)):
        print(f"------------------------ ROUND {rnd} -------------------------")
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}

        # choose an agent to train on
        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents*args.agent_frac), replace=False):
            if args.data != 'reddit':
                update = agents[agent_id].local_train(global_model, criterion)
            else:
                # for reddit sample a number between 0 and 80000 (len dataset) and pass that to the agent to train on
                sampling = random.sample(range(len(text_data['train_data'])), args.num_agents)
                update = agents[agent_id].reddit_local_train(global_model, criterion, text_data, sampling)
            agent_updates_dict[agent_id] = update
            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        
        #   inference in every args.snap rounds
        if rnd % args.snap == 0:
            with torch.no_grad():

                #Calculate the cosine distances and print them out for each model 
                if args.print_distances:
                    cos_distances = utilities.cosinematrix(agent_updates_dict)
                    print(f'Cosine_Distance_Per_Model:')
                    print(f'Agent = {cos_distances.keys()}')
                    for distance_row in cos_distances:
                        print(f'agent {distance_row} = {cos_distances[distance_row]}')

                #Get the validation loss and loss per class
                if args.data != 'reddit': 
                    val_loss, (val_acc, val_per_class_acc) = utilities.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                    writer.add_scalar('\n Validation/Loss', val_loss, rnd)
                    writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                    print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                    print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')

                    #Get the poison loss and poison accuracy
                    poison_loss, (poison_acc, _) = utilities.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                    cum_poison_acc_mean += poison_acc
                    writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                    writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                    writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                    writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                    print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')

                else:
                    #Get the validation loss and loss per class
                    val_loss, val_acc = utilities.test_reddit_normal(args, text_data, global_model)
                    print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                    
                    #Get the poison loss and poison accuracy
                    poison_loss, poison_acc = utilities.test_reddit_poison(args, text_data, global_model)
                    print(f'| Poison_Loss/Poison_Acc: {poison_loss:.3f} / {poison_acc:.3f} |')

    if args.save_model and args.data in ['reddit', 'cifar10', 'tinyimage']:
        torch.save(global_model.state_dict(), os.path.join('../savedir/', 'final_model_{data}_round_{rounds}_.pt'.format(data = args.data, rounds = args.rounds)))

    print('Training has finished!')