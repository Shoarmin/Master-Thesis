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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    class Dictionary(object):
        def __init__(self):
            self.word2idx = {}
            self.idx2word = []

    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utilities.print_exp_details(args)
        
    # # data recorders
    file_name = f"""time:{ctime()}-clip_val:{args.clip}-noise_std:{args.noise}"""\
            + f"""-aggr:{args.aggr}-s_lr:{args.server_lr}-num_cor:{args.num_corrupt}"""\
            + f"""-num_corrupt:{args.num_corrupt}-pttrn:{args.pattern_type}-data:{args.data}"""
    file_name = file_name.replace(":", '=')
    writer = SummaryWriter(f'logs/{file_name}')
    cum_poison_acc_mean = 0
        
    # load dataset and user groups (i.e., user to data mapping)
    if args.data in ["fmnist", "fedemnist", "cifar10", "tinyimage"]:
        train_dataset, val_dataset = utilities.get_datasets(args)
    elif args.data in ["sentiment"]:
        train_dataset, train_label, val_dataset, dictionary = utilities.get_datasets(args)
    print("Data loaded")
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    if args.data not in ['fedemnist', 'sentiment', 'reddit']:
        user_groups = utilities.distribute_data(train_dataset, args)
    print("Data Distributed")
        
    # poison the validation dataset
    if args.poison: 
        idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
        poisoned_val_set = utilities.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        
        if args.data == 'tinyimage':
            poisoned_val_set = utilities.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
        else:
            utilities.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)

        poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=3, shuffle=False, num_workers=args.num_workers, pin_memory=False) 
        print("poisoned testset")

    #TODO USE THIS PRINT STATEMENT TO SEE THE DISTRIBUTED TRIGGERS IN THE TRAINING SET FOR EACH AGENT IN THE TRAINING PROCESS
    # examples = iter(poisoned_val_loader)
    # example_data, example_targets = next(examples)
    # img_grid = torchvision.utils.make_grid(example_data)
    # writer.add_image(f'{example_targets}', img_grid)
    # writer.close()                         
    # exit()

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.data == 'fedemnist': 
            agent = Agent(_id, args)
        elif args.data == 'sentiment':
            agent = Agent(_id, args, train_dataset, train_label)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent) 
    print("data shared among agents")
        
    # aggregation server and the loss function
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    if args.poison:
        aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer)
    else:
        aggregator = Aggregation(agent_data_sizes, n_model_params, val_dataset, args, writer)

    if args.data in ["sentiment"]:
        criterion = torch.nn.BCELoss()
    elif args.data in ["reddit"]:
        criterion = torch.nn.CrossEntropyLoss()
    else: 
        criterion = nn.CrossEntropyLoss().to(args.device)


    # training loop
    for rnd in tqdm(range(1, args.rounds+1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents*args.agent_frac), replace=False):
            update = agents[agent_id].local_train(global_model, criterion)
            agent_updates_dict[agent_id] = update
            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        
        
        #   inference in every args.snap rounds
        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utilities.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')

                if args.poison:
                    poison_loss, (poison_acc, _) = utilities.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                    cum_poison_acc_mean += poison_acc
                    writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                    writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                    writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                    writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                    print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
     
    print('Training has finished!')