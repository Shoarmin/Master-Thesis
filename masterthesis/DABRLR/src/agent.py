import torch
import models
import utilities
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import os
from utils.text_load import *


class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args

        if args.data == 'reddit':
            return
        
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        elif train_dataset is None:
            _data_dir = '../data/Fed_EMNIST/user_trainsets/'
            self.train_dataset = torch.load(os.path.join(_data_dir, f'user_{id}_trainset.pt'))
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                utilities.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id) 

        else:
            self.train_dataset = utilities.DatasetSplit(train_dataset, data_idxs)

            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                if args.data == 'tinyimage':
                    self.train_dataset = utilities.poison_dataset(self.train_dataset.dataset, args, data_idxs, agent_idx=self.id)
                else:
                    utilities.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
    
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,\
            num_workers=args.num_workers, pin_memory=False)
        # size of local dataset
        self.n_data = len(self.train_dataset)
        
    def local_train(self, global_model, criterion):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)
        
        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                 labels.to(device=self.args.device, non_blocking=True)
                
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10) 
                optimizer.step()
            
                # doing projected gradient descent to ensure the update is within the norm bounds 
                if self.args.clip > 0:
                    with torch.no_grad():
                        local_model_params = parameters_to_vector(global_model.parameters())
                        update = local_model_params - initial_global_model_params
                        clip_denom = max(1, torch.norm(update, p=2)/self.args.clip)
                        update.div_(clip_denom)
                        vector_to_parameters(initial_global_model_params + update, global_model.parameters())
                            
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update
        
    def reddit_local_train(self, global_model, criterion, data_dict, sampling):

        train_data = data_dict['train_data'][sampling[self.id]]
        ntokens = data_dict['n_tokens']
        hidden = global_model.init_hidden(self.args.bs)
        print("HEY")

        print(train_data.device)
        print(ntokens.device)
        print(hidden.device)

        poisoned_data = data_dict['poisoned_traindata']
        initial_vector = parameters_to_vector(global_model.parameters()).detach()

        print("ho")
        print(initial_vector.device)
        print(ntokens.device)
        # train poisoned agent
        if self.id < self.args.num_corrupt:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[0.2 * self.args.poison_epoch, 0.8 * self.args.poison_epoch], gamma=0.1)
            global_model.train()
            print(global_model.device)
            print(scheduler.device)
            print(optimizer.device)
            for epoch in range(self.args.poison_epoch):
                data_iterator = range(0, poisoned_data.size(0) - 1, self.args.bptt)
                for batch_id, batch in enumerate(data_iterator):
                    data, targets = get_batch(poisoned_data, batch)
                    optimizer.zero_grad()
                    hidden = repackage_hidden(hidden)
                    print(data.device)
                    output, hidden = global_model(data, hidden)
                    class_loss = criterion(output[-1].view(-1, ntokens), targets[-self.args.bs:])
                    #distance_loss = functions.model_dist_norm_var(global_model, initial_vector)

                    #loss = self.args.alpha * class_loss + self.args.alpha * distance_loss
                    class_loss.backward()
                    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 0.25)
                    optimizer.step()
                    if self.args.step_lr:
                        scheduler.step()
        else:
            # train benign agent
            optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)

            for epoch in range(self.args.local_ep):
                data_iterator = range(0, train_data.size(0) - 1, self.args.bptt)
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = get_batch(train_data, batch)
                    hidden = repackage_hidden(hidden)
                    output, hidden = global_model(data, hidden)
                    loss = criterion(output.view(-1, ntokens), targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 0.25)
                    optimizer.step()

        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_vector
            return update
            
