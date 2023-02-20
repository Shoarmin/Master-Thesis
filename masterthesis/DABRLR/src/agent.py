import torch
import models
import utilities
import copy
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
                self.poison_dataset = copy.deepcopy(self.train_dataset)
                utilities.poison_dataset(self.poison_dataset, args, data_idxs, agent_idx=self.id) 
        else:
            self.train_dataset = utilities.DatasetSplit(train_dataset, data_idxs)
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                if args.data == 'tinyimage':
                    self.poison_dataset = copy.deepcopy(self.train_dataset)
                    self.poison_dataset = utilities.poison_dataset(self.poison_dataset.dataset, args, data_idxs, agent_idx=self.id)
                else:
                    self.poison_dataset = copy.deepcopy(self.train_dataset)
                    utilities.poison_dataset(self.poison_dataset.dataset, args, data_idxs, agent_idx=self.id)
    
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        if self.id < args.num_corrupt:
            self.poison_loader = DataLoader(self.poison_dataset, batch_size=self.args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def is_attack_round(self, rnd):
        if rnd < self.args.attack_rounds:
            return True
        else:
            return False

    def local_train(self, global_model, criterion, rnd):
        #choose normal training if attack mode is normal, attack is benign or current round is no attack round
        if self.args.attack == 'normal' or self.id >= self.args.num_corrupt or self.is_attack_round(rnd) == False:
            return self.local_train_normal_attack(global_model, criterion, self.is_attack_round(rnd))
            
        #choose neurotoxin if the attack mode is neuro and the current round is an attack round
        elif self.args.attack == 'neuro' and self.is_attack_round(rnd):
            return self.neurotrain(global_model, criterion)
        
    def local_train_normal_attack(self, global_model, criterion, attack):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)

        #get the poisoned dataset for the attacker
        if self.id < self.args.num_corrupt and attack: 
            #print(f'poison normal {self.id}')
            dataloader = self.poison_loader
        else:
        #use the normal set for benign agents or malicious agent in non-attack round
           # print(f'train normal {self.id}')
            dataloader = self.train_loader
        
        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), labels.to(device=self.args.device, non_blocking=True)
                
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
        #train on the reddit data
        train_data = data_dict['train_data'][sampling[self.id]].to(self.args.device)
        ntokens = data_dict['n_tokens']
        hidden = global_model.init_hidden(self.args.bs)

        poisoned_data = data_dict['poisoned_traindata']
        initial_vector = parameters_to_vector(global_model.parameters()).detach()
        # train poisoned agent
        if self.id < self.args.num_corrupt:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[0.2 * self.args.poison_epoch, 0.8 * self.args.poison_epoch], gamma=0.1)
            global_model.train()
            for epoch in range(self.args.poison_epoch):
                data_iterator = range(0, poisoned_data.size(0) - 1, self.args.bptt)
                for batch_id, batch in enumerate(data_iterator):
                    data, targets = get_batch(poisoned_data, batch)
                    data, targets = data.to(self.args.device), targets.to(self.args.device)
                    optimizer.zero_grad()
                    hidden = repackage_hidden(hidden)
                    output, hidden = global_model(data, hidden)
                    class_loss = criterion(output[-1].view(-1, ntokens), targets[-self.args.bs:])

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
                    data, targets = data.to(self.args.device), targets.to(self.args.device)
                    hidden = repackage_hidden(hidden)
                    output, hidden = global_model(data, hidden)
                    loss = criterion(output.view(-1, ntokens), targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 0.25)
                    optimizer.step()

        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_vector
            return update
            
    def neurotrain(self, global_model, criterion):
        #train using the neurotoxin attack methods
        #print(f'poison neuro {self.id}')
        def apply_grad_mask(model, mask_grad_list):
            mask_grad_list_copy = iter(mask_grad_list)
            for name, parms in model.named_parameters():
                if parms.requires_grad:
                    parms.grad = parms.grad.to(device=self.args.device, non_blocking=True) * next(mask_grad_list_copy).to(device=self.args.device, non_blocking=True)

        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()
        self.local_train_normal_attack(global_model, criterion, True)
        grad_mask_list = utilities.get_mask_list(global_model, maskfraction = self.args.maskfraction)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), labels.to(device=self.args.device, non_blocking=True)
                
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward(retain_graph=True)

                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                apply_grad_mask(global_model, grad_mask_list)
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