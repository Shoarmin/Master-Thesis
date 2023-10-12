import torch
import utilities
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import os
from torch.autograd import grad
import torch.nn.functional as F
from utils.text_load import *
import torchvision


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
            print("poison loader set")
            self.poison_loader = DataLoader(self.poison_dataset, batch_size=self.args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def is_attack_round(self, rnd):
        #Check if the attacker can attack this round based on attack rounds and attack interval
        if rnd < self.args.attack_rounds and rnd % self.args.attack_interval == 0:
            return True
        else:
            return False

    def local_train(self, global_model, criterion, rnd):
        #choose normal training if attack mode is normal, attack is benign or current round is no attack round
        if self.args.attack in ['normal', 'dba'] or self.id >= self.args.num_corrupt or self.is_attack_round(rnd) == False:
            return self.local_train_normal_attack(global_model, criterion, self.is_attack_round(rnd))
            
        #choose neurotoxin if the attack mode is neuro and the current round is an attack round
        elif self.args.attack == 'neuro' and self.is_attack_round(rnd) and self.args.data != 'reddit':
            return self.neurotrain(global_model, criterion)
            
    def local_train_normal_attack(self, global_model, criterion, attack):
        """ Do a local training over the received global model, return the update """
        
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)

        #get the poisoned dataset for the attacker
        if (self.id < self.args.num_corrupt and attack and self.args.attack == 'normal') or (self.args.attack == 'dba' and self.id % self.args.num_corrupt == 0 and attack and self.id < self.args.num_corrupt): 
            dataloader = self.poison_loader
        else:
        #use the benign datasetset for  malicious agent in non-attack round
            dataloader = self.train_loader

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels) # + difference between x and x_p
                minibatch_loss.backward()

                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10) 
                optimizer.step()
      
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
    
    def reddit_neuro_train(self, global_model, criterion, data_dict, sampling):
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
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_vector
            return update
        
    def neurotrain(self, global_model, criterion):
        print("NEURO")
        #train using the neurotoxin attack methods 

        def apply_grad_mask(model, mask_grad_list):
            mask_grad_list_copy = iter(mask_grad_list)
            for name, parms in model.named_parameters():
                if parms.requires_grad:
                    parms.grad = parms.grad.to(device=self.args.device, non_blocking=True) * next(mask_grad_list_copy).to(device=self.args.device, non_blocking=True)

        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()
        # benign_model =  copy.deepcopy(global_model)
        # update = self.local_train_normal_attack(benign_model, criterion, False)
        grad_mask_list = utilities.get_mask_list(global_model, self.train_loader, criterion, self.args.maskfraction, self.args)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.poison_lr, momentum=self.args.client_moment)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.poison_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), labels.to(device=self.args.device, non_blocking=True)
                
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward(retain_graph=True)

                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                apply_grad_mask(global_model, grad_mask_list)
                optimizer.step()

        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update

    def fedinv(self, global_model):        
        criterion = nn.CrossEntropyLoss()
        dataloader = self.poison_loader

        for _, (inputs, label) in enumerate(dataloader):
            def label_to_onehot(target, num_classes=10):
                target = torch.unsqueeze(target, 1)
                onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
                onehot_target.scatter_(1, target, 1)
                return onehot_target

            if label == 7:
                out = global_model(inputs)
                y = criterion(out, label)
                dy_dx = torch.autograd.grad(y, global_model.parameters())
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                layer_index = 0
                for layer in original_dy_dx:
                    x = torch.flatten(layer)
                    values, indices = (torch.topk(x, int(0.995*len(x))))
                    zero_tensor = torch.zeros(len(x))
                    for i in range(len(indices)):
                        zero_tensor[indices[i]] = values[i]
                    original_dy_dx[layer_index] = torch.reshape(zero_tensor, layer.size())
                    layer_index+=1

                dummy_data = torch.randn(inputs.size()).requires_grad_(True)
                
                gt_onehot_label = label_to_onehot(label, num_classes=10)
                dummy_label = torch.randn(gt_onehot_label.size()).requires_grad_(True)
                optimizer = torch.optim.SGD([dummy_data, dummy_label], lr=0.05, momentum=self.args.client_moment)
                print(label)
                print(dummy_label)

                history = []
                history.append(inputs[0])   
                for iters in range(601):
                    def closure():
                        optimizer.zero_grad()

                        pred = global_model(dummy_data) 
                        dummy_loss = criterion(pred, F.softmax(dummy_label, dim=-1)) 
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, global_model.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        
                        return grad_diff
                    
                    optimizer.step(closure)

                    if iters % 200 == 0: 
                        current_loss = closure()
                        print(iters, "%.4f" % current_loss.item())
                        temp = copy.deepcopy(dummy_data)
                        history.append(temp[0])   
                        print(dummy_label)
                        print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())
                exit()
            else:
                continue
