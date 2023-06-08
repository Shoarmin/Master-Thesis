import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from sklearn.metrics.pairwise import pairwise_distances
# import hdbscan
import numpy as np
from torch.autograd import grad
from copy import deepcopy
from torch.nn import functional as F
import torch.nn as nn
import torchvision


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args, writer, poisoned_val_loader=None):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        
    def aggregate_updates(self, global_model, agent_updates_dict, cur_round):
        
        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)

        #Calculate the average L2 norm for all the benign agents and cut updates on that dynamic norm calculated
        if self.args.norm == "true":
            l2_norms = []
            for agent in agent_updates_dict:
                if agent > self.args.num_corrupt:
                    l2_norms.append(torch.norm(agent_updates_dict[agent], p=2).item())
            mean_norm = np.mean(l2_norms)
            for agent in agent_updates_dict:
                norm_cut = max(1, torch.norm(agent_updates_dict[agent], p=2) / mean_norm)
                agent_updates_dict[agent] = agent_updates_dict[agent] / norm_cut

        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)
        
        aggregated_updates = 0
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)
        elif self.args.aggr == 'krum':
            aggregated_updates, mal_pos = self.krum(agent_updates_dict)
        elif self.args.aggr == 'flame':
            aggregated_updates = self.flame(agent_updates_dict)
        elif self.args.aggr == 'fedinv':
            aggregated_updates = self.deep_leakage_from_gradients(agent_updates_dict, global_model)
            
        if self.args.noise > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args.noise*self.args.clip, size=(self.n_params,)).to(self.args.device))

        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        vector_to_parameters(new_global_params, global_model.parameters())
        l2_mal, l2_benign = self.plot_norms(agent_updates_dict, cur_round)

        if self.args.aggr == 'krum':
            return l2_mal, l2_benign, mal_pos 

        return l2_mal, l2_benign, 0
    
    def deep_leakage_from_gradients(self, global_model, agent): 
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)
        examples = iter(agent.train_loader)
        criterion = nn.CrossEntropyLoss().to(self.args.device)
        example_data, example_targets = next(examples)

        for i in range(self.args.local_ep):
            optimizer.zero_grad()
            pred = global_model(example_data)
            outputs = global_model(example_data)
            minibatch_loss = criterion(outputs, example_targets)
            minibatch_loss.backward()
            optimizer.step()        
        
        dummy_data = torch.randn(example_data.size())
        dummy_label =  torch.randn(example_targets.size())

        for iters in range(300):
            def closure():
                optimizer.zero_grad()
                dummy_pred = global_model(dummy_data) 
                dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1)) 
                dummy_grad = grad(dummy_loss, global_model.parameters(), create_graph=True)

                grad_diff = sum(((dummy_grad - original_dy_dx) ** 2).sum() for dummy_g, origin_g in zip(dummy_grad, original_dy_dx))
                
                grad_diff.backward()
                return grad_diff
            
            optimizer.step(closure)
            
        return  dummy_data, dummy_label
    
    def krum(self, agent_updates_dict):
        #assume a maximum of half of the agents is malicious
        num_agents = len(agent_updates_dict.keys())
        max_malicious = 2

        #make a matrix of all distance scores between each of the models
        dist_matrix = [list() for i in range(num_agents)]
        for i in range(num_agents - 1):
            score = dist_matrix[i]
            for j in range(i + 1, num_agents):
                distance = torch.dist(agent_updates_dict[i], agent_updates_dict[j]).item()
                score.append(distance)
                dist_matrix[j].append(distance)

        #for every score of the user take the sum of the most minimal scores
        k = num_agents - max_malicious - 1
        for i in range(num_agents):
            score = dist_matrix[i]
            score.sort()
            #Take only the distance of the nearest updates to create some sort of clusters of the nearest models
            dist_matrix[i] = sum(score[:k])

        #Add the gradients of the selected models with the least summed distance
        pairs = [(agent_updates_dict[i], dist_matrix[i]) for i in range(num_agents)]
        pairs.sort(key=lambda pair: pair[1])
        result = pairs[0][0]

        #take the average of all the models
        for i in range(1, max_malicious):
            result += pairs[i][0]
        result /= float(max_malicious)
        print(agent_updates_dict[0])
        print(pairs)

        for i in range(len(pairs)):
            if agent_updates_dict[0][0] == pairs[i][0][0]:
                mal_pos = i
        print(mal_pos)
        return result, mal_pos

    def flame(self, agent_updates_dict):
        """ fed avg with flame """
        update_len = len(agent_updates_dict.keys())
        weights = np.zeros((update_len, np.array(len(agent_updates_dict[0]))))
        for _id, update in agent_updates_dict.items():
            weights[_id] = update.cpu().detach().numpy()  # np.array
        # grad_in = weights.tolist()  #list
        benign_id = self.flame_filter(weights, cluster_sel=0)
        print('!!!FLAME: remained ids are:')
        print(benign_id)
        accepted_models_dict = {}
        for i in range(len(benign_id)):
            accepted_models_dict[i] = torch.tensor(weights[benign_id[i], :]).to(self.args.device)
        sm_updates, total_data = 0, 0
        for _id, update in accepted_models_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

    def flame_filter(self, grad_in, cluster_sel=0):
        distance_matrix = pairwise_distances(grad_in, metric='cosine')
        cluster_base = hdbscan.HDBSCAN(
            #metric='l2',
            metric = 'precomputed',
            min_cluster_size=int(self.args.num_agents/2 + 1),  # the smallest size grouping that you wish to consider a cluster
            allow_single_cluster=True,  # False performs better in terms of Backdoor Attack
            min_samples=1,  # how conservative you want you clustering to be
        #    cluster_selection_epsilon=0,
        )
        cluster_lastLayer = hdbscan.HDBSCAN(
            metric='l2',
            min_cluster_size=2,
            allow_single_cluster=True,
            min_samples=1,
        )
        if cluster_sel == 0:
            cluster = cluster_base
        elif cluster_sel == 1:
            cluster = cluster_lastLayer
        cluster.fit(distance_matrix)
        label = cluster.labels_
        print("label: ",label)
        if (label == -1).all():
            bengin_id = np.arange(len(distance_matrix)).tolist()
        else:
            label_class, label_count = np.unique(label, return_counts=True)
            if -1 in label_class:
                label_class, label_count = label_class[1:], label_count[1:]
            majority = label_class[np.argmax(label_count)]
            bengin_id = np.where(label == majority)[0].tolist()

        return bengin_id

    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr                                            
        return sm_of_signs.to(self.args.device)
        
    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            if self.args.data != 'reddit':
                n_agent_data = self.agent_data_sizes[_id]
            else:
                n_agent_data = 1
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/self.args.clip))
        return
                  
    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])
                              
        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        print(avg_l2_honest_updates)
        
        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
            print(avg_l2_corrupt_updates)
            return avg_l2_corrupt_updates, avg_l2_honest_updates
        return 0, avg_l2_honest_updates