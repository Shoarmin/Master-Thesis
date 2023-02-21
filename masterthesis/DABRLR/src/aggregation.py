import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from sklearn.metrics.pairwise import pairwise_distances
#import hdbscan
import numpy as np
from copy import deepcopy
from torch.nn import functional as F

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
            aggregated_updates = self.krum(agent_updates_dict)
        elif self.args.aggr == 'flame':
            aggregated_updates = self.flame(agent_updates_dict)
            
        if self.args.noise > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args.noise*self.args.clip, size=(self.n_params,)).to(self.args.device))
                
        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        vector_to_parameters(new_global_params, global_model.parameters())
        
        # some plotting stuff if desired
        # self.plot_sign_agreement(lr_vector, cur_global_params, new_global_params, cur_round)
        self.plot_norms(agent_updates_dict, cur_round)
        return           
     
    def krum(self, agent_updates_dict):
        #CONSIDER CHANGING THIS FUNCTION
        num_agents = len(agent_updates_dict.keys())
        max_malicious = num_agents // 2

        # Compute list of scoress
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
        for i in range(1, max_malicious):
            result += pairs[i][0]
        #take the average of all the models
        result /= float(max_malicious)
        return result

    def flame_filter(self, inputs, cluster_sel=0):
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
            cluster.fit(inputs)
            label = cluster.labels_
            print("label: ",label)
            if (label == -1).all():
                bengin_id = np.arange(len(inputs)).tolist()
            else:
                label_class, label_count = np.unique(label, return_counts=True)
                if -1 in label_class:
                    label_class, label_count = label_class[1:], label_count[1:]
                majority = label_class[np.argmax(label_count)]
                bengin_id = np.where(label == majority)[0].tolist()

            return bengin_id

    def flame(self, agent_updates_dict):
        """ fed avg with flame """
        update_len = len(agent_updates_dict.keys())
        weights = np.zeros((update_len, np.array(len(agent_updates_dict[0]))))
        for _id, update in agent_updates_dict.items():
            weights[_id] = update.cpu().detach().numpy()  # np.array
        # grad_in = weights.tolist()  #list
        distance_matrix = pairwise_distances(grad_in, metric='cosine')
        benign_id = flame_filter(distance_matrix, cluster_sel=cluster_sel)
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
        return
        
    def comp_diag_fisher(self, model_params, data_loader, adv=True):

        model = models.get_model(self.args.data)
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data
            
        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                    labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)
                
            outputs = model(inputs)
            log_all_probs = F.log_softmax(outputs, dim=1)
            target_log_probs = outputs.gather(1, labels)
            batch_target_log_probs = target_log_probs.sum()
            batch_target_log_probs.backward()
            
            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)
                
        return parameters_to_vector(precision_matrices.values()).detach()

    def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
        """ Getting sign agreement of updates between honest and corrupt agents """
        # total update for this round
        update = new_global_params - cur_global_params
        
        # compute FIM to quantify these parameters: (i) parameters which induces adversarial mapping on trojaned, (ii) parameters which induces correct mapping on trojaned
        fisher_adv = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader)
        fisher_hon = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader, adv=False)
        _, adv_idxs = fisher_adv.sort()
        _, hon_idxs = fisher_hon.sort()
        
        # get most important n_idxs params
        n_idxs = self.args.top_frac #math.floor(self.n_params*self.args.top_frac)
        adv_top_idxs = adv_idxs[-n_idxs:].cpu().detach().numpy()
        hon_top_idxs = hon_idxs[-n_idxs:].cpu().detach().numpy()
        
        # minimized and maximized indexes
        min_idxs = (robustLR == -self.args.server_lr).nonzero().cpu().detach().numpy()
        max_idxs = (robustLR == self.args.server_lr).nonzero().cpu().detach().numpy()
        
        # get minimized and maximized idxs for adversary and honest
        max_adv_idxs = np.intersect1d(adv_top_idxs, max_idxs)
        max_hon_idxs = np.intersect1d(hon_top_idxs, max_idxs)
        min_adv_idxs = np.intersect1d(adv_top_idxs, min_idxs)
        min_hon_idxs = np.intersect1d(hon_top_idxs, min_idxs)
       
        # get differences
        max_adv_only_idxs = np.setdiff1d(max_adv_idxs, max_hon_idxs)
        max_hon_only_idxs = np.setdiff1d(max_hon_idxs, max_adv_idxs)
        min_adv_only_idxs = np.setdiff1d(min_adv_idxs, min_hon_idxs)
        min_hon_only_idxs = np.setdiff1d(min_hon_idxs, min_adv_idxs)
        
        # get actual update values and compute L2 norm
        max_adv_only_upd = update[max_adv_only_idxs] # S1
        max_hon_only_upd = update[max_hon_only_idxs] # S2
        
        min_adv_only_upd = update[min_adv_only_idxs] # S3
        min_hon_only_upd = update[min_hon_only_idxs] # S4


        #log l2 of updates
        max_adv_only_upd_l2 = torch.norm(max_adv_only_upd).item()
        max_hon_only_upd_l2 = torch.norm(max_hon_only_upd).item()
        min_adv_only_upd_l2 = torch.norm(min_adv_only_upd).item()
        min_hon_only_upd_l2 = torch.norm(min_hon_only_upd).item()
       
        self.writer.add_scalar(f'Sign/Hon_Maxim_L2', max_hon_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Maxim_L2', max_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Minim_L2', min_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Minim_L2', min_hon_only_upd_l2, cur_round)
        
        
        net_adv =  max_adv_only_upd_l2 - min_adv_only_upd_l2
        net_hon =  max_hon_only_upd_l2 - min_hon_only_upd_l2
        self.writer.add_scalar(f'Sign/Adv_Net_L2', net_adv, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Net_L2', net_hon, cur_round)
        
        self.cum_net_mov += (net_hon - net_adv)
        self.writer.add_scalar(f'Sign/Model_Net_L2_Cumulative', self.cum_net_mov, cur_round)
        return