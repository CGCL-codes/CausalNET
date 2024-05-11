import os
import tqdm
import numpy as np
import argparse
import torch

from os.path import join as opj
from os.path import dirname as opd
from omegaconf import OmegaConf
from torch import nn
from datetime import datetime

from utils.util_logger import MyLogger
from utils.util_dataset import get_formated_dataset, training_data_generater
from utils.util_gumbel import gumbel_softmax
from utils.util_misc import plot_causal_matrix, reproduc, plot_causal_matrix_in_training, calc_and_log_metrics
from transformer.Models import *
from utils.util_loss import *
from utils.util_os import count_subdirectories, save_graph_or_decay

# Project Dir
proj_dir = '/home/user_name/causalnet'
# Dataset Dir
dataset_dir = '/home/user_name/causalnet/datasets/'


class CausalNET(object):
    def __init__(self, num_types, topology, prior_cm, args, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device
        self.epochs = self.args.total_epoch
        
        self.pred_model = Transformer(
            d_model=self.args.data_pred.d_model,
            d_inner=self.args.data_pred.d_inner_hid,
            n_layers=self.args.data_pred.n_layers,
            n_head=self.args.data_pred.n_head,
            d_k=self.args.data_pred.d_k,
            d_v=self.args.data_pred.d_v,
            dropout=self.args.data_pred.dropout,
            num_types=num_types,
            num_topos=topology.shape[0]
        ).to(self.device)

        self.pred_optimizer = torch.optim.Adam(self.pred_model.parameters(),
                                                    lr=self.args.data_pred.lr_data_start,
                                                    weight_decay=self.args.data_pred.weight_decay)
        
        if self.args.data_pred.smooth > 0:
            self.data_pred_loss = LabelSmoothingLoss(self.args.data_pred.smooth, num_types, ignore_index=-1)
        else:
            self.data_pred_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        
        gamma = (self.args.data_pred.lr_data_end / self.args.data_pred.lr_data_start) ** (1 / self.epochs)
        self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(self.pred_optimizer, step_size=1, gamma=gamma)
        
        
        # Part.2 Causal Graph
        # [1] Random Init
        # self.graph = nn.Parameter(2*torch.rand([self.args.dataset.n_nodes, self.args.dataset.n_nodes]).to(self.device) - 1)
        # [2] Negative Init
        # self.graph = nn.Parameter(torch.rand([self.args.dataset.n_nodes, self.args.dataset.n_nodes]).to(self.device) * (-1))
        # [3] Zero Init
        self.graph = nn.Parameter(torch.ones([self.args.dataset.n_nodes, self.args.dataset.n_nodes]).to(self.device) * 0)
        # [4] Positive Init
        # self.graph = nn.Parameter(torch.rand([self.args.dataset.n_nodes, self.args.dataset.n_nodes]).to(self.device) * (1))
        ## Optimizer
        self.graph_optimizer = torch.optim.Adam([self.graph], lr=self.args.graph_discov.lr_graph_start)
        gamma = (self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start) ** (1 / self.epochs)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)
        ## Gumbel Softmax 
        end_tau, start_tau = self.args.graph_discov.end_tau, self.args.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.epochs)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau

        
        # Part.3 Topology Graph
        topology = torch.from_numpy(topology).float().to(self.device)
        max_hop = self.args.dataset.max_hop
        ##
        self.theta = nn.Parameter(torch.ones([max_hop+1, self.args.dataset.n_topos, self.args.dataset.n_topos, self.args.dataset.n_nodes, self.args.dataset.n_nodes]).to(self.device) * 0)
        self.A = []
        for k in range(0, max_hop+1):  # k-hop
            A_k = torch.matrix_power(topology, k)
            A_k = A_k.ne(0).float()
            if k > 0:
                torch.diagonal(A_k).fill_(1)
            self.A.append(A_k)
        ##
        self.decay_optimizer = torch.optim.Adam([self.theta], lr=self.args.graph_discov.lr_decay_start)
        gamma = (self.args.graph_discov.lr_decay_end / self.args.graph_discov.lr_decay_start) ** (1 / self.epochs)
        self.decay_scheduler = torch.optim.lr_scheduler.StepLR(self.decay_optimizer, step_size=1, gamma=gamma)


        # Part.4 Loss weight
        self.lmd_l = self.args.loss.lmd_l
        self.lmd_a1 = self.args.loss.lmd_a1
        self.lmd_a2 = self.args.loss.lmd_a2
        self.lmd_a3 = self.args.loss.lmd_a3
        self.lmd_s = self.args.loss.lmd_s
        self.lmd_d = self.args.loss.lmd_d
        

    def latent_data_pred(self, event_type, event_time, event_topo, event_num):
        def sample_graph(sample_matrix, batch_size):
            """ 
            Sample a causal graph from the causal probability graph via Bernouli
            note: Bernoulli is a special case of Gumbel Softmax 
            """
            sample_matrix = torch.sigmoid(
                sample_matrix[None, :, :].expand(batch_size, -1, -1))
            return torch.bernoulli(sample_matrix)
        
        self.pred_model.train()
        self.pred_optimizer.zero_grad()
        # self.decay_optimizer.zero_grad()
        
        bs = event_num.shape[0]
        sampled_graph = sample_graph(self.graph, bs)
            
        enc_out, prediction = self.pred_model(event_num, event_type, event_time, event_topo, sampled_graph, self.theta, self.A)  #  batch*max_seq_len, including PAD

        # indices of events to be predicted
        bs, max_seq_len = enc_out.shape[0], enc_out.shape[1] - 1
        event_index = torch.arange(bs).to(self.device) * max_seq_len + (event_num - 2)
        
        # negative log-likelihood
        event_ll, non_event_ll = log_likelihood(self.pred_model, enc_out, event_time, event_type, event_index)
        event_loss = -torch.sum(event_ll - non_event_ll) 
        
        # time prediction
        se = time_loss(prediction[1], event_time, event_index)
        
        # type prediction
        pred_loss, pred_num_event = type_loss(prediction[0], event_type, event_index, self.data_pred_loss)
        pred_acc = pred_num_event / bs
        
        # topo prediction
        pass
        
        # Loss function
        loss = event_loss*self.lmd_l + se*self.lmd_a1 + pred_loss*self.lmd_a2
        
        # Optimization
        loss.backward()
        self.pred_optimizer.step()
                
        return event_loss, pred_loss, se, loss, pred_acc


    def graph_discov(self, event_type, event_time, event_topo, event_num):

        def sigmoid_gumbel_sample(graph, batch_size, tau=1):
            """ 
            Sample a causal graph from the causal probability graph via Gumbel Softmax
            """
            prob = torch.sigmoid(graph[None, :, :, None].expand(batch_size, -1, -1, -1))
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau)[:, :, :, 0]
            return samples

        # self.pred_model.eval()
        self.graph_optimizer.zero_grad()
        self.decay_optimizer.zero_grad()

        prob_graph = torch.sigmoid(self.graph[None, :, :])
        sampled_graph = sigmoid_gumbel_sample(self.graph, self.args.batch_size, tau=self.gumbel_tau)
        
        # pass through Transformer
        enc_out, prediction = self.pred_model(event_num, event_type, event_time, event_topo, sampled_graph, self.theta, self.A)  #  batch*max_seq_len, including PAD
        
        # indices of events to be predicted
        bs, max_seq_len = enc_out.shape[0], enc_out.shape[1] - 1
        event_index = torch.arange(bs).to(self.device) * max_seq_len + (event_num - 2)
        
        # negative log-likelihood
        event_ll, non_event_ll = log_likelihood(self.pred_model, enc_out, event_time, event_type, event_index)
        event_loss = -torch.sum(event_ll - non_event_ll) 
        
        # time prediction
        se = time_loss(prediction[1], event_time, event_index)
        
        # type prediction
        pred_loss, pred_num_event = type_loss(prediction[0], event_type, event_index, self.data_pred_loss)
        pred_acc = pred_num_event / bs
        
        # topo prediction
        pass
        
        # L_pred
        loss_data = event_loss*self.lmd_l + se*self.lmd_a1 + pred_loss*self.lmd_a2
        
        # L1 loss
        gs = prob_graph.shape
        loss_sparsity = torch.norm(prob_graph, p=1) / (gs[0] * gs[1] * gs[2])
        # loss_sparsity = torch.norm(prob_graph, p=1)
        
        # Dag loss
        loss_dag = compute_dag_ness(prob_graph[0])
        
        # Loss Function
        loss =  loss_data + loss_sparsity * self.lmd_s + loss_dag * self.lmd_d
        
        # Optimization
        loss.backward()
        self.graph_optimizer.step()
        self.decay_optimizer.step()
        
        return loss, loss_data, loss_sparsity, loss_dag, pred_acc


    def train(self, dataset, data, topology, true_cm=None):
        
        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.epochs)
        auc = 0
        
        for epoch_i in range(self.epochs):
            ### Stag1: given causal graph and decay matrix, learn to conduct event prediction
            data_dir = proj_dir + '/model/' + self.args.dataset.input_format + '/' + dataset + '/' + 'epoch_data' + '/'
            if self.args.dataset.input_format == 'data':
                sub_dir = str(self.epochs) + '_' + str(self.args.dataset.max_time_lag) + '_' + str(self.args.dataset.max_seq_len) + '/'
            elif self.args.dataset.input_format == 'data_topo':
                sub_dir = str(self.epochs) + '_' + str(self.args.dataset.max_hop) + '_' + str(self.args.dataset.max_time_lag) + '_' + str(self.args.dataset.max_seq_len) + '/'
            file_name = str(epoch_i) + '.pt'
            batch_gen = torch.load(data_dir+sub_dir+file_name)
            
            count = 0  # batch idx
            for event_type, event_time, event_topo, event_num in batch_gen:
                event_type = event_type.to(self.device)
                event_time = event_time.to(self.device)
                event_topo = event_topo.to(self.device)
                event_num = event_num.to(self.device)
                ## 
                latent_pred_step += self.args.batch_size
                pred_loss_likelihood, pred_loss_type, pred_loss_time, data_pred_loss, pred_acc = \
                    self.latent_data_pred(event_type, event_time, event_topo, event_num)
                # print('loss={}'.format(data_pred_loss.item()))
                self.log.log_metrics({"latent_data_pred/likelihood": pred_loss_likelihood,
                    "latent_data_pred/type_loss": pred_loss_type,
                    "latent_data_pred/time_loss": pred_loss_time,
                    "latent_data_pred/data_pred_loss": data_pred_loss,
                    "latent_data_pred/pred_acc": pred_acc}, latent_pred_step)
                pbar.set_postfix_str(f"S1 id={count} loss={data_pred_loss.item():.2f}, spr=IDLE, acc={pred_acc:.4f}")
                count += 1

            current_data_pred_lr = self.pred_optimizer.param_groups[0]['lr']
            self.log.log_metrics({"latent_data_pred/lr": current_data_pred_lr}, latent_pred_step)
            self.data_pred_scheduler.step()

            
            ### Stage2: given Transformer, optimize the causal graph and decay matrix
            count = 0  # batch idx
            for event_type, event_time, event_topo, event_num in batch_gen:
                event_type = event_type.to(self.device)
                event_time = event_time.to(self.device)
                event_topo = event_topo.to(self.device)
                event_num = event_num.to(self.device)
                ## 
                graph_discov_step += self.args.batch_size
                loss, loss_data, loss_sparsity, loss_dag, pred_acc = self.graph_discov(event_type, event_time, event_topo, event_num)
                self.log.log_metrics({"graph_discov/total_loss": loss.item(),
                                        "graph_discov/data_loss": loss_data.item(),
                                        "graph_discov/sparsity_loss": loss_sparsity.item(),
                                        "graph_discov/dag_loss": loss_dag.item(),
                                        "graph_discov/pred_acc": pred_acc,}, graph_discov_step)
                pbar.set_postfix_str(f"S2 id={count}, loss={loss_data.item():.2f}, spr={loss_sparsity.item():.2f}, acc={pred_acc:.4f}")
                count += 1
                
            self.graph_scheduler.step()
            self.decay_scheduler.step()
            current_graph_disconv_lr = self.graph_optimizer.param_groups[0]['lr']
            self.log.log_metrics({"graph_discov/lr": current_graph_disconv_lr}, graph_discov_step)
            self.log.log_metrics({"graph_discov/tau": self.gumbel_tau}, graph_discov_step)
            self.gumbel_tau *= self.gumbel_tau_gamma

            pbar.update(1)
            # self.lambda_s *= self.lambda_gamma
            # self.lambda_d *= self.lambda_gamma_d  # for DAG ness
            
            # Transform graph & theta => final graph            
            ##   a.  
            prob_mat = torch.sigmoid(self.graph.detach())
            ##   b.
            adjacency = torch.stack(self.A)  # (max_hop,n_topo,n_topo)
            decay = self.theta.detach().permute(3,4,0,1,2)  # (max_hop,n_topo,n_topo,n_type,n_type) -> (n_type,n_type,max_hop,n_topo,n_topo)
            decay_adj = decay * adjacency
            prob_decay = torch.sigmoid(decay_adj)
            prob_decay = prob_decay.permute(2,3,4,0,1)  # (n_type,n_type,max_hop,n_topo,n_topo) -> (max_hop,n_topo,n_topo,n_type,n_type)
            prob_mat_decayed = prob_mat * prob_decay
            prob_mat_decayed = prob_mat_decayed.permute(3,4,0,1,2)  # (max_hop,n_topo,n_topo,n_type,n_type) -> (n_type,n_type,max_hop,n_topo,n_topo)
            ##   c.
            prob_mat_decayed_adj = prob_mat_decayed * adjacency.ne(0).float()
            ##   d.
            prob_mat_decayed_adj_max = prob_mat_decayed_adj + torch.zeros_like(prob_mat_decayed_adj)
            for _ in range(3):
                prob_mat_decayed_adj_max, _ = torch.max(prob_mat_decayed_adj_max, dim=-1)
            ##   e.  Prior
            # prob_mat_decayed_adj_max[torch.where(self.prior_cm==0)] = 0
            # prob_mat_decayed_adj_max[torch.where(self.prior_cm==1)] = 1
            ##   f.
            prob_mat_np = prob_mat_decayed_adj_max.cpu().numpy()


            # Prune (post-processing)
            p_prob_mat = prune_by_dag_ness_soft(prob_mat_decayed_adj_max, self.args.causal_thres)
            p_prob_mat_np = p_prob_mat.cpu().numpy()
            
            # Hard Prune (post-processing)
            p_prob_mat_hard = prune_by_dag_ness_hard(prob_mat_decayed_adj_max, self.args.causal_thres)
            p_prob_mat_np_hard = p_prob_mat_hard.cpu().numpy()
            
            # Show graph
            if (epoch_i+1) % self.args.show_graph_every == 0:
                plot_causal_matrix_in_training(prob_mat_np, self.log, graph_discov_step, threshold=self.args.causal_thres, pruned=False)
                plot_causal_matrix_in_training(p_prob_mat_np, self.log, graph_discov_step, threshold=self.args.causal_thres, pruned=True, prune_strategy="soft")
                plot_causal_matrix_in_training(p_prob_mat_np_hard, self.log, graph_discov_step, threshold=self.args.causal_thres, pruned=True, prune_strategy="hard")
            
            # Show metrics
            if true_cm is not None:
                calc_and_log_metrics(prob_mat_np, true_cm, self.log, graph_discov_step, threshold=self.args.causal_thres, pruned=False)
                calc_and_log_metrics(p_prob_mat_np, true_cm, self.log, graph_discov_step, threshold=self.args.causal_thres, pruned=True, prune_strategy="soft")
                calc_and_log_metrics(p_prob_mat_np_hard, true_cm, self.log, graph_discov_step, threshold=self.args.causal_thres, pruned=True, prune_strategy="hard")
            
            # Save Graph
            if epoch_i == (self.epochs-1):
                dir_path = proj_dir + '/model/dags/final_prob/'
                data_mat = prob_mat_decayed_adj_max.cpu().numpy()
                save_graph_or_decay(dir_path=dir_path, file_name=dataset, data_type='final_prob', data_mat=data_mat)
                ##
                dir_path = proj_dir + '/model/dags/prob/'
                data_mat = torch.sigmoid(self.graph.detach()).cpu().numpy()
                save_graph_or_decay(dir_path=dir_path, file_name=dataset, data_type='prob', data_mat=data_mat)
                ##
                dir_path = proj_dir + '/model/dags/decay/'
                data_mat = torch.sigmoid(self.theta.detach()).cpu().numpy()
                save_graph_or_decay(dir_path=dir_path, file_name=dataset, data_type='decay', data_mat=data_mat)                


def preprocess_data(data, topology, opt):
    """
    Preprocess the raw dataset to obtain the training data required for each epoch.
    """
    epoch = opt.model.total_epoch
    
    data_dir = proj_dir + '/model/' + opt.model.dataset.input_format + '/' + opt.dataset + '/' + 'epoch_data' + '/'
    if opt.model.dataset.input_format == 'data':
        sub_dir = str(epoch) + '_' + str(opt.model.dataset.max_time_lag) + '_' + str(opt.model.dataset.max_seq_len) + '/'
    elif opt.model.dataset.input_format == 'data_topo':
        sub_dir = str(epoch) + '_' + str(opt.model.dataset.max_hop) + '_' + str(opt.model.dataset.max_time_lag) + '_' + str(opt.model.dataset.max_seq_len) + '/'
    if not os.path.exists(data_dir+sub_dir):
        os.makedirs(data_dir+sub_dir)
    else:
        print('This dataset is already preprocessed.')
        return 'ok'
    
    pbar = tqdm.tqdm(total=epoch)
    for epoch_i in range(epoch):
        # generate
        if opt.model.dataset.input_format == 'data':
            batch_gen = training_data_generater(data,
                    bs=opt.model.batch_size,
                    max_time_lag=opt.model.dataset.max_time_lag,
                    max_seq_len=opt.model.dataset.max_seq_len,
                    device='cpu')
        elif opt.model.dataset.input_format == 'data_topo':
            raise NotImplementedError
        batch_gen = list(batch_gen)
        # save
        file_name = str(epoch_i) + '.pt'
        torch.save(batch_gen, data_dir + sub_dir + file_name)
        pbar.set_postfix_str(f"epoch_i={epoch_i}/{epoch}")

    return 'End preprocess'


def load_formated_data(dataset_name):
    """ load dataset """
    print('[Info] Loading data...')
    data, topology, true_cm, prior_cm, num_types = get_formated_dataset(dataset_dir, dataset_name)
    
    if topology.sum() <= 0:
        print('[Info] No topology info in this dataset ...')
    
    return data, topology, true_cm, prior_cm, num_types


def main(opt, device="cuda"):
    reproduc(**opt.reproduc)
    log_id = count_subdirectories(opt.dir_name) + 1
    opt.task_name += '_' + str(log_id)
    timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    opt.task_name += timestamp
    if opt.dataset.split('/')[0] == '24V_439N_Microwave':
        opt.task_name += '_m24'
    elif opt.dataset.split('/')[0] == '25V_474N_Microwave':
        opt.task_name += '_m25'
    else:
        opt.task_name += opt.dataset.split('/')[0]
    proj_path = opj(opt.dir_name, opt.task_name)
    log = MyLogger(log_dir=proj_path, **opt.log)
    log.log_opt(opt)

    # load data
    data, topology, true_cm, prior_cm, num_types = load_formated_data(opt.dataset)
    opt.model.dataset.n_nodes = true_cm.shape[0]
    opt.model.dataset.n_topos = topology.shape[0]
    
    # preprocess
    preprocess_data(data, topology, opt)
    
    # visualize ground-truth
    if true_cm is not None:
        sub_cg = plot_causal_matrix(true_cm, figsize=[1.5*true_cm.shape[1], 1*true_cm.shape[1]])
        log.log_figures(name="True Graph", figure=sub_cg, iters=0)
    
    # training
    model = CausalNET(num_types, topology, prior_cm, opt.model, log, device=device)
    model.train(opt.dataset, data, topology, true_cm)
    

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    parser = argparse.ArgumentParser(description='Batch Compress')
    parser.add_argument('-opt', type=str, default=opj(opd(__file__),
                        './configs/config_m24.yaml'), help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list', default='0', type=str)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-log', action='store_true')
    parser.add_argument('-task', default='causalnet', type=str)
    args = parser.parse_args()

    if args.g == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = "mps"
    elif args.g == "cpu":
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        device = "cuda"
        print('Use GPU:{}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))

    main(OmegaConf.load(args.opt), device=device)
