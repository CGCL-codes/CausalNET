import pandas as pd
import numpy as np
import os
import torch
import random
from copy import deepcopy


def read_csv(path, dataset, file='alarm.csv'):
    df = pd.read_csv(os.path.join(path, dataset, file))
    return df


def read_dag(path, dataset, file='true_graph.npy'):
    if os.path.exists(os.path.join(path, dataset, file)):
        dag = np.load(os.path.join(path, dataset, file))
    else:
        df = read_csv(path, dataset)
        alarm_num = df['alarm_id'].unique().shape[0]
        dag = np.random.randint(2, size=(alarm_num, alarm_num))
    return dag


def read_prior(path, dataset, file='causal_prior.npy'):
    if os.path.exists(os.path.join(path, dataset, file)):
        prior = np.load(os.path.join(path, dataset, file))
    else:
        df = read_csv(path, dataset)
        alarm_num = df['alarm_id'].unique().shape[0]
        prior = np.full((alarm_num, alarm_num), -1)
    return prior


def read_topology(path, dataset, file='topology.npy'):
    if os.path.exists(os.path.join(path, dataset, file)):
        topology = np.load(os.path.join(path, dataset, file))
    else:
        df = read_csv(path, dataset)
        device_num = df['device_id'].unique().shape[0]
        topology = np.zeros((device_num, device_num))
    return topology


def get_formated_dataset(path, dataset, fold='fold2'):
    """
    target format: {'dim_process': x, 'train':[[{},...{}]], 'dev':[[{},...{}]], 'test':[[{},...{}]]}
    keys: time_since_start, time_since_last_event, type_event
    """
    ds = read_csv(path, dataset)
    true_cm = read_dag(path, dataset)
    topology = read_topology(path, dataset)
    prior_cm = read_prior(path, dataset)
    
    ds = ds.sort_values(by='start_timestamp')
    
    alarm_ids = ds['alarm_id'].tolist()
    device_ids = ds['device_id'].tolist()
    start_timestamps = ds['start_timestamp'].tolist()
    end_timestamps = ds['end_timestamp'].tolist()
    start_timestamp_gaps = np.concatenate((np.array([0]), ds['start_timestamp'].values[1:] - ds['start_timestamp'].values[:-1]))
    
    # Dataset 
    dataset_info = {}
    dataset_info['type_num'] = ds['alarm_id'].unique().shape[0]
    dataset_info['topology'] = topology
    dataset_info['true_cm'] = true_cm
    dataset_info['prior_cm'] = prior_cm
    seq_len = len(alarm_ids)
    sequence = []
    for event_id in range(0, seq_len):
        event_item = {'time_since_start': start_timestamps[event_id], 
                'time_since_last_event': start_timestamp_gaps[event_id],
                'type_event': alarm_ids[event_id],
                'topo': device_ids[event_id]}
        sequence.append(event_item)
    dataset_info['seqs'] = [sequence]

    return dataset_info['seqs'], dataset_info['topology'], dataset_info['true_cm'], dataset_info['prior_cm'], dataset_info['type_num']


def get_IPTV_dataset_create_by_CAUSE():
    """
    return: {"n_types":, "event_seqs":, "event_type_names":, "train_test_splits":}
    """
    input_dir = '/home/user_name/causalnet/datasets/iptv/'
    data = np.load(os.path.join(input_dir, "data.npz"), allow_pickle=True)

    return data


def get_formated_dataset_IPTV(partial=1.0):
    """
    return: data, topology, true_cm, prior_cm, num_types
    """
    IPTV_data = get_IPTV_dataset_create_by_CAUSE()
    num_seqs = IPTV_data["event_seqs"].shape[0]
    num_samples = int(num_seqs*partial)

    data = np.random.choice(IPTV_data["event_seqs"], size=num_samples, replace=False)

    topology = np.zeros((1, 1))
    true_cm = np.random.randint(2, size=(IPTV_data['n_types'], IPTV_data['n_types']))
    prior_cm = np.full((IPTV_data['n_types'], IPTV_data['n_types']), -1)
    num_types = int(IPTV_data['n_types'])

    return data, topology, true_cm, prior_cm, num_types


def visualize_event_distribution(event_time, max_time_lag, max_seq_len):
    """
    Visualize the number of events within each (N*60) second interval
    """
    import matplotlib.pyplot as plt
    
    event_count = []
    for i, time in enumerate(event_time):
        count = 0
        for j in range(i-1, -1, -1):
            lag = time - event_time[j]
            if  lag > 0 and lag <= max_time_lag:
                count += 1
            elif lag == 0:
                continue
            else:
                break
                
        event_count.append(count)

    plt.plot(event_time, event_count)
    plt.xlabel('Event Time')
    plt.ylabel('Event Count')
    plt.title('Event Distribution')
    plt.show()
    plt.savefig("event_distribution.png")

    print('mean of history events = {}'.format(np.mean(event_count)))
    print('{} events have more than {} history events'.format(np.sum(np.array(event_count) > max_seq_len), max_seq_len))
    return 'ok'


def training_data_generater(data, bs, max_time_lag, max_seq_len, device):
    """
    batch generator for Event Sequence data
    
    Input:
    - max_time_lag: the longest time distance
    Output: 
    - event_type: batch*(max_seq_len+1) 
    - event_time: batch*(max_seq_len+1)
    """
    seq_num = len(data)  # Event Sequence data: seq_num*seq_len
    
    data_type = [[] for i in range(seq_num)]
    data_time = [[] for i in range(seq_num)]
    data_topo = [[] for i in range(seq_num)]
    for seq_id,seq in enumerate(data):
        for idx,event in enumerate(seq):
            data_type[seq_id].append(event['type_event'] + 1)  # +1 because 0 corresponds to PAD
            data_time[seq_id].append(event['time_since_start'] + 1)
            data_topo[seq_id].append(event['topo'] + 1)  # +1 because 0 corresponds to PAD
            
    # @. Test
    # visualize_event_distribution(data_time[0], max_time_lag, max_seq_len)
    
    seq_2_t_list = {}
    seq_len_sum = 0
    for seq_id,seq in enumerate(data):
        random_t_list = np.arange(1, len(seq), 1).tolist()
        np.random.shuffle(random_t_list)
        seq_2_t_list[seq_id] = random_t_list
        seq_len_sum += len(random_t_list)
    
        
    # Iterable Generator
    for batch_i in range(seq_len_sum // bs // 1):   
        sample_seq_ids = np.zeros(bs, dtype=int)
        
        # Sampling
        tmp_max_seq_len = 0
        skip_batches = []
        succ_batches = []
        batch_seq_len = max_seq_len + 1
        event_type = torch.zeros((bs, batch_seq_len), dtype=torch.int32).to(device)
        event_time = torch.zeros((bs, batch_seq_len), dtype=torch.float32).to(device)
        event_topo = torch.zeros((bs, batch_seq_len), dtype=torch.int32).to(device)
        event_num = torch.zeros(bs, dtype=torch.int32).to(device)  # used to identify the event to be predicted
        for idx, seq_id in enumerate(sample_seq_ids):
            # Deal with exception due to [].pop()
            if len(seq_2_t_list[seq_id]) == 0:
                print('Warning: seq {} is empty, Skip batch {} in BATCH {} !'.format(seq_id, idx, batch_i))
                skip_batches.append(idx)
                continue                
            # Extract "the event to be predicted" and "its historical events"
            data_t = seq_2_t_list[seq_id].pop()
            data_t_time = data_time[seq_id][data_t]
            data_t_e = data_t
            data_t_s = data_t
            for t in range(data_t-1, -1, -1):
                data_t_s = t
                tmp_time = data_time[seq_id][t]
                time_lag = data_t_time - tmp_time
                if time_lag == 0:
                    # print('Warning: multi events occur at the same time !')
                    data_t_e = t
                    continue
                if time_lag > max_time_lag:
                    break
                if (data_t_e - data_t_s) == max_seq_len:
                    break
            if (data_t_e - data_t_s) == 0:
                print('Warning: less than 1 history events left, Skip batch {} in BATCH {} !'.format(idx, batch_i))
                skip_batches.append(idx)
                continue
            x_type = data_type[seq_id][data_t_s:data_t_e]
            x_time = data_time[seq_id][data_t_s:data_t_e]
            x_topo = data_topo[seq_id][data_t_s:data_t_e]
            y_type = data_type[seq_id][data_t]  # len([y])=1
            y_time = data_time[seq_id][data_t]  # len([y])=1
            y_topo = data_topo[seq_id][data_t]  # len([y])=1
            # Pad meaningless events
            seq_len = len(x_type+[y_type])
            z_type = [0] * (batch_seq_len - seq_len)
            z_time = [x_time[0]-1] * (batch_seq_len - seq_len)
            z_topo = [0] * (batch_seq_len - seq_len)
            # Construct the dataset (batch data)
            event_type[idx] = torch.tensor((x_type+[y_type]+z_type))
            event_time[idx] = torch.tensor((x_time+[y_time]+z_time))
            event_topo[idx] = torch.tensor((x_topo+[y_topo]+z_topo))
            event_num[idx] = seq_len
            succ_batches.append(idx)  # flag
            
        # Deal with the skiped batch
        non_skip_batches = list(set(range(bs)) - set(skip_batches))
        np.random.shuffle(non_skip_batches)
        for idx in skip_batches:
            impute_idx = non_skip_batches.pop()
            print('Warning: Impute batch {} by data of batch {} in BATCH {} !'.format(idx, impute_idx, batch_i))
            event_type[idx] = deepcopy(event_type[impute_idx])
            event_time[idx] = deepcopy(event_time[impute_idx])
            event_topo[idx] = deepcopy(event_topo[impute_idx])
            event_num[idx] = deepcopy(event_num[impute_idx])
    
        # time window shift & rescale
        event_time = event_time -  event_time[:,0][:,None].expand(-1,batch_seq_len) + 1
        event_time /= 60
        yield event_type, event_time, event_topo, event_num
    

        