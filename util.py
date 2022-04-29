import torch
from torch.utils.data.dataset import Dataset
import os
from io import open
import pickle
import numpy as np
import random
import dgl
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def load_all(directory, rough_filter_threshold):
    item_encoder_map = pd.read_csv(directory + '/item_encoder_map.csv')
    item_num = len(item_encoder_map)

    user_encoder_map = pd.read_csv(directory + '/user_encoder_map.csv')
    user_num = len(user_encoder_map)


    with open(directory + '/train_tri', "rb") as f:
        train_tri = pickle.load(f)

    with open(directory + '/test_tri', "rb") as f:
        test_tri = pickle.load(f)

    session_data = pd.read_pickle(directory + '/session_data.pkl')
    max_sequence_length = max(session_data['session'].apply(lambda x: len(x)))

    user_vector = np.zeros((user_num, item_num), dtype=np.int)

    neighbors_for_all_user = [[0]*rough_filter_threshold] * user_num
    for row in train_tri:
        user_id = row[0]
        social_friends = row[2]
        if len(social_friends) < rough_filter_threshold:
            social_friends = social_friends + [0] * (rough_filter_threshold - len(social_friends))
        else:
            social_friends = social_friends[:rough_filter_threshold]

        neighbors_for_all_user[user_id] = social_friends

    neighbors_for_all_user = np.array(neighbors_for_all_user)

    return train_tri, test_tri, user_num, item_num, max_sequence_length, neighbors_for_all_user



class Train_dataset(Dataset):
    def __init__(self, train_tri, max_sequence_length, similar_matrix, neighbors_for_all_user, keep_rate = 1, rough_filter_threshold= 100, k_node2choose= 30):
        super(Train_dataset, self).__init__()
        self.train_tri = train_tri
        self.max_sequence_length = max_sequence_length
        self.rough_filter_threshold= rough_filter_threshold
        self.k_node2choose = k_node2choose
        self.neighbors_for_all_user = neighbors_for_all_user
        self.keep_rate = keep_rate

        user_num = similar_matrix.shape[0]
        initial_similar_matrix = torch.zeros(user_num, user_num)
        adj_index = torch.from_numpy(neighbors_for_all_user)
        initial_similar_matrix = initial_similar_matrix.scatter_(1, adj_index, 1).numpy()

        self.similar_matrix = similar_matrix
        self.initial_similar_matrix = initial_similar_matrix

    def __len__(self):
        return len(self.train_tri)

    def __getitem__(self, idx):
        entry = self.train_tri[idx]
        user_id = entry[0]
        active_list = entry[4]
        pos_item_id = entry[1]
        cores_sessions = entry[5]

        
        if np.random.binomial(1, self.keep_rate) == 1:
            rough_filter_score = self.initial_similar_matrix[user_id, active_list].tolist()
        else:
            rough_filter_score = self.similar_matrix[user_id, active_list].tolist()

        indices = list(range(len(rough_filter_score)))
        random.shuffle(indices)
        if len(active_list) > self.rough_filter_threshold:
            index = sorted(indices, key=lambda k: rough_filter_score[k])[:self.rough_filter_threshold]
            node_list = [user_id] + [active_list[i] for i in index]
            node_sessions = [cores_sessions[0]] + [cores_sessions[i+1] for i in index]
        else:
            node_list = [user_id] + active_list
            node_sessions = cores_sessions

        num_of_nodes = len(node_list)
        src = torch.arange(num_of_nodes, dtype=torch.int32)
        dst = torch.tensor([0]*num_of_nodes, dtype=torch.int32)
        g = dgl.graph((src, dst), num_nodes = num_of_nodes)
        g = dgl.remove_self_loop(g)

        initial_connected = self.neighbors_for_all_user[user_id][:self.k_node2choose].tolist()
        initial_score = []
        for n in g.edges()[0].numpy():
            if n in initial_connected:
                initial_score.append(1)
            else:
                initial_score.append(0)

        g.edata['initial_score'] = torch.tensor(initial_score)

        g.ndata['id'] = torch.tensor(node_list)
        g.ndata['is_core'] = torch.zeros(len(node_list))
        g.ndata['is_core'][0] = 1
        session_matrix = []
        length = []
        for seq in node_sessions:
            pad_size = self.max_sequence_length - len(seq)
            seq_padded = seq + [0]*pad_size
            session_matrix.append(seq_padded)
            length.append(len(seq))

        g.ndata['session'] = torch.tensor(session_matrix)
        g.ndata['length'] = torch.tensor(length)

        return user_id, node_list, g, pos_item_id



class Test_dataset(Dataset):
    def __init__(self, test_tri, max_sequence_length, similar_matrix, rough_filter_threshold= 100):
        super(Test_dataset, self).__init__()
        self.test_tri = test_tri
        self.max_sequence_length = max_sequence_length
        self.rough_filter_threshold= rough_filter_threshold

        self.similar_matrix = similar_matrix


    def __len__(self):
        return len(self.test_tri)

    def __getitem__(self, idx):
        entry = self.test_tri[idx]
        user_id = entry[0]
        active_list = entry[5]
        item_id = entry[1]
        score = entry[2]
        cores_sessions = entry[6]

        rough_filter_score = self.similar_matrix[user_id, active_list].tolist()

        indices = list(range(len(rough_filter_score)))
        random.shuffle(indices)
        if len(active_list) > self.rough_filter_threshold:
            index = sorted(indices, key=lambda k: rough_filter_score[k])[:self.rough_filter_threshold]
            node_list = [user_id] + [active_list[i] for i in index]
            node_sessions = [cores_sessions[0]] + [cores_sessions[i+1] for i in index]
        else:
            node_list = [user_id] + active_list
            node_sessions = cores_sessions

        num_of_nodes = len(node_list)            
        src = torch.arange(num_of_nodes, dtype=torch.int32)
        dst = torch.tensor([0]*num_of_nodes, dtype=torch.int32)
        g = dgl.graph((src, dst), num_nodes = num_of_nodes)
        g = dgl.remove_self_loop(g)

        g.edata['initial_score'] = torch.ones(len(g.edges()[0].numpy()))

        g.ndata['id'] = torch.tensor(node_list)
        g.ndata['is_core'] = torch.zeros(len(node_list))
        g.ndata['is_core'][0] = 1
        session_matrix = []
        length = []
        for seq in node_sessions:
            pad_size = self.max_sequence_length - len(seq)
            seq_padded = seq + [0]*pad_size
            session_matrix.append(seq_padded)
            length.append(len(seq))

        g.ndata['session'] = torch.tensor(session_matrix)
        g.ndata['length'] = torch.tensor(length)

        return user_id, node_list, g, item_id, score



