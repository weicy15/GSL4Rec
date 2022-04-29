import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
import torch.nn.init as init
import numpy as np
import dgl


class GCN(nn.Module):
    def __init__(self, input_size, output_size, k_node2choose):
        super(GCN, self).__init__()
        self.k_node2choose = k_node2choose
        self.fc = nn.Linear(input_size, output_size, bias= False)
    
    def message_func(self, edges):
        return {'source_id': edges.src['id'], 'source_information': edges.src['buffer'], 'interact_score': edges.data['interact_score'], 'initial_score': edges.data['initial_score']}

    def reduce_func(self, nodes):
        _, indices = nodes.mailbox['interact_score'].topk(self.k_node2choose, dim= -1)
        selected_ids = torch.gather(nodes.mailbox['source_id'], 1, indices)

        initial_score = nodes.mailbox['initial_score']

        mask = nodes.mailbox['interact_score'].new_zeros(nodes.mailbox['interact_score'].shape)
        mask = mask.scatter(1, indices, 1)

        pseudo_adj = torch.sigmoid(nodes.mailbox['interact_score'] - (1-mask) * 100000000000)
        pseudo_adj = (1-self.keep_rate) * pseudo_adj + self.keep_rate * initial_score

        coefficient = pseudo_adj.sum(dim=-1, keepdim= False) + 1

        friends_info = torch.mul(nodes.mailbox['source_information'], pseudo_adj.unsqueeze(-1)).sum(dim=-2, keepdim=False)
        self_info = nodes.data['buffer']

        aggregate = torch.div((friends_info+self_info), coefficient.unsqueeze(-1))
        h = torch.relu(self.fc(aggregate))
        return {'h': aggregate, 'selected_ids': selected_ids}

    def forward(self, G, keep_rate):
        self.keep_rate = keep_rate
        G.update_all(self.message_func, self.reduce_func)
        return G.ndata.pop('h'), G.ndata.pop('selected_ids')


class EDGE_INFERENCE(nn.Module):
    def __init__(self, embedding_size):
        super(EDGE_INFERENCE, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * embedding_size, embedding_size, bias=True), nn.ReLU(), nn.Linear(embedding_size, 1, bias=False))

    def edge_inference(self, edges):

        src_representation = edges.src['short_term_representation']
        dst_representation = edges.dst['short_term_representation']

        initial_score = edges.data['initial_score']

        interact_with_neighbor = torch.cat((src_representation, src_representation-dst_representation), dim= -1)

        interact_score = self.mlp(interact_with_neighbor).squeeze(-1)
        return {'interact_score': interact_score}
        
    def forward(self, G_batch, keep_rate):
        self.keep_rate = keep_rate
        G_batch.apply_edges(self.edge_inference)
        return G_batch


class ROUGH_FILTER(nn.Module):
    def __init__(self, user_num, embedding_size):
        super(ROUGH_FILTER, self).__init__()
        self.in_user_embedding = nn.Embedding(user_num, embedding_size)

    def forward(self, out_user_embedding_weight):
        score = torch.mm(self.in_user_embedding.weight, out_user_embedding_weight.permute(1, 0))
        score = torch.tanh(score)
        score = torch.relu(score)

        return score

class PROTOTYPE(nn.Module):
    def __init__(self, user_num, item_num, embedding_size, L, dropout_hidden, k_node2choose=20):
        super(PROTOTYPE, self).__init__()

        self.user_embedding = nn.Embedding(user_num, embedding_size)
        self.item_embedding = nn.Embedding(item_num, embedding_size)
        self.embedding_size = embedding_size

        self.lstm_dynamic = nn.LSTM(embedding_size, embedding_size, dropout= dropout_hidden)

        self.egde_inference = EDGE_INFERENCE(embedding_size)

        self.gat_list = nn.ModuleList([GCN(embedding_size, embedding_size, k_node2choose) for i in range(L)])

        self.W1 = nn.Linear(2 * embedding_size, embedding_size, bias= False)
        self.W2 = nn.Linear(2 * embedding_size, embedding_size, bias= False)

    def forward(self, G_batch, keep_rate=0):
        seq_embedding = self.item_embedding(G_batch.ndata['session']).permute(1, 0, 2)
        packed = nn.utils.rnn.pack_padded_sequence(seq_embedding, G_batch.ndata['length'].cpu(), enforce_sorted= False)

        _, (user_represent, _) = self.lstm_dynamic(packed)
        user_represent = user_represent.squeeze(0)

        G_batch.ndata['short_term_representation'] = user_represent
        G_batch = self.egde_inference(G_batch, keep_rate)

        friend_index = torch.nonzero((1 - G_batch.ndata['is_core'])).squeeze(-1)

        friend_longterm = self.user_embedding(torch.index_select(G_batch.ndata['id'], 0, friend_index))
        
        friend_shortterm = torch.index_select(user_represent, 0, friend_index)

        friend_represent = functional.relu(self.W1(torch.cat((friend_longterm, friend_shortterm), dim=-1)))

        user_represent[friend_index, :] = friend_represent


        G_batch.ndata['buffer'] = user_represent

        for index, propogate_layer in enumerate(self.gat_list):
            G_batch.ndata['buffer'], selected_ids = propogate_layer(G_batch, keep_rate)

        final_present = G_batch.ndata.pop('buffer')

        core_user_index = torch.nonzero(G_batch.ndata['is_core']).squeeze(-1)

        selected_ids = torch.index_select(selected_ids, 0, core_user_index)
        core_user = torch.index_select(G_batch.ndata['id'], 0, core_user_index)

        core_user_social_influence = torch.index_select(final_present, 0, core_user_index)
        core_user_recent_behavior = torch.index_select(user_represent, 0, core_user_index)
        cat = torch.cat((core_user_recent_behavior, core_user_social_influence), dim=-1)
        core_user_represent = core_user_recent_behavior + core_user_social_influence 
        score = torch.matmul(core_user_represent, self.item_embedding.weight.permute(1, 0))
        score = functional.softmax(score, dim=-1)

        return score, selected_ids, core_user










