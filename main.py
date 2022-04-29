import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import visdom
import random

from io import open
import pickle
import numpy as np
import os
import argparse
import time
import shutil
import dgl
from tqdm import tqdm

from prototype import PROTOTYPE
from prototype import ROUGH_FILTER
import prototype_util as tr
import prototype_evaluate as ev

def my_collate_train(batch):
    user_id = [item[0] for item in batch]
    node_list = [item[1] for item in batch]
    social_graphs = [item[2] for item in batch]
    pos_item_id = [item[3] for item in batch]

    social_graphs = dgl.batch(social_graphs)
    pos_item = torch.LongTensor(pos_item_id)


    return [social_graphs, pos_item]

def my_collate_test(batch):
    user_id = [item[0] for item in batch]
    node_list = [item[1] for item in batch]
    social_graphs = [item[2] for item in batch]
    item_id = [item[3] for item in batch]
    score = [item[4] for item in batch]

    social_graphs = dgl.batch(social_graphs)
    items = torch.LongTensor(item_id)
    score = torch.FloatTensor(score)

    return [social_graphs, items, score]


def my_collate_test_plus_random(batch):
    user_id = [item[0] for item in batch]
    node_list = [item[1] for item in batch]
    social_graphs = [item[2] for item in batch]
    item_id = [item[3] for item in batch]
    score = [item[4] for item in batch]

    social_graphs = social_graphs[0]
    items = torch.LongTensor(item_id)
    score = torch.FloatTensor(score)

    return [social_graphs, items, score]

def inverse_sigmoid(x, k):
    s = k / (k + np.exp(x/k))
    return s

note = ''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, default='delicious_process')
    parser.add_argument("--loadFilename", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default='PROTOTYPE/data/save')

    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epoch", type=int, default=50)

    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--k_node2choose", type=int, default=40)
    parser.add_argument("--rough_filter_threshold", type=int, default=100)
    parser.add_argument("--dropout_hidden", type=float, default=0)
    parser.add_argument("--convergence", type=float, default=700)


    parser.add_argument("--K", type=int, default=20)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--lr_decay_every_step", type=int, default=5)
    opt = parser.parse_args()
    print(opt)

    print("Loading data >>>>>")
    train_tri, test_tri, user_num, item_num, max_sequence_length, neighbors_for_all_user = tr.load_all(opt.dataset_directory, opt.rough_filter_threshold)

    initial_similar_matrix = torch.zeros(user_num, user_num)
    adj_index = torch.from_numpy(neighbors_for_all_user)
    initial_similar_matrix = initial_similar_matrix.scatter_(1, adj_index, 1).numpy()

    test_tri = test_tri[:int(len(test_tri)/2)]
    print(user_num)
    print(item_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if opt.loadFilename:
        checkpoint = torch.load(opt.loadFilename)
        sd = checkpoint['sd']
        rough_sd = checkpoint['rough_sd']
        optimizer_sd = checkpoint['opt']
        rough_optimizer_sd = checkpoint['rough_opt']

    print("building model >>>>>>>>>>>>>>>")
    model = PROTOTYPE(user_num, item_num, opt.embedding_size, opt.L, opt.dropout_hidden, k_node2choose=opt.k_node2choose)
    rough_filter = ROUGH_FILTER(user_num, opt.embedding_size)

    if opt.loadFilename:
        model.load_state_dict(sd)
        rough_filter.load_state_dict(rough_sd)

    for name, param in model.named_parameters():
        print(name)

    print('Building optimizers >>>>>>>')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_every_step, gamma=opt.lr_decay)
    filter_optimizer = optim.SGD(rough_filter.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    filter_scheduler = optim.lr_scheduler.StepLR(filter_optimizer, step_size=opt.lr_decay_every_step, gamma=opt.lr_decay)
    filter_optimizer.zero_grad()
    
    print('Start training...')
    start_epoch = 0
    if opt.loadFilename:
        checkpoint = torch.load(opt.loadFilename)
        start_epoch = checkpoint['epoch'] + 1

    param_list = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(opt.dataset_directory, opt.batch_size, opt.epoch, opt.embedding_size, opt.L, opt.dropout_hidden, opt.k_node2choose, opt.rough_filter_threshold, opt.convergence, opt.K, opt.lr, opt.weight_decay, opt.lr_decay, opt.lr_decay_every_step)

    directory = os.path.join(opt.save_dir, param_list)
    directory = directory + '-' + \
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    directory = directory + '__' + note

    os.makedirs(directory)

    model = model.to(device)
    rough_filter = rough_filter.to(device)

    best_NDCG = 0
    best_HR = 0

    best_epoch = 0

    XE_loss = nn.BCELoss(reduction='mean')
    env = visdom.Visdom(env=directory)


    panel = env.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='train_loss'))
    x_index = 0
    for epoch in range(start_epoch, opt.epoch):
        print('Building new train dataloader for this epoch>>>>>>>>>>>>>>>>>>>')
        model.eval()
        rough_filter.train()

        similar_matrix = rough_filter(model.user_embedding.weight.clone().detach())
        chosen_indice_list = []
        user_batch_list = []
        train_dataset = tr.Train_dataset(train_tri, max_sequence_length, similar_matrix.cpu().detach().numpy(), neighbors_for_all_user, keep_rate = inverse_sigmoid(x_index, opt.convergence), rough_filter_threshold= opt.rough_filter_threshold, k_node2choose = opt.k_node2choose)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=my_collate_train)
        model.train()

        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, (social_graphs, pos_item) in enumerate(train_loader):
                optimizer.zero_grad()

                social_graphs = social_graphs.to(device)
                pos_item = pos_item.to(device)

                score, chosen_indice, core_user = model(social_graphs, keep_rate= inverse_sigmoid(x_index, opt.convergence))
                
                chosen_indice_list.append(chosen_indice.clone().detach())
                user_batch_list.append(core_user.clone().detach())

                pos_prediction = torch.gather(score, 1, pos_item.unsqueeze(-1)).squeeze(-1)
                loss = -(pos_prediction.log().sum())

                print('train_loss:', loss)
                env.line(X=np.array([x_index]), Y=np.array(
                    [loss.item()/pos_prediction.shape[0]]), win=panel, update='append')
                x_index += 1
                
                loss.backward()
                optimizer.step()

                if len(chosen_indice_list) == 100:
                    chosen_indice_list_tensor = torch.cat(chosen_indice_list, dim=0)
                    user_batch_list_tensor = torch.cat(user_batch_list, dim=0)

                    chosen_indice_list = []
                    user_batch_list = []

                    pos_num = chosen_indice_list_tensor.shape[1]
                    neg_num = 19
                    unchosen_indice = []
                    for users_chosen in chosen_indice_list_tensor.detach().cpu().numpy().tolist():
                        users_unchosen = []
                        count = 0
                        while count < neg_num:
                            neg_user = random.randint(0,user_num-1)
                            if neg_user not in users_chosen:
                                users_unchosen.append(neg_user)
                                count += 1
                        unchosen_indice.append(users_unchosen)
                    
                    unchosen_indice = torch.tensor(unchosen_indice, dtype= torch.int64).to(device)

                    pos_sample = torch.index_select(similar_matrix, 0, user_batch_list_tensor)
                    pos_sample = torch.gather(pos_sample, 1, chosen_indice_list_tensor)

                    neg_sample = torch.index_select(similar_matrix, 0, user_batch_list_tensor)
                    neg_sample = torch.gather(neg_sample, 1, unchosen_indice)

                    pos_sample = pos_sample.unsqueeze(-1).expand(pos_sample.shape[0], pos_num, neg_num).reshape(-1)
                    neg_sample = neg_sample.unsqueeze(1).expand(neg_sample.shape[0], pos_num, neg_num).reshape(-1)

                    rough_loss = -(pos_sample - neg_sample).sigmoid().log().sum() + (neg_sample ** 2).sum()
                    rough_loss.backward(retain_graph=True)

                pbar.update(1)

        filter_optimizer.step()
        filter_optimizer.zero_grad()
        scheduler.step()
        filter_scheduler.step()

        print('eval {}'.format(epoch))
        model.eval()
        rough_filter.eval()

        print('Building new test dataloader for this epoch>>>>>>>>>>>>>>>>>>>')
        similar_matrix = rough_filter(model.user_embedding.weight)
        test_dataset = tr.Test_dataset(test_tri, max_sequence_length, similar_matrix, opt.rough_filter_threshold)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100, collate_fn=my_collate_test)

        NDCG, HR = ev.metric_ranking_no_random(model, test_loader, opt.K, device)
        print("NDCG@{}:{}".format(opt.K, NDCG))
        print("HR@{}:{}".format(opt.K, HR))
        if NDCG > best_NDCG:
            best_NDCG = NDCG
            best_epoch = epoch
        if HR > best_HR:
            best_HR = HR

        if not os.path.exists(directory + '/buffer'):
            os.makedirs(directory + '/buffer')
        torch.save({
            'epoch': epoch,
            'sd': model.state_dict(),
            'rough_sd': rough_filter.state_dict(),
            'opt': optimizer.state_dict(),
            'rough_opt': filter_optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(directory + '/buffer', '{}_{}_{}.tar'.format(epoch, NDCG, 'checkpoint')))

    print(opt)
    print("The best NDCG@{} is {}".format(opt.K, best_NDCG))
    print("The best HR@{} is {}".format(opt.K, best_HR))
    best_checkpoint = torch.load(os.path.join(
        directory + '/buffer', '{}_{}_{}.tar'.format(best_epoch, best_NDCG, 'checkpoint')))
    torch.save({
        'sd': best_checkpoint['sd'],
        'rough_sd': best_checkpoint['rough_sd'],
    }, os.path.join('output.tar'))

    last_checkpoint = torch.load(os.path.join(
        directory + '/buffer', '{}_{}_{}.tar'.format(epoch, NDCG, 'checkpoint')))
    torch.save({
        'sd': last_checkpoint['sd'],
        'rough_sd': last_checkpoint['rough_sd'],
    }, os.path.join(directory, 'last_{}.tar'.format(NDCG)))
    
    shutil.rmtree(directory + '/buffer')