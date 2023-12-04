# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(abs(x-y))

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))



class MF(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
           
    def fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)

                xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.linear_1(z_emb))

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class Embedding_Sharing(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        if is_training:
            return torch.squeeze(z_emb), U_emb, V_emb
        else:
            return torch.squeeze(z_emb)        
    
    
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size / 2, bias = False)
        self.linear_2 = torch.nn.Linear(input_size / 2, 1, bias = True)
        self.xent_func = torch.nn.BCELoss()        
    
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        
        return torch.squeeze(x)    
    
class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")        

    
    def fit(self, x, y, gamma,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        
        num_sample = len(x)
        total_batch = num_sample // self.batch_size
        early_stop = 0              

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop)

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()        

class MF_ASIPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction1_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction2_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()

                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ASIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS-PS] Reach preset epochs, it seems does not converge.")        

    
    def fit(self, x, y, gamma, tao, G = 4,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer_prediction1 = torch.optim.Adam(
            self.prediction1_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction2 = torch.optim.Adam(
            self.prediction2_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction1_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop)

                loss = xent_loss

                optimizer_prediction1.zero_grad()
                loss.backward()
                optimizer_prediction1.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred1] Reach preset epochs, it seems does not converge.")

        early_stop = 0
        last_loss = 1e9
        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction2_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop)

                loss = xent_loss

                optimizer_prediction2.zero_grad()
                loss.backward()
                optimizer_prediction2.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred2] Reach preset epochs, it seems does not converge.")
        
        early_stop = 0
        last_loss = 1e9
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                pred_u1 = self.prediction1_model.forward(x_sampled)
                pred_u2 = self.prediction2_model.forward(x_sampled)

                x_sampled_common = x_sampled[(pred_u1.detach().cpu().numpy() - pred_u2.detach().cpu().numpy()) < tao]

                pred_u3 = self.prediction_model.forward(x_sampled_common)

                sub_y = self.prediction1_model.forward(x_sampled_common)

                xent_loss = F.binary_cross_entropy(pred_u3, sub_y.detach())

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()    
    
class MF_SNIPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-SNIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS-PS] Reach preset epochs, it seems does not converge.")        
                
    def fit(self, x, y, gamma,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=True):

        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)
                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop, reduction = "sum")
                
                xent_loss = xent_loss / (torch.sum(inv_prop))

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()        
    
    
    
    
class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()

                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-PS] Reach preset epochs, it seems does not converge.")        

    def fit(self, x, y, prior_y, gamma,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, G = 1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size
        
        prior_y = prior_y.mean()
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)  

                x_sampled = x_all[ul_idxs[G * idx* self.batch_size: G * (idx+1)*self.batch_size]] # batch size

                pred_ul,_,_ = self.prediction_model.forward(x_sampled, True)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                
                imputation_y = torch.Tensor([prior_y]* G *selected_idx.shape[0]).cuda()
                imputation_loss = F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction="sum") # e^ui

                ips_loss = (xent_loss - imputation_loss) # batch size

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y,reduction="sum")

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()

class MF_CVIB(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_CVIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        alpha=0.1, gamma=0.01,
        tol=1e-4, verbose=True):

        self.alpha = alpha
        self.gamma = gamma

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred,sub_y)

                # pair wise loss
                x_sampled = x_all[ul_idxs[idx* batch_size:(idx+1)*batch_size]]

                pred_ul,_,_ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                logp_hat = pred.log()

                pred_avg = pred.mean()
                pred_ul_avg = pred_ul.mean()

                info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (1-pred_ul_avg).log()) + self.gamma* torch.mean(pred * logp_hat)

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-CVIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

class MF_DR_JL(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
        
        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)

                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()

                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DRJL-PS] Reach preset epochs, it seems does not converge.")        

    def fit(self, x, y, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch): 
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()                
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                 
                ips_loss = (xent_loss - imputation_loss) # batch size
                                
                # direct loss                
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                                                           
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)

                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
    

class MF_MRDR_JL(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDRJL-PS] Reach preset epochs, it seems does not converge.")        


    def fit(self, x, y, stop = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch): 
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)             
                
                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
          
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                
                # direct loss
                
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                 
                # propensity loss
                
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                     
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2
                            ) * (inv_prop.detach())**2 *(1-1/inv_prop.detach())).sum()   

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")
                
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()            
        
class MF_Multi_IPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size_prop
        self.embedding_sharing = Embedding_Sharing(self.num_users, self.num_items, self.embedding_k)
        self.propensity_model = MLP(input_size = 2 * embedding_k)
        self.prediction_model = MLP(input_size = 2 * embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, verbose=True): 

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                commom_emb = self.embedding_sharing.forward(sub_x)
                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(commom_emb), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(commom_emb)

                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-Multi-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Multi-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Multi-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.embedding_sharing.forward(x)
        pred = self.prediction_model.forward(pred)
        return pred.detach().cpu().numpy()        

class MF_Multi_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size_prop
        self.propensity_model = MF_BaseModel(num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.prediction_model = MF_BaseModel(num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, G = 4,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1, lamb_es = 0, lamb_prop = 0, lamb_pred = 0, lamb_imp = 0, 
        tol=1e-4, verbose=True): 

        optimizer_prop = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr, weight_decay=lamb_prop)
        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb_pred)
        optimizer_imp = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb_imp)
        
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(sub_x)
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                
                imputation_y = self.imputation_model.forward(sub_x)                
                imputation_loss = -torch.sum(pred * torch.log(imputation_y + 1e-6) + (1-pred) * torch.log(1 - imputation_y + 1e-6))
                
                ips_loss = xent_loss - imputation_loss
                
                x_all_idx = ul_idxs[G * idx * self.batch_size : G * (idx+1) * self.batch_size]
                x_sampled = x_all[x_all_idx]
                
                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.forward(x_sampled)
                
                direct_loss = -torch.sum(pred_u * torch.log(imputation_y1 + 1e-6) + (1-pred_u) * torch.log(1 - imputation_y1 + 1e-6))

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prop.zero_grad()
                optimizer_pred.zero_grad()
                optimizer_imp.zero_grad()
                loss.backward()

                optimizer_prop.step()
                optimizer_pred.step()
                optimizer_imp.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-Multi-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Multi-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Multi-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        # pred = self.embedding_sharing.forward(x)
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()       
    
    
class MF_ESMM(nn.Module):
    def __init__(self, num_users, num_items, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.propensity_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, alpha = 1, stop = 5,
        num_epoch=1000, lr=0.05, lamb_pred=0, lamb_prop=0,gamma = 0.1,
        tol=1e-4, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb_pred)
        optimizer_propensity = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr, weight_decay=lamb_prop)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(obs)
        total_batch = num_sample // self.batch_size
        early_stop = 0

        for epoch in range(num_epoch): 
            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size : (idx+1) * self.batch_size]
                x_sampled = x_all[x_all_idx]
                
                # ctr loss
                
                prop = torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                sub_y = torch.Tensor(y[x_all_idx]).cuda()
                
                prop_loss = F.binary_cross_entropy(prop, sub_obs)                                    
                
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(prop * pred, sub_y)                          
                
                loss = alpha * prop_loss + pred_loss

                optimizer_prediction.zero_grad()
                optimizer_propensity.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                optimizer_propensity.step()
                     
                epoch_loss += loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESMM] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESMM] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESMM] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
    
    
class MF_ESCM2_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)        
        self.propensity_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, alpha = 1, beta = 1, theta = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1, lamb_prop = 0, lamb_pred = 0, lamb_imp = 0,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prop = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr, weight_decay=lamb_prop)
        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb_pred)
        optimizer_imp = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb_imp)

        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x).cuda()                
                
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.forward(x_sampled).cuda()
                
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                imputation_loss = -torch.sum(imputation_y * torch.log(pred + 1e-6) + (1-imputation_y) * torch.log(1 - pred + 1e-6))
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                # direct loss
                                
                direct_loss = -torch.sum(imputation_y1 * torch.log(pred_u + 1e-6) + (1-imputation_y1) * torch.log(1 - pred_u + 1e-6))
                
                # propensity loss
                
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = -sub_y * torch.log(pred + 1e-6) - (1-sub_y) * torch.log(1 - pred + 1e-6)
                e_hat_loss = -imputation_y * torch.log(pred + 1e-6) - (1-imputation_y) * torch.log(1 - pred + 1e-6)
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()

                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()
                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)

                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                loss = alpha * prop_loss + beta * pred_loss + theta * imp_loss + dr_loss

                optimizer_pred.zero_grad()
                optimizer_imp.zero_grad()
                optimizer_prop.zero_grad()
                loss.backward()
                optimizer_pred.step()
                optimizer_imp.step()
                optimizer_prop.step()
                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
    
    
class MF_ESCM2_IPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, alpha = 1, beta = 1, theta = 1,
        num_epoch=1000, lr=0.05, lamb=0, lamb_prop=0, lamb_pred = 0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer_pred = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prop = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)                          
                                       
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
           
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()

                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()
                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)

                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    

                pred = self.prediction_model.forward(x_sampled)

                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                loss = alpha * prop_loss + beta * pred_loss + theta * xent_loss

                optimizer_pred.zero_grad()
                optimizer_prop.zero_grad()
                loss.backward()
                optimizer_pred.step()
                optimizer_prop.step()  

                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESCM2-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2-IPS] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()

class MF_DT_IPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, embedding_k1 = 8, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.embedding_k1 = embedding_k1
        self.batch_size = batch_size
        
        self.propensity_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k1)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, G = 4, alpha = 1, beta = 1,
        num_epoch=1000, lr=0.05, gamma = 0.05, lamb = 0, lamb_prop=0, lamb_pred=0, 
        tol=1e-4, verbose=True): 

        optimizer_prop = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr, weight_decay=lamb_prop)
        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb_pred)

        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items) # [0, 0], [0,1], ...[0, 299], [1, 0], ...
        num_sample = len(x) #sum(O)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]

                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
        
                x_sampled = x_all[x_all_idx]                
                
                # propensity score
                
                _, u_emb_prop1, v_emb_prop1 = self.prediction_model.forward(x_sampled, True) # A
                _, u_emb_prop2, v_emb_prop2 = self.propensity_model.forward(x_sampled, True) # K - A
                
                u_emb_prop = torch.cat([u_emb_prop1, u_emb_prop2], dim = 1)
                v_emb_prop = torch.cat([v_emb_prop1, v_emb_prop2], dim = 1)
                pred_prop = self.sigmoid(torch.sum(u_emb_prop.mul(v_emb_prop), 1))
                
                prop_loss = F.binary_cross_entropy(pred_prop, sub_obs)

                pred = self.prediction_model.forward(sub_x, False)
                sub_y = torch.Tensor(sub_y).cuda()
        
                _, u_emb_prop_obs1, v_emb_prop_obs1 = self.prediction_model.forward(sub_x, True) # A
                _, u_emb_prop_obs2, v_emb_prop_obs2 = self.propensity_model.forward(sub_x, True) # K - A
                
                u_emb_prop_obs = torch.cat([u_emb_prop_obs1, u_emb_prop_obs2], dim = 1) # K
                v_emb_prop_obs = torch.cat([v_emb_prop_obs1, v_emb_prop_obs2], dim = 1) # K
                pred_prop_obs = self.sigmoid(torch.sum(u_emb_prop_obs.mul(v_emb_prop_obs), 1))
                
                inv_prop_obs = 1/torch.clip(pred_prop_obs, gamma, 1)

                pred_loss = torch.mean((-sub_y * torch.log(pred+1e-6) - (1 - sub_y) * torch.log(1 - pred+1e-6)) * inv_prop_obs)
                
                U_penalty = torch.mm(u_emb_prop1.T, u_emb_prop2)
                V_penalty = torch.mm(v_emb_prop1.T, v_emb_prop2)

                U_penalty_orig = torch.mean(U_penalty ** 2)
                V_penalty_orig = torch.mean(V_penalty ** 2)
                F_D = U_penalty_orig + V_penalty_orig
                U_penalty = U_penalty_orig
                V_penalty = V_penalty_orig

                U_penalty += torch.mean(u_emb_prop ** 2)
                V_penalty += torch.mean(v_emb_prop ** 2)
                
                reg_loss_D = U_penalty + V_penalty

                U_penalty_obs = torch.mm(u_emb_prop_obs1.T, u_emb_prop_obs2)
                V_penalty_obs = torch.mm(v_emb_prop_obs1.T, v_emb_prop_obs2)

                U_penalty_obs_orig = torch.mean(U_penalty_obs ** 2)
                V_penalty_obs_orig = torch.mean(V_penalty_obs ** 2)
                F_O = U_penalty_obs_orig + V_penalty_obs_orig
                U_penalty_obs = U_penalty_obs_orig
                V_penalty_obs = V_penalty_obs_orig

                U_penalty_obs += torch.mean(u_emb_prop_obs ** 2)
                V_penalty_obs += torch.mean(v_emb_prop_obs ** 2)
                
                reg_loss_O = U_penalty_obs + V_penalty_obs                
                
                loss = pred_loss + alpha * prop_loss + beta * (reg_loss_D + reg_loss_O)
                
                optimizer_prop.zero_grad()
                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_prop.step()
                optimizer_pred.step()
                
                epoch_loss += loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-L-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-L-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-L-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()
    
    
    
class MF_DT_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, embedding_k1 = 8, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.embedding_k1 = embedding_k1
        self.batch_size = batch_size
        
        self.propensity_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k1)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k1)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, G = 4, alpha = 1, beta = 1, theta = 1,
        num_epoch=1000, lr=0.05, lamb_prop=0, lamb_pred = 0, lamb_imp = 0, gamma = 0.05, beta1 = 1,
        tol=1e-4, verbose=True): 

        optimizer_prop = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr, weight_decay=lamb_prop)
        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb_pred)
        optimizer_imp = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb_imp)

        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]

                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                x_sampled = x_all[x_all_idx]                
                
                pred = self.prediction_model.forward(sub_x, False)
                pred_u = self.prediction_model.forward(x_sampled, False)
                
                imputation_y = self.imputation_model.forward(sub_x, False)
                imputation_y1 = self.imputation_model.forward(x_sampled, False)
                
                sub_y = torch.Tensor(sub_y).cuda()
                # propensity score
                
                _, u_emb_prop1, v_emb_prop1 = self.prediction_model.forward(x_sampled, True) # A
                _, u_emb_prop2, v_emb_prop2 = self.propensity_model.forward(x_sampled, True) # K - A
                
                u_emb_prop = torch.cat([u_emb_prop1, u_emb_prop2], dim = 1)
                v_emb_prop = torch.cat([v_emb_prop1, v_emb_prop2], dim = 1)
                pred_prop = self.sigmoid(torch.sum(u_emb_prop.mul(v_emb_prop), 1))

                prop_loss = F.binary_cross_entropy(pred_prop, sub_obs)
                
                inv_prop = 1/torch.clip(pred_prop, gamma, 1)
                
                _, u_emb_prop_obs1, v_emb_prop_obs1 = self.prediction_model.forward(sub_x, True) # A
                _, u_emb_prop_obs2, v_emb_prop_obs2 = self.propensity_model.forward(sub_x, True) # K - A
                
                u_emb_prop_obs = torch.cat([u_emb_prop_obs1, u_emb_prop_obs2], dim = 1)
                v_emb_prop_obs = torch.cat([v_emb_prop_obs1, v_emb_prop_obs2], dim = 1)
                pred_prop_obs = self.sigmoid(torch.sum(u_emb_prop_obs.mul(v_emb_prop_obs), 1))
                
                inv_prop_obs = 1/torch.clip(pred_prop_obs, gamma, 1)
                

                xent_loss = torch.sum((sub_y * torch.log(pred+1e-6) + (1-sub_y) * torch.log(1 - pred+1e-6)) * inv_prop_obs)
                imputation_loss = torch.sum((imputation_y * torch.log(pred+1e-6) + (1-imputation_y) * torch.log(1 - pred+1e-6)) * inv_prop_obs) 

                ips_loss = xent_loss - imputation_loss # batch size
                
                direct_loss = -torch.sum(imputation_y1 * torch.log(pred_u+1e-6) + (1-imputation_y1) * torch.log(1 - pred_u+1e-6))
                dr_loss = (-ips_loss + direct_loss)/x_sampled.shape[0]

                e_loss = -sub_y * torch.log(pred+1e-6) - (1-sub_y) * torch.log(1 - pred+1e-6)
                e_hat_loss = -imputation_y * torch.log(pred+1e-6) - (1-imputation_y) * torch.log(1 - pred+1e-6)
                
                imp_loss = torch.mean(((e_loss - e_hat_loss) ** 2) * inv_prop_obs)
                
                U_penalty = torch.mm(u_emb_prop1.T, u_emb_prop2)
                V_penalty = torch.mm(v_emb_prop1.T, v_emb_prop2)
                
                U_penalty_orig = torch.mean(U_penalty ** 2)
                V_penalty_orig = torch.mean(V_penalty ** 2)
                F_D = U_penalty_orig + V_penalty_orig
                U_penalty = U_penalty_orig
                V_penalty = V_penalty_orig

                U_penalty += torch.mean(u_emb_prop ** 2)
                V_penalty += torch.mean(v_emb_prop ** 2)
                
                reg_loss_D = U_penalty + V_penalty
                
                U_penalty_obs = torch.mm(u_emb_prop_obs1.T, u_emb_prop_obs2)
                V_penalty_obs = torch.mm(v_emb_prop_obs1.T, v_emb_prop_obs2)
                
                U_penalty_obs_orig = torch.mean(U_penalty_obs ** 2)
                V_penalty_obs_orig = torch.mean(V_penalty_obs ** 2)
                F_O = U_penalty_obs_orig + V_penalty_obs_orig
                U_penalty_obs = U_penalty_obs_orig
                V_penalty_obs = V_penalty_obs_orig

                U_penalty_obs += torch.mean(u_emb_prop_obs ** 2)
                V_penalty_obs += torch.mean(v_emb_prop_obs ** 2)
                
                reg_loss_O = U_penalty_obs + V_penalty_obs  

                loss = dr_loss + theta * imp_loss + alpha * prop_loss + beta * (reg_loss_D + reg_loss_O)

                optimizer_prop.zero_grad()
                optimizer_pred.zero_grad()
                optimizer_imp.zero_grad()
                loss.backward()
                optimizer_prop.step()
                optimizer_pred.step()
                optimizer_imp.step()
                
                epoch_loss += loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-L-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss
            
            if epoch % 10 == 0 and verbose:
                print("[MF-L-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-L-DR] Reach preset epochs, it seems does not converge.")


    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()
    

class MF_DR_BIAS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  
    
    
class MF_DR_MSE(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, gamma = 1,
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                imp_bias_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).sum()
                imp_mrdr_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2 ) * (1 - 1 / inv_prop.detach())).sum()
                imp_loss = gamma * imp_bias_loss + (1-gamma) * imp_mrdr_loss
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl      


class MF_DIB(nn.Module):
    def __init__(self, num_users, num_items, batch_size,embedding_k=4, *args, **kwargs):
        super(MF_DIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.bias_user_emd = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.unbias_user_emd = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.bias_item_emd = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.unbias_item_emd = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

           
    def fit(self, x, y,
        num_epoch=1000, lr=0.05,  
        alpha=0.1, gamma=0.2,
        lamb=1e-3,tol=1e-5,verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]

                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                sub_user_idx = torch.LongTensor(sub_x[:,0]).cuda()
                sub_item_idx = torch.LongTensor(sub_x[:,1]).cuda()

                user_emb_bias = self.bias_user_emd(sub_user_idx)
                user_emb_unbias = self.unbias_user_emd(sub_user_idx)
                item_emb_bias = self.bias_item_emd(sub_item_idx)
                item_emb_unbias = self.unbias_item_emd(sub_item_idx)
                user_emb =  user_emb_bias + user_emb_unbias
                item_emb =  item_emb_bias + item_emb_unbias

                y_hat_unbias = self.sigmoid(torch.sum(user_emb_unbias.mul(item_emb_unbias), 1))
                y_hat_bias = self.sigmoid(torch.sum(user_emb_bias.mul(item_emb_bias), 1))
                y_hat_all = self.sigmoid(torch.sum(user_emb.mul(item_emb), 1))

                loss_unbias = self.xent_func(y_hat_unbias,sub_y)
                loss_bias = self.xent_func(y_hat_bias,sub_y)
                loss_all = self.xent_func(y_hat_all,sub_y)

                loss = (1 - alpha) * loss_unbias + gamma * loss_bias + alpha * loss_all

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        user_emb_unbias = self.unbias_user_emd(user_idx)
        item_emb_unbias = self.unbias_item_emd(item_idx)
        pred = self.sigmoid(torch.sum(user_emb_unbias.mul(item_emb_unbias), 1))
        return pred.detach().cpu().numpy()

        
    
def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)

