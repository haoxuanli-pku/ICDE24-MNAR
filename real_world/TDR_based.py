# -*- coding: utf-8 -*-
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
from sklearn.metrics import roc_auc_score

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

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
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
        
    def complete(self):
        for i in range(self.embedding_k):
            complete = torch.zeros([self.num_users,self.num_items])
            for m in range(self.num_users):                   
                 complete[m, :] = self.W(torch.LongTensor([m]))[0][i]
            if i == 0:
                complete1 = complete.reshape(self.num_users*self.num_items, 1)
            else:
                complete1 = torch.cat((complete1 ,complete.reshape(self.num_users*self.num_items, 1)), dim = 1)                
                
        for i in range(self.embedding_k):
            complete = torch.zeros([self.num_users,self.num_items])
            for k in range(self.num_items):
                complete[:,k] = self.H(torch.LongTensor([k]))[0][i]
            if i == 0:
                complete2 = complete.reshape(self.num_users*self.num_items, 1)
            else:
                complete2 = torch.cat((complete2 ,complete.reshape(self.num_users*self.num_items, 1)), dim = 1)
        
        complete = torch.cat((complete1, complete2),dim = 1)

        return complete
    
    def fit(self, x, y, stop = 5,
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
                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
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
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    
class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

    

class MF_TDR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.original_model = MF(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x, y, gamma):
        self.original_model.fit(x = x, y = y, num_epoch=1000, lr=0.05, lamb=0, tol=le-4, verbose=False)
        x_train = torch.zeros([self.num_users,self.num_items])

        for i in range(len(x)):
            # x_train[x[i][0],x[i][1]] = 1
            x_train[int(x[i][0]),int(x[i][1])] = 1      
        prediction = (x_train.reshape(self.num_users*self.num_items,1)).type(torch.FloatTensor)

        x_train = self.original_model.complete().type(torch.FloatTensor).detach()
        
        optimizer = torch.optim.SGD([self.linear_1.weight, self.linear_1.bias], lr=1e-3, momentum=0.9)

        last_loss = 1e9
        early_stop = 0
        
        
        for epoch in range(1000):
            all_idx = np.arange(self.num_users*self.num_items)
            np.random.shuffle(all_idx)
            total_batch = (self.num_users*self.num_items) // self.batch_size
            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x_train = x_train[selected_idx].detach()
                sub_prediction = prediction[selected_idx]
             
                out = self.linear_1(sub_x_train)
                out = self.sigmoid(out)
                loss = self.xent_func(out, sub_prediction)
                
                xent_loss = loss
                
                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()

            if (epoch + 1) % 15 == 0:
                print('*'*10)
                print('epoch {}'.format(epoch+1))
              
            if loss > last_loss:
                early_stop += 1 
            else:
                last_loss = loss
            
            if early_stop >= 5:
                break
        
        x_train = x_train.detach()
        propensity = self.sigmoid(self.linear_1(x_train.detach())) 
        propensity[np.where(propensity.cpu() <= gamma)] = gamma
        one_over_zl = 1 / propensity
        return prediction, one_over_zl  

    def fit(self, x, y, prior_y, gamma, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, G = 1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb) 

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        observation, one_over_zl = self._compute_IPS(x, y, gamma)
        one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        prior_y = prior_y.mean()
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

                # propensity score
                inv_prop = one_over_zl_obs[selected_idx].detach()

                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)  
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[G * idx* self.batch_size: G * (idx+1)*self.batch_size]]

                pred_ul,_,_ = self.prediction_model.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")               
                
                imputation_y = torch.Tensor([prior_y]* G *selected_idx.shape[0])
                imputation_loss = F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction="sum")

                ips_loss = (xent_loss - imputation_loss)

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y,reduction="sum")

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    
                    print("[MF-TDR] epoch:{}, xent:{}".format(epoch, epoch_loss))

                    e_loss = F.binary_cross_entropy(pred, sub_y, reduction = "none")                    
                    e_hat_loss = F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction = "none")
                    
                    TMLE_beta = inv_prop-1
                    TMLE_alpha = e_loss - e_hat_loss
                    TMLE_epsilon = ((TMLE_alpha * TMLE_beta).sum()/(TMLE_beta * TMLE_beta).sum())
                    e_hat_TMLE = TMLE_epsilon.item() * (one_over_zl.float()- torch.tensor([1.])) 
                    e_hat_TMLE_obs = e_hat_TMLE[np.where(observation.cpu() == 1)]
                    
                    np.random.shuffle(all_idx)
                    np.random.shuffle(x_all)
                    
                    selected_idx = all_idx[0:self.batch_size]
                    sub_x = x[selected_idx]
                    sub_y = y[selected_idx]

                    # propensity score
                    inv_prop = one_over_zl_obs[selected_idx].detach()

                    sub_y = torch.Tensor(sub_y)

                    pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)  
                    pred = self.sigmoid(pred)

                    x_sampled = x_all[ul_idxs[0: G * self.batch_size]]

                    pred_ul,_,_ = self.prediction_model.forward(x_sampled, True)
                    pred_ul = self.sigmoid(pred_ul)

                    xent_loss = ((F.binary_cross_entropy(pred, sub_y, reduction="none") ** 2) * inv_prop).sum() # o*eui/pui
                        
                    imputation_loss = ((F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction="none") + e_hat_TMLE_obs[selected_idx].squeeze().detach()) ** 2).sum()
 
                    ips_loss = (xent_loss - imputation_loss)

                    sub_x_sampled_number = []
                    for i in x_sampled:
                        sub_x_sampled_number.append((self.num_items * i[0] + i[1]))
                    sub_x_sampled_number = np.array(sub_x_sampled_number)
                    
                    direct_loss = ((F.binary_cross_entropy(pred_ul, imputation_y, reduction="none") + e_hat_TMLE[sub_x_sampled_number].squeeze().detach()) ** 2).sum()                    
                    
                    loss = (ips_loss + direct_loss)/sub_x_sampled_number.shape[0]

                    optimizer_prediction.zero_grad()
                    loss.backward()
                    optimizer_prediction.step()
                    
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-TDR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-TDR] Reach preset epochs, it seems does not converge.")
                        
    def predict(self, x):
        pred = self.prediction_model.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

         

    
class MF_TDR_JL(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.original_model = MF(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x, y, gamma):
        self.original_model.fit(x = x, y = y, num_epoch=1000, lr=0.05, lamb=0, tol=le-4, verbose=False)
        x_train = torch.zeros([self.num_users,self.num_items])

        for i in range(len(x)):
            # x_train[x[i][0],x[i][1]] = 1 # o
            x_train[int(x[i][0]),int(x[i][1])] = 1
        prediction = (x_train.reshape(self.num_users*self.num_items,1)).type(torch.FloatTensor)

        x_train = self.original_model.complete().type(torch.FloatTensor).detach()
        
        optimizer = torch.optim.SGD([self.linear_1.weight, self.linear_1.bias], lr=1e-3, momentum=0.9)

        last_loss = 1e9 
        early_stop = 0
        for epoch in range(1000):
            all_idx = np.arange(self.num_users*self.num_items)
            np.random.shuffle(all_idx)
            total_batch = (self.num_users*self.num_items)// self.batch_size
            #print(early_stop)
            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x_train = x_train[selected_idx].detach()
                sub_prediction = prediction[selected_idx]
             
                out = self.linear_1(sub_x_train)
                out = self.sigmoid(out)
                loss = self.xent_func(out, sub_prediction)
                
                xent_loss = loss
                optimizer.zero_grad()
                xent_loss.backward()

                optimizer.step()

            if (epoch + 1) % 15 == 0:
                print('*'*10)
                print('epoch {}'.format(epoch+1))
            
            if loss > last_loss:
                early_stop += 1 
            else:
                last_loss = loss
            
            if early_stop >= 5:
                break
        
        x_train = x_train.detach()
        propensity = self.sigmoid(self.linear_1(x_train.detach())) 
        propensity[np.where(propensity.cpu() <= gamma)] = gamma
        one_over_zl = 1 / propensity
                      
        return prediction, one_over_zl  
        
    def fit(self, x, y, stop = 1,
        num_epoch=1000,lr=0.05, lamb=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
     
        last_loss = 1e9
   
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) 
        total_batch = num_sample // self.batch_size
        observation, one_over_zl = self._compute_IPS(x, y, gamma)

        early_stop = 0
        one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        
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

                # propensity score

                inv_prop = one_over_zl_obs[selected_idx].detach()                
                
                sub_y = torch.Tensor(sub_y)
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation.predict(x_sampled)
                pred_u = self.sigmoid(pred_u)     
                imputation_y1 = self.sigmoid(imputation_y1)          
                                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                        
                ips_loss = xent_loss - imputation_loss 
                                
                # direct loss                                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")             

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.imputation.forward(sub_x)
                imputation_y = self.sigmoid(imputation_y)
                    
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-TDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    
                    e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                    e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")   
                    
                    TMLE_beta = inv_prop-1
                    TMLE_alpha = e_loss - e_hat_loss
                    TMLE_epsilon = ((TMLE_alpha * TMLE_beta).sum()/(TMLE_beta * TMLE_beta).sum())
                    e_hat_TMLE = TMLE_epsilon.item() * (one_over_zl.float()- torch.tensor([1.]))
                    e_hat_TMLE_obs = e_hat_TMLE[np.where(observation.cpu() == 1)]

                    np.random.shuffle(x_all)
                    np.random.shuffle(all_idx)
                    
                    selected_idx = all_idx[0:self.batch_size]
                    sub_x = x[selected_idx] 
                    sub_y = y[selected_idx]

                    inv_prop = one_over_zl_obs[selected_idx].detach()                
                
                    sub_y = torch.Tensor(sub_y)
                       
                    pred = self.prediction_model.forward(sub_x)
                    imputation_y = self.imputation.predict(sub_x)
                    pred = self.sigmoid(pred)
                    imputation_y = self.sigmoid(imputation_y)
                               
                    x_sampled = x_all[ul_idxs[0 : G*self.batch_size]]
                                       
                    pred_u = self.prediction_model.forward(x_sampled) 
                    imputation_y1 = self.imputation.predict(x_sampled)
                    pred_u = self.sigmoid(pred_u)     
                    imputation_y1 = self.sigmoid(imputation_y1)                             
                
                    xent_loss = ((F.binary_cross_entropy(pred, sub_y, reduction ="none") ** 2) * inv_prop).sum()
                    imputation_loss = ((F.binary_cross_entropy(pred, imputation_y, reduction="none")
                                        + e_hat_TMLE_obs[selected_idx].squeeze().detach()) ** 2).sum()
                        
                    ips_loss = xent_loss - imputation_loss
                    
                    # direct loss
                    sub_x_sampled_number = []
                    for i in x_sampled:
                        sub_x_sampled_number.append((self.num_items * i[0] + i[1]))
                    sub_x_sampled_number = np.array(sub_x_sampled_number)                 
                
                    direct_loss = ((F.binary_cross_entropy(pred_u, imputation_y1, reduction="none") + e_hat_TMLE[sub_x_sampled_number].squeeze().detach()) ** 2).sum()
                    
                    loss = (ips_loss + direct_loss)/sub_x_sampled_number.shape[0]
                    
                    optimizer_prediction.zero_grad()
                    loss.backward()
                    optimizer_prediction.step()
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-TDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-TDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()        
