import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
import pdb

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))



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

    
class MF_Stable_DR(nn.Module):
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

    def fit(self, x, y, y_ips, mu = 0, eta = 1, stop = 5,
        num_epoch=1000, batch_size=128, lr=0.05, lr1 = 10, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        mu = torch.Tensor([mu])
        mu.requires_grad_(True)
        
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        optimizer_propensity = torch.optim.Adam(
            [mu], lr=lr1, weight_decay=lamb)
        
        last_loss = 1e9

        observation = torch.zeros([self.num_users, self.num_items])
        for i in range(len(x)):
            # observation[x[i][0], x[i][1]] = 1
            observation[int(x[i][0]),int(x[i][1])] = 1
        observation = observation.reshape(self.num_users * self.num_items)
        
        y1 = []
        for i in range(len(x)):
            if y[i] == 1:
                y1.append(self.num_items * x[i][0] + x[i][1])
        
        
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)
        
        
        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y, y1, mu)
        else:
            one_over_zl = self._compute_IPS(x, y, y1, mu, y_ips)
        
        one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)].detach()
        
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
            # propensity score
                inv_prop = one_over_zl_obs[selected_idx]                

                sub_y = torch.Tensor(sub_y)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop.detach()).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()  
                
                
                x_all_idx = ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]
                x_sampled = x_all[x_all_idx]                  
                    
                imputation_y1 = self.imputation.predict(x_sampled)  
                imputation_y1 = self.sigmoid(imputation_y1)
                
                prop_loss = F.binary_cross_entropy(1/one_over_zl[x_all_idx], observation[x_all_idx], reduction="sum")                
                pred_y1 = self.prediction_model.predict(x_sampled)
                pred_y1 = self.sigmoid(pred_y1)

                imputation_loss = F.binary_cross_entropy(imputation_y1, pred_y1, reduction = "none")
                
                loss = prop_loss + eta * ((1 - observation[x_all_idx] * one_over_zl[x_all_idx]) * (imputation_loss - imputation_loss.mean())).sum() ** 2      
                
                optimizer_propensity.zero_grad()
                loss.backward()
                optimizer_propensity.step()
                
                #print("mu = {}".format(mu))
                
                one_over_zl = self._compute_IPS(x, y, y1, mu, y_ips)        
                one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]
                inv_prop = one_over_zl_obs[selected_idx].detach()                                                
                
                pred = self.prediction_model.forward(sub_x)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight = inv_prop.detach(), reduction="sum")
                xent_loss = (xent_loss)/(inv_prop.detach().sum())
                
                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()      
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Stable-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self, x, y, y1, mu, y_ips=None):
        if y_ips is None:
            y_ips = 1
            print("y_ips is none")
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = (len(x) + mu)/ (x[:,0].max() * x[:,1].max() + 2*mu)
            py1o1 = (y.sum() + mu)/ (len(y) +2*mu)
            py0o1 = 1 - py1o1

            propensity = torch.zeros(self.num_users * self.num_items)
            
            propensity += (py0o1 * po1) / py0

            propensity[np.array(y1)] = (py1o1 * po1) / py1
            
            one_over_zl = (1 / propensity)
            
        #one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl           
    
