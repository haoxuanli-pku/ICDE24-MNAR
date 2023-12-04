# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
import pdb
import arguments

from dataset import load_data
from matrix_factorization_DT import *
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, precision_func, recall_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))



def train_and_eval(dataset_name, train_args, model_args):
    
    top_k_list = [5]
    top_k_names = ("precision_5", "recall_5", "ndcg_5", "f1_5")
    if dataset_name == "coat":
        train_mat, test_mat = load_data("coat")        
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]

    elif dataset_name == "yahoo":
        x_train, y_train, x_test, y_test = load_data("yahoo")
        x_train, y_train = shuffle(x_train, y_train)
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1

    elif dataset_name == "kuai":
        x_train, y_train, x_test, y_test = load_data("kuai")
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1
        top_k_list = [50]
        top_k_names = ("precision_50", "recall_50", "ndcg_50", "f1_50")

    np.random.seed(2020)
    torch.manual_seed(2020)

    print("# user: {}, # item: {}".format(num_user, num_item))
    # binarize
    if dataset_name == "kuai":
        y_train = binarize(y_train, 1)
        y_test = binarize(y_test, 1)
    else:
        y_train = binarize(y_train)
        y_test = binarize(y_test)

    "ESCM2-DR"
    mf = MF_ESCM2_DR(num_user, num_item, embedding_k=model_args['embedding_k'],batch_size=train_args['batch_size'])
    mf.cuda()
    mf.fit(x_train, y_train, 
        lr=model_args['lr'], theta = train_args['theta'],
        lamb_prop=model_args['lamb_prop'],
        lamb_pred=model_args['lamb_pred'],
        lamb_imp=model_args['lamb_imp'],
        gamma = train_args['gamma'])

    test_pred = mf.predict(x_test)
    mse_mf = mse_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    mae_mf = mae_func(y_test, test_pred)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)
    f1 = 2 / (1 / np.mean(precisions[top_k_names[0]]) + 1 / np.mean(recalls[top_k_names[1]]))

    print("***"*5 + "[ESCM2-DR]" + "***"*5)
    print("[ESCM2-DR] test mse:", mse_mf)
    print("[ESCM2-DR] test mse:", mae_mf)
    print("[ESCM2-DR] test auc:", auc)
    print("[ESCM2-DR] {}:{:.6f}".format(
            top_k_names[2].replace("_", "@"), np.mean(ndcgs[top_k_names[2]])))
    print("[ESCM2-DR] {}:{:.6f}".format(top_k_names[3].replace("_", "@"), f1))
    print("[ESCM2-DR] {}:{:.6f}".format(
            top_k_names[0].replace("_", "@"), np.mean(precisions[top_k_names[0]])))
    print("[ESCM2-DR] {}:{:.6f}".format(
            top_k_names[1].replace("_", "@"), np.mean(recalls[top_k_names[1]])))
    user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + "[ESCM2-DR]" + "***"*5)

def para(args):
    if args.dataset=="coat":
        args.train_args = {"batch_size":128, "gamma": 0.01, "theta": 1}
        args.model_args = {"embedding_k":8, "lr":0.01, "lamb_prop": 1e-3,"lamb_pred": 1e-3,"lamb_imp": 1e-3}
    elif args.dataset=="yahoo":
        args.train_args = {"batch_size":4096, "gamma": 0.05, "theta": 1e-4}
        args.model_args = {"embedding_k":32, "lr":0.01, "lamb_prop": 1e-2,"lamb_pred": 1e-5,"lamb_imp": 1e-5}
    elif args.dataset=="kuai":
        args.train_args = {"batch_size":4096, "gamma": 0.05, "theta": 1e-4}
        args.model_args = {"embedding_k":32, "lr":0.01, "lamb_prop": 1e-2,"lamb_pred": 1e-4,"lamb_imp": 1e-5}
    return args

if __name__ == "__main__":
    args = arguments.parse_args()
    para(args=args)

    train_and_eval(args.dataset, args.train_args, args.model_args)