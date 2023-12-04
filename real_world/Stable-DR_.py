# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
import pdb
import arguments

from dataset import load_data
from matrix_factorization_DT import *
from SDR import *
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

    "Stable-DR"
    mf = MF_Stable_DR(num_user, num_item, embedding_k=model_args['embedding_k'])
    ips_idxs = np.arange(len(y_test))
    np.random.shuffle(ips_idxs)
    y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
    mf.fit(x_train, y_train, y_ips=y_ips,
        stop = train_args['stop'],
        eta = train_args['eta'],
        lr=model_args['lr'], 
        G=train_args["G"],
        batch_size = train_args['batch_size'],
        lr1 = model_args['lr1'],
        lamb = model_args['lamb'])

    test_pred = mf.predict(x_test)
    mse_mf = mse_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    mae_mf = mae_func(y_test, test_pred)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)
    f1 = 2 / (1 / np.mean(precisions[top_k_names[0]]) + 1 / np.mean(recalls[top_k_names[1]]))

    print("***"*5 + "[TDR-JL]" + "***"*5)
    print("[Stable-DR] test mse:", mse_mf)
    print("[Stable-DR] test mse:", mae_mf)
    print("[Stable-DR] test auc:", auc)
    print("[Stable-DR] {}:{:.6f}".format(
            top_k_names[2].replace("_", "@"), np.mean(ndcgs[top_k_names[2]])))
    print("[Stable-DR] {}:{:.6f}".format(top_k_names[3].replace("_", "@"), f1))
    print("[Stable-DR] {}:{:.6f}".format(
            top_k_names[0].replace("_", "@"), np.mean(precisions[top_k_names[0]])))
    print("[Stable-DR] {}:{:.6f}".format(
            top_k_names[1].replace("_", "@"), np.mean(recalls[top_k_names[1]])))
    user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + "[Stable-DR]" + "***"*5)

def para(args):
    if args.dataset=="coat":
        args.train_args = {"stop":5, 'eta':100,'G':3, "batch_size":128}
        args.model_args = {"lr":0.01, "lr1": 0.01, "lamb": 5e-5, "embedding_k":4}
    elif args.dataset=="yahoo":
        args.train_args = {"stop":1, 'eta':1000,'G':5, "batch_size":8192}
        args.model_args = {"lr":0.05, "lr1": 150, "lamb": 5e-6, "embedding_k":8}
    elif args.dataset=="kuai":
        args.train_args = {"stop":1, 'eta':1000,'G':5, "batch_size":8192}
        args.model_args = {"lr":0.05, "lr1": 150, "lamb": 1e-5, "embedding_k":4}
    return args

if __name__ == "__main__":
    args = arguments.parse_args()
    para(args=args)

    train_and_eval(args.dataset, args.train_args, args.model_args)