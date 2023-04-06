#!/usr/bin/python2.7
from clearml import Task, Logger
import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
from os import listdir
from os.path import isfile, join
import numpy as np


def ClearMl_integration(mode,expname):
    if mode == "baseline":
        task = Task.init(project_name='Testing', task_name=expname + " Baseline")
    elif mode == "train":
        task = Task.init(project_name='Testing', task_name=expname + " Chosen model")
    else:
        task = Task.init(project_name='Testing', task_name=expname)
    return task.get_logger()

def set_seeds():
    seed = 3162366
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def fold_split(features_path, val_path, test_path):
    with open(val_path, 'r') as f:
        val_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
    with open(test_path, 'r') as f:
        test_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
    train_files = [f for f in listdir(features_path) if isfile(join(features_path, f))]
    train_files = list(set(train_files) - set(test_files + val_files))
    return train_files, val_files, test_files

def manage_training(val_path_fold, test_path_fold, features_path_fold, device, results_dir, model_dir, actions_dict,
              num_layers_PG,sc, num_layers_R, num_R,
              num_f_maps, num_epochs, bz, lr, features_dim, clogger, kl_flag=False, sample_size=5):
    fold_num = features_path_fold.split("/")[-2]
    print(f"\t{fold_num}")
    sample_rate = 1
    num_classes = len(actions_dict)
    gt_path = '/datashare/APAS/transcriptions_gestures/'
    sample_size_flag = f"Sample size {sample_size}"
    exts = [sample_size_flag, "EfficienetB0"] #TODO: add name of the embedding
    folder_name = results_dir.format(fold_num, "_".join(exts))
    model_folder_name = model_dir.format(fold_num, "_".join(exts))
    try:
        os.makedirs(folder_name)
        os.makedirs(model_folder_name)
    except:
        pass
    vid_list_file, vid_list_file_val, vid_list_file_test = fold_split(features_path_fold, val_path_fold,
                                                                      test_path_fold)
    batch_gen_train = BatchGenerator(num_classes, actions_dict, gt_path, features_path_fold,
                                     sample_rate)

    batch_gen_train.read_data(vid_list_file)
    batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, features_path_fold, sample_rate)

    batch_gen_val.read_data(vid_list_file_val)
    trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes,
                      fold_num, fold_num, sc=sc, kl=kl_flag, sample_size=sample_size)
    trainer.train(model_folder_name, batch_gen_train, batch_gen_val, num_epochs=num_epochs, batch_size=bz,
                  learning_rate=lr, device=device, clogger=clogger)

    trainer.predict(model_folder_name, folder_name,
                    features_path_fold, vid_list_file_test, num_epochs, actions_dict, device,
                    sample_rate)



def args_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train')
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')

    parser.add_argument('--features_dim', default='1280', type=int)
    parser.add_argument('--bz', default='1', type=int)
    parser.add_argument('--lr', default='0.0005', type=float)

    parser.add_argument('--num_f_maps', default='65', type=int)

    # Need input
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_layers_PG', default=10, type=int)
    parser.add_argument('--num_layers_R', default=10, type=int)
    parser.add_argument('--num_R', default=3, type=int)
    parser.add_argument('--sc', default=1, type=float)
    return parser.parse_args()


def actions_handler(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = args_handler()
    num_epochs = 15
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr
    sc= args.sc
    num_layers_PG = args.num_layers_PG
    num_layers_R = args.num_layers_R
    num_R = args.num_R
    num_f_maps = args.num_f_maps
    fvec_name= "Efficient0"
    exp_name = f"try Fvecs: {fvec_name}"
    clogger = ClearMl_integration(args.action,exp_name)
    fold_srcs=[]
    mapping_file = "/datashare/APAS/mapping_gestures.txt"
    for i in range(5):
        fold_srcs.append([f"/datashare/APAS/folds/valid {i}.txt",
                    f"/datashare/APAS/folds/test {i}.txt",
                    f"/datashare/APAS/features/fold{i}/"])
   
    model_dir = "./models/" + str(exp_name) + "/{}/{}"
    results_dir = "./results/" + str(exp_name) + "/{}/{}"

    actions_dict= actions_handler(mapping_file=mapping_file)
    

    print("Task training")
    
    if args.action == "train_tradeoff":
        sample_sizes=[1, 5, 10, 30, 60]
    if args.action == "train":
        sample_sizes=[5]
    if args.action == "baseline":
        sample_sizes=[1]
    for val_path_fold, test_path_fold, features_path_fold in fold_srcs:
        for sample_size in sample_sizes :
            manage_training(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                        features_path_fold=features_path_fold, device=device, results_dir=results_dir,
                        model_dir=model_dir,
                        actions_dict=actions_dict,
                        num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                        num_f_maps=num_f_maps, num_epochs=num_epochs,sc=sc, bz=bz, lr=lr, features_dim=features_dim,
                        clogger=clogger, kl_flag=False,
                        sample_size=sample_size)

# /datashare/APAS/transcriptions_gestures/P028_balloon1.txt'
