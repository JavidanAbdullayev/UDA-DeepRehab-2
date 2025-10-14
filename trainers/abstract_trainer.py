import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, dataset_class
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
# from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject,AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class

from typing import Tuple
from torch.utils.data import DataLoader
import json
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.experiment_description = args.dataset
        print(f'self.experiment_description: ', self.experiment_description)
        # self.run_description = f"{args.da_method} _{args.exp_name}"
        self.run_description = f"{args.da_method}" # _{args.exp_name}"
        
        
        # paths
        self.home_path =  os.getcwd() #os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        print(f'self.exp_log_dir: ', self.exp_log_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)




        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        dataset_path = os.path.join(self.data_path, "fold" + str(0))
        
        s_test = np.load(
            os.path.join(dataset_path, "s_test_fold" + str(0) + ".npy"),
            allow_pickle=True,
        )
        
        test_subject_ids = np.unique(s_test)
        print('test_subject_ids: ', test_subject_ids)
        self.test_sub_ids = test_subject_ids
        
        with open(os.path.join(self.data_path, 'info.json')) as file:
            data_info = json.load(file)
        print('Number of classes: ', data_info['nb_classes'])
        print('Number of channels: ', data_info['n_joints'] * data_info['n_dim'])
        
        self.num_classes = data_info['nb_classes']
        # metrics
        # self.num_classes = self.dataset_configs.num_classes
        self.dataset_configs.num_classes = self.num_classes
        self.dataset_configs.input_channels = data_info['n_joints'] * data_info['n_dim']
        
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)        

        # # metrics

    def sweep(self):
        # sweep configurations
        pass
    
    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # Training the model
        self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)
        return self.last_model, self.best_model
    
    def evaluate(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_configs(self):
        dataset_class = get_dataset_class('RehabPile')
        hparams_class = get_hparams_class('RehabPile')
        return dataset_class(), hparams_class()


    def _normalize_skeletons(self,
        X: np.ndarray,
        min_max_list: list = None,
    ):
        dim = int(X.shape[2])
        n_X = np.zeros(shape=X.shape)

        if min_max_list is None:
            min_max_list = []
            for d in range(dim):
                min_ = np.min(X[:, :, d, :])
                max_ = np.max(X[:, :, d, :])
                min_max_list.append((min_, max_))

        for d in range(dim):
            n_X[:, :, d, :] = (X[:, :, d, :] - min_max_list[d][0]) / (
                1.0 * (min_max_list[d][1] - min_max_list[d][0])
            )

        return n_X, min_max_list


    def stratified_subject_split(self, X, y, subjects, test_size=0.2, random_state=42):
        """
        Split X, y into train/test with stratification by subject IDs.
        Ensures each subject appears in both splits (assuming each subject has >= 2 samples
        and test_size not too small).
        """
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        # Use subjects as the stratification labels
        (train_idx, test_idx), = sss.split(X, subjects)
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        subj_tr, subj_te = subjects[train_idx], subjects[test_idx]
        
        # assert set(np.unique(subj_tr)) == set(np.unique(subj_te)), \
        #     "Not all subjects appear in both splits -- consider increasing test_size or checking \
        #         per-subject counts"
                
        return (X_tr, y_tr, subj_tr), (X_te, y_te, subj_te)

    def load_data(self, fold_number=0, test_subject_id=None):
        """Load the chosen classification dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the classification dataset chosen to load.
        root_path: str
            The directory containing all datasets.
        fold_number: int
            The fold number.

        Returns
        -------
        xtrain: np.ndarray
            The training sequences of shape (n_cases, n_channels, n_timepoints).
        ytrain: np.ndarray
            The labels of the training samples of shape (n_cases,).
        xtest: np.ndarray
            The testing sequences of shape (n_cases, n_channels, n_timepoints).
        ytest: np.ndarray
            The labels of the testing samples of shape (n_cases,).
        dataset_info: dict
            The dataset information in dictionary format.
        """
        dataset_path = os.path.join(self.data_path, "fold" + str(fold_number))

        with open(os.path.join(self.data_path, "info.json")) as f:
            content_info_datset = f.read()

        content_info_datset = re.sub(r",\s*([\]}])", r"\1", content_info_datset)
        content_info_datset = re.sub(r",\s*([\]}])", r"\1", content_info_datset)

        dataset_info = json.loads(content_info_datset)

        length_TS = dataset_info["length_TS"]
        n_joints = dataset_info["n_joints"]
        
        try:
            dim = dataset_info["dim"]
        except KeyError:
            dim = dataset_info["n_dim"]

        xtrain = np.load(
            os.path.join(dataset_path, "x_train_fold" + str(fold_number) + ".npy"),
            allow_pickle=True,
        )
        ytrain = np.load(
            os.path.join(dataset_path, "y_train_fold" + str(fold_number) + ".npy"),
            allow_pickle=True,
        )
        s_train = np.load(
            os.path.join(dataset_path, "s_train_fold" + str(fold_number) + ".npy"),
            allow_pickle=True,
        )
        
        xtest = np.load(
            os.path.join(dataset_path, "x_test_fold" + str(fold_number) + ".npy"),
            allow_pickle=True,
        )
        ytest = np.load(
            os.path.join(dataset_path, "y_test_fold" + str(fold_number) + ".npy"),
            allow_pickle=True,
        )
        s_test = np.load(
            os.path.join(dataset_path, "s_test_fold" + str(fold_number) + ".npy"),
            allow_pickle=True,
        )
        
    
        xtrain = np.array(xtrain.tolist())
        ytrain = np.array(ytrain.tolist())
        xtest = np.array(xtest.tolist())
        ytest = np.array(ytest.tolist())

        xtrain = np.reshape(xtrain, (len(xtrain), n_joints, dim, length_TS))
        xtest = np.reshape(xtest, (len(xtest), n_joints, dim, length_TS))

        xtrain, min_max_list = self._normalize_skeletons(X=xtrain, min_max_list=None)

        xtest, _ = self._normalize_skeletons(X=xtest, min_max_list=min_max_list)

        xtrain = np.reshape(xtrain, (len(xtrain), n_joints * dim, length_TS))
        xtest = np.reshape(xtest, (len(xtest), n_joints * dim, length_TS))

        le = LabelEncoder()
        ytrain = le.fit_transform(ytrain)
        ytest = le.transform(ytest)

        f.close()
        
        print(f"xtest shape: {xtest.shape}")
        print(f"ytest shape: {ytest.shape}")
        print(f"s_test shape: {s_test.shape}")                

        train_ds = dataset_class(xtrain, np.reshape(ytrain, (len(ytrain),)))        
        test_ds = dataset_class(xtest, np.reshape(ytest, (len(ytest),)))
        
        
        self.train_dl = DataLoader(train_ds, batch_size=self.hparams['batch_size'],  shuffle=True,  drop_last=True,  pin_memory=True)
        self.test_dl =  DataLoader(test_ds,  batch_size=self.hparams['batch_size'],  shuffle=True,  drop_last=True,  pin_memory=True)

        self.input_channels = xtrain.shape[1]
        self.num_classes = np.unique(ytrain)
        
        print('self.input_channels: ', self.input_channels)
        print('self.num_classes: ', self.num_classes)

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        # self.evaluate(self.few_shot_dl_5)
        # fst_risk = self.loss.item()
        fst_risk = 0
        # calculation based target test data
        self.evaluate(self.test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1_sk learn
        # f1 = f1_score(self.full_preds.argmax(dim=1).cpu().numpy(), self.full_labels.cpu().numpy(), average='macro')

        risks = src_risk, fst_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self,table_results, fold_id, run_id, name):
        # save to file if needed
        print(self.exp_log_dir, f"fold_{fold_id}", f"run_{run_id}", f"{name}.csv")

        table_results.to_csv(os.path.join(self.exp_log_dir, f"fold_{fold_id}", f"run_{run_id}", f"{name}.csv"))


    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):
       
        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}


        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self):
       
        self.evaluate(self.test_dl)
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # balanced accuracy
        metric = get_scorer('balanced_accuracy')._score_func
        bal_acc = metric(y_true=self.full_labels.cpu(), y_pred=self.full_preds.argmax(dim=1).cpu())
        
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        
        return acc, bal_acc, f1, auroc

    def calculate_risks(self):
         # calculation based source test data
        self.evaluate(self.test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        # self.evaluate(self.few_shot_dl_5)
        # fst_risk = self.loss.item()
        fst_risk = 0
        # calculation based target test data
        self.evaluate(self.test_dl)
        trg_risk = self.loss.item()

        return src_risk, fst_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [run_id, scenario, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        print('table; ', table)
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table 