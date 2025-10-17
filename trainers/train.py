import sys
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()       
from collections import OrderedDict


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super().__init__(args)

        self.results_columns = ["run", "test_subject_id",  "acc",       "bal_acc",        "f1_score",   "auroc"]
        self.risks_columns =   ["run", "test_subject_id",  "src_risk",  "few_shot_risk",  "trg_risk"]
   
            
    def fit(self):

        # table with metrics
        table_results = pd.DataFrame(columns=self.results_columns)

        # table with risks
        table_risks = pd.DataFrame(columns=self.risks_columns)


        # Trainer
        for fold_id in range(0, 1):
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, run_id, fold_id)
                
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                print(f'self.loss_avg_meters: {self.loss_avg_meters}')

                # Load data
                self.load_data(fold_number=fold_id, test_subject_id=None)
                
                # initiate the domain adaptation algorithm
                self.initialize_algorithm()                

                # Load checkpint
                # pretrained_model_log_dir = os.path.join('experiments_logs_NO_ADAPT/', self.dataset, "NO_ADAPT", "fold_" + str(fold_id), "run_" + str(run_id))
                # last_chk, best_chk = self.load_checkpoint(pretrained_model_log_dir)
                # self.algorithm.network.load_state_dict(best_chk)                
                
                # # print('best_chk: ', best_chk.keys())
                # bb_std = OrderedDict()
                # for k, v in best_chk.items():
                #     if k.startswith("0."):
                #         bb_std[k[len("0."):]] = v  # drop the "0." prefix

                # self.algorithm.network[0].load_state_dict(bb_std)
                
                print('\n\n Trainig will start now!!! \n\n')
                # Train the domain adaptation algorithm
                self.first_model, self.last_model, self.best_model, src_train_losses, trg_val_losses, src_train_accs, trg_val_accs = \
                                                                    self.algorithm.update(self.train_dl, self.test_dl, self.loss_avg_meters, self.logger)

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, self.first_model, self.last_model, self.best_model)

                # !!!!!!!!!!!!!!!!!!!!!! Load the best model for the evaluation !!!!!!!!!!!!!!!!!!!!!!                                           
                self.algorithm.network.load_state_dict(self.best_model)
                
                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{self.test_sub_ids}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

                # Calculate and append mean and std to tables
                table_results = self.add_mean_std_table(table_results, self.results_columns)
                table_risks = self.add_mean_std_table(table_risks, self.risks_columns)

                # Save tables to file if needed
                self.save_tables_to_file(table_results, fold_id, run_id, 'results')
                self.save_tables_to_file(table_risks, fold_id, run_id, 'risks')

                table_results = pd.DataFrame(columns=self.results_columns)
                table_risks = pd.DataFrame(columns=self.risks_columns)


                # SAVE LOGS
                
                epochs = range(1, len(src_train_losses) + 1)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(9, 4))
                plt.plot(epochs, src_train_losses, label="source train loss")
                plt.plot(epochs, trg_val_losses, label="target test loss")                
                plt.xlabel("Epoch")
                plt.ylabel("Value")  # change to "Loss" or "Accuracy" as appropriate
                plt.title("Losses over epochs")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.savefig(os.path.join(self.exp_log_dir, f"fold_{fold_id}", f"run_{run_id}", "losses.pdf"))



                epochs = range(1, len(src_train_losses) + 1)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(9, 4))
                plt.plot(epochs, src_train_accs, label="source train accuracy")
                plt.plot(epochs, trg_val_accs, label="target test accuracy")                
                plt.xlabel("Epoch")
                plt.ylabel("Value")  # change to "Loss" or "Accuracy" as appropriate
                plt.title("Accuracies over epochs")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.savefig(os.path.join(self.exp_log_dir, f"fold_{fold_id}", f"run_{run_id}", "accuracies.pdf"))


    def test(self):
        # Results dataframes
        last_results = pd.DataFrame(columns=self.results_columns)
        best_results = pd.DataFrame(columns=self.results_columns)

        # Cross-domain scenarios
        for run_id in range(self.num_runs):
            # for subject_id in self.test_sub_ids:
            # Results dataframes
            last_results = pd.DataFrame(columns=self.results_columns)
            best_results = pd.DataFrame(columns=self.results_columns)

            # fixing random seed
            fix_randomness(run_id)

            # Logging
            # self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))
            self.scenario_log_dir = os.path.join(self.exp_log_dir, "run_" + str(run_id))
            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            # Load data
            self.load_data(fold_number=0, test_subject_id=None)

            # Build model
            self.initialize_algorithm()

            # Load chechpoint 
            last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)

            # Testing the last model
            self.algorithm.network.load_state_dict(last_chk)
            self.evaluate(self.trg_test_dl)
            last_metrics = self.calculate_metrics()
            last_results = self.append_results_to_tables(last_results, "run_" + str(run_id), run_id, last_metrics)
            

            # Testing the best model
            self.algorithm.network.load_state_dict(best_chk)
            self.evaluate(self.trg_test_dl)
            best_metrics = self.calculate_metrics()
            # Append results to tables
            best_results = self.append_results_to_tables(best_results, "run_" + str(run_id), run_id, 
                                                            best_metrics)

            print(f'last_results: \n{last_results}')
            print(f'best_results: \n{best_results}')
            
            
            
            # last_scenario_mean_std = last_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])
            # best_scenario_mean_std = best_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])


            # Save tables to file if needed
            self.save_tables_to_file(last_results, run_id, 'last_results')
            self.save_tables_to_file(best_results, run_id, 'best_results')

            # printing summary 
            summary_last = {metric: np.mean(last_results[metric]) for metric in self.results_columns[2:]}
            summary_best = {metric: np.mean(best_results[metric]) for metric in self.results_columns[2:]}
            for summary_name, summary in [('Last', summary_last), ('Best', summary_best)]:
                for key, val in summary.items():
                    print(f'{summary_name}: {key}\t: {val:2.4f}')


