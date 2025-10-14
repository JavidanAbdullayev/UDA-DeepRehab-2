from trainers.train import Trainer
import argparse
import copy




parser = argparse.ArgumentParser()


if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='train',         type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',               default='EXP1',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='CoDATS',               type=str, help='NO_ADAPT, Deep_Coral, MMDA, DANN, CDAN, DIRT, DSAN, HoMM, CoDATS, AdvSKM, SASA, CoTMix, TARGET_ONLY')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'/home/jabdullayev/phd/datasets/RehabPile/classification/',                  type=str, help='Path containing datase2t')
    # parser.add_argument('--dataset',                default='UIPRMD_clf_bn_DS',  type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')


    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='LITEMV',                      type=str, help='Backbone of choice: (LITEMV - CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=5,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or gpu')


   # arguments
    args = parser.parse_args()

    # build the list of datasets to iterate
    dataset_list = [

                    # 'KERAAL_clf_mc_CTK', 'KERAAL_clf_mc_ELK', 'KERAAL_clf_mc_RTK', 
                    # 'KERAAL_clf_bn_CTK', 'KERAAL_clf_bn_ELK', 'KERAAL_clf_bn_RTK',                     
                   
                   # -------------------------------------------------------------------------
                    # 'KIMORE_clf_bn_LA', 
                    # 'KIMORE_clf_bn_LT', 'KIMORE_clf_bn_PR', 'KIMORE_clf_bn_Sq', 'KIMORE_clf_bn_TR', 
                    # 'KINECAL_clf_bn_3WFV', 'KINECAL_clf_bn_GGFV', 'KINECAL_clf_bn_QSEC', 'KINECAL_clf_bn_QSEO', 'SPHERE_clf_bn_WUS',                     
                   # -------------------------------------------------------------------------
                   
                    # 'UIPRMD_clf_bn_DS', 'UIPRMD_clf_bn_HS', 'UIPRMD_clf_bn_IL', 'UIPRMD_clf_bn_SASLR', 'UIPRMD_clf_bn_SL',
                    # 'UIPRMD_clf_bn_SSA', 'UIPRMD_clf_bn_SSE', 'UIPRMD_clf_bn_SSIER', 'UIPRMD_clf_bn_SSS', 'UIPRMD_clf_bn_STS',                    
                    
                    # 'IRDS_clf_bn_EFL', 
                    # 'IRDS_clf_bn_EFR', 
                    # 'IRDS_clf_bn_SAL', 
                    # 'IRDS_clf_bn_SAR',                     
                    # 'IRDS_clf_bn_SFE', 
                    # 'IRDS_clf_bn_SFL', 
                    # 'IRDS_clf_bn_SFR', 
                    # 'IRDS_clf_bn_STL', 
                    # 'IRDS_clf_bn_STR', 
                   
                   
                   
                    # 'UCDHE_clf_bn_MP', 'UCDHE_clf_bn_Rowing', 'UCDHE_clf_mc_MP', 'UCDHE_clf_mc_Rowing'  
                    
                    
                    'KERAAL_clf_mc_ELK',
                    ]



    # run each dataset sequentially
    for ds in dataset_list:
        # copy args so we don't mutate the original Namespace across runs
        run_args = copy.deepcopy(args)
        run_args.dataset = ds
        # (optional) make exp_name unique per dataset
        run_args.exp_name = f"{args.exp_name}_{ds}"

        trainer = Trainer(run_args)

        if run_args.phase == 'train':
            trainer.fit()
            # print('Test phase started')
            # trainer.test()
            
        # elif run_args.phase == 'test':
            # trainer.test()
            

    # # arguments
    # args = parser.parse_args()

    # # create trainier object
    # trainer = Trainer(args)

    # # train and test
    # if args.phase == 'train':
    #     trainer.fit()
    # elif args.phase == 'test':
    #     trainer.test()



#TODO:
# 1- Change the naming of the functions ---> ( Done)
# 2- Change the algorithms following DCORAL --> (Done)
# 3- Keep one trainer for both train and test -->(Done)
# 4- Create the new joint loader that consider the all possible batches --> Done
# 5- Implement Lower/Upper Bound Approach --> Done
# 6- Add the best hparams --> Done
# 7- Add pretrain based methods (ADDA, MCD, MDD)
