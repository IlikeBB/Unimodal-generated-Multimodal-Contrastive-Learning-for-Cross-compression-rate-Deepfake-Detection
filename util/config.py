# parameter setting
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    elif value.lower() in {'false', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()

    # ----------------- General ------------------
    parser.add_argument('--exper_name', default="only_test", type=str,
                        help="""
                        experiment name!!!!!
                        """)
    parser.add_argument('--train_dataset', default="", type=str,
                        help="""
                        Options => C: COHFACE, P: PURE, U: UBFC, M: MR-NIRP, V: VIPL-HR,
                        e.g. --dataset="C"  for intra-training/testing on COHFACE
                             --dataset="UP" for cross-training/testing on PURE and UBFC 
                        """)
    parser.add_argument('--test_dataset', default="", type=str,help="Same as above")
    parser.add_argument('--finetune_dataset', default="FF++", choices=['FF++'], type=str,help="FF++ only")
    # arser.add_argument('--conv', default="conv3d", type=str,help="Convolution type for 3DCNN")
    
    parser.add_argument('--warmup', default=True, type=str2bool)
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')  #default=0.0001
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning reate decay')
    
    
    
    # parser.add_argument('--log', type=str, default="Physformer_log", help='log and save model name')
    parser.add_argument('--bs', default=6, type=int,help="batch size")
    parser.add_argument('--epoch', default=100, type=int,help="training/testing epoch")
    parser.add_argument('--fps', default=30, type=int,help="fps for dataset")
    parser.add_argument('--model_S', default=2, type=int,help="spatial dimension of model")

    # ----------------- Training -----------------
    parser.add_argument('--train_T', default=10, type=int,help="training clip length(seconds))")
    parser.add_argument('--seq_len', default=160, type=int,help="fix")
    parser.add_argument('--frames_per_subject', default=200, type=int,
                        help="最多讀取多少 ExDDV 影格（<=0 代表使用全部 metadata）。")
    parser.add_argument('--loss', default="", type=str,help="")


    # ----------------- spacial data augmentation in training -----------------
    parser.add_argument('--inject_noise', action='store_true')
    

    # ----------------- Testing -----------------
    parser.add_argument('--test_T', default=10, type=int,help="training clip length(seconds))")
    parser.add_argument('--test_comp', default="c23", type=str,help="testing on c23 or c40 comp")
    
    # ----------------- Finetune -----------------
    parser.add_argument('--fix_weight',action='store_true')
    parser.add_argument('--continue_model',action='store_true')
    
    # ----------------- Validate-----------------
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--epoch_val', default=0, type=int,help="epoch validate")

    # ----------------- FF++ config ----------------- 
    parser.add_argument('--subset', default="all", type=str,)
    parser.add_argument('--cross_subset', default="", type=str,)
    parser.add_argument('--real_or_fake', default="both", type=str,)
    parser.add_argument('--comp_1_loader', default="c23", type=str,)
    parser.add_argument('--comp_2_loader', default="c40", type=str,)  
    parser.add_argument('--comp_3_loader', default="raw", type=str,)  
    # parser.add_argument('--test_comp', default="", type=str,)    

    # ----------------- Training Dataset Rate ----------------- 
    parser.add_argument('--train_rate', default='raw', choices=['raw','c23','c40'], type=str, help='')
    # ----------------- mask config ----------------- 
    parser.add_argument('--Mask_prob', default=0.0, type=float,)
    parser.add_argument('--Mask_percentage', default=0.0, type=float,)
    parser.add_argument('--max_mask_count', default=0, type=int,)
    parser.add_argument('--text_prompt_filter', default=False, type=str2bool)

    # ----------------- cross entropy weight rate ----------------- 
    parser.add_argument('--twin_dissim_sim_prompt_set', default=False, type=str2bool)
    parser.add_argument('--bce_alpha', default=0.7, type=float, help='similar classifier loss weight')
    parser.add_argument('--bce_beta', default=0.3, type=float, help='dissimilar classifier loss weight')
    # ----------------- text prompt weight rate ----------------- 
    parser.add_argument('--text_prompt_weight', default=0.3, type=float,)
    
    # ----------------- CCAL weight rate ----------------- 
    parser.add_argument('--CCAL_alpha', default=0.5, type=float,)
    parser.add_argument('--CCAL_beta', default=0.5, type=float,)
    # -----------------  affinity loss weight rate ----------------- 
    parser.add_argument('--aff_alpha', default=0.5, type=float,)
    parser.add_argument('--aff_beta', default=0.5, type=float,)
    
    # -----------------  model architecture config ----------------- 
    parser.add_argument('--classifier_module', default='mlp', choices=['mlp','trans_mlp'], type=str, help='')
    parser.add_argument('--classifier_module_heads', default=8, type=int, help='')
    parser.add_argument('--classifier_module_layers', default=2, type=int, help='')
    # -----------------  final loss weight ----------------- 
    parser.add_argument('--phy_loss_w', default=0.2, type=float,)
    parser.add_argument('--CCTL_w', default=0.25, type=float,)
    parser.add_argument('--aff_loss_w', default=0.3, type=float,)
    return parser.parse_args()


def get_name(args, finetune=False, model_name="", pretrain_model_name=""):
    

    if pretrain_model_name == "":
        pretrain_model_name = model_name
        
    trainName = f"{args.train_dataset}_train_{pretrain_model_name}"
    testName  = f"{args.train_dataset}_to_{args.test_dataset}_{model_name}"
    
    # nouse
    if args.inject_noise:
        testName += "_injectNoise"
        
    if finetune:
        
        finetune_dataset = args.finetune_dataset
        test_dataset = args.test_dataset
        
        if finetune_dataset == "FF++":
            finetune_dataset += f"_{args.subset}"
            
            if args.test_comp == "":
                args.test_comp = args.comp_1_loader
            test_dataset += f"_{args.subset}_{args.test_comp}"
                    
        finetuneName = f"{args.train_dataset}_finetune_{finetune_dataset}_{model_name}"
        testName = f"{args.train_dataset}_finetune_{finetune_dataset}_to_{test_dataset}_{model_name}"
        
        if args.fix_weight:
            finetuneName += "_fixWeight"
            
        return trainName, testName, finetuneName

    return trainName, testName


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    
    print(get_name(args, finetune=True, model_name="asdasdasd", pretrain_model_name="Physformer"))
