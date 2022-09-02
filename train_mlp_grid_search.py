
import argparse
import random
from datetime import datetime

import pandas as pd

from utils.train_utils import *

from train import create_model, train_evaluate_and_save_model
from sklearn.model_selection import ParameterGrid
import utils.train_parser as train_parser


def main():
    args = train_parser.get_parser().parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    name_dir = args.name + "__" + datestring
    root_log_dir = os.path.join(args.output_dir, name_dir)
    os.makedirs(root_log_dir)



    msn_train, msn_validation, msn_test, original_model, scaler, imputer, n_features = load_dataset_and_orginal_model(args)

    params = {}
    params['lr'] = [0.005, 0.001]
    params['gamma'] = [0.1, 0.5]
    params['weight-decay'] = [0, 1e-6, 1e-5]
    params['dropout'] = [0, 0.1, 0.2]
    columns = list(params.keys())
    columns.append('ndcg@10')
    df_log = pd.DataFrame(columns=columns)

    for param_conf in ParameterGrid(params):
        print(param_conf)
        log_dir_name = ""
        for key in params.keys():
            setattr(args, key, param_conf[key])
            log_dir_name+="_"+key+"_"+str(param_conf[key])
        log_dir = os.path.join(root_log_dir, log_dir_name)
        os.mkdir(log_dir)

        train_loader, validation_loader, scaler, imputer = create_data_loaders(args, original_model, scaler, msn_train, msn_validation,
                                                              imputer)
        model = create_model(args, n_features)

        ndcg_test = train_evaluate_and_save_model(model, train_loader, validation_loader, msn_validation, msn_test, scaler, log_dir=log_dir, imputer=imputer, args=args, state=state)
        df_line = list(param_conf.values())
        df_line.append(ndcg_test)
        current_df = pd.DataFrame([df_line], columns=columns)
        df_log = df_log.append(current_df)

    csv_path = os.path.join(root_log_dir, "overall_log.csv")
    print("Log file saved to " + csv_path)
    df_log.to_csv(csv_path)




if __name__== "__main__":
    main()
