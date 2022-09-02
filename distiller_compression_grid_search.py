import random
from datetime import datetime

import pandas as pd

from utils.train_utils import *

from train import create_model, train_evaluate_and_save_model
from sklearn.model_selection import ParameterGrid
import utils.train_parser as train_parser
import distiller

import utils.compress_parser as parser
from distiller_compression import train_compress_evaluate_and_save_model

import distiller
import distiller.apputils as apputils
from distiller.data_loggers import *
import torch.backends.cudnn as cudnn

def main():
    args = parser.get_parser().parse_args()
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))





    if args.deterministic:
        distiller.set_deterministic(args.seed)  # For experiment reproducability
    else:
        if args.seed is not None:
            distiller.set_seed(args.seed)
        cudnn.benchmark = True
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    name_dir = args.name + "__" + datestring
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    root_log_dir = os.path.join(args.output_dir,name_dir )
    os.makedirs(root_log_dir)

    start_epoch = 0
    ending_epoch = args.epochs

    args.device = 'cuda'
    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
        available_gpus = torch.cuda.device_count()
        for dev_id in args.gpus:
            if dev_id >= available_gpus:
                raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                 .format(dev_id, available_gpus))
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(args.gpus[0])

    #LOAD DATASETS

    msn_train, msn_validation, msn_test, original_model, scaler, imputer, n_features = load_dataset_and_orginal_model(
        args)

    params = {}
    params['lr'] = [0.001, 0.0005]
    params['batch_size'] = [1000]
    params['gamma'] = [0.1, 0.5]
    params['weight-decay'] = [0, 1e-6, 1e-5]
    columns = list(params.keys())
    columns.append('ndcg@10')
    df_log = pd.DataFrame(columns=columns)

    for param_conf in ParameterGrid(params):
        print(param_conf)
        log_dir_name = ""
        for key in params.keys():
            setattr(args, key, param_conf[key])
            log_dir_name += "_" + key + "_" + str(param_conf[key])

        #global msglogger

        msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), log_dir_name, root_log_dir)

        msglogger.debug("Distiller: %s", distiller.__version__)

        train_loader, validation_loader, scaler, imputer = create_data_loaders(args, original_model, scaler, msn_train,
                                                                               msn_validation,
                                                                               imputer)
        model = create_model(args, n_features)

        ndcg_test = train_compress_evaluate_and_save_model(model, train_loader, validation_loader, msn_validation, msn_test,
                                                           scaler, imputer, args, start_epoch, ending_epoch, msglogger)

        df_line = list(param_conf.values())
        df_line.append(ndcg_test)
        current_df = pd.DataFrame([df_line], columns=columns)
        df_log = df_log.append(current_df)

    csv_path = os.path.join(root_log_dir, "overall_log.csv")
    print("Log file saved to " + csv_path)
    df_log.to_csv(csv_path)




if __name__== "__main__":
    main()

