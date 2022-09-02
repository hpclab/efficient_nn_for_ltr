
#todo Feature eliminate rispetto a Distiller: reset_optimizer, automl, greedy, sensitivity, knowledge distillation, sensitivity_anlysis, activation_stats_collector, num_best_scores


import math
import traceback
import logging
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import distiller
import distiller.apputils as apputils
from distiller.data_loggers import *
#import parser
import argparse
import os
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.train_utils import *
from utils import AverageMeter, Bar
import utils.compress_parser as parser



# Logger handle
msglogger = None


from train import create_model, valid, compute_metrics, adjust_learning_rate


def main():

    args = parser.get_parser().parse_args()
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    #global msglogger



    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    msglogger.debug("Distiller: %s", distiller.__version__)

    if args.deterministic:
        distiller.set_deterministic(args.seed) # For experiment reproducability
    else:
        if args.seed is not None:
            distiller.set_seed(args.seed)
        cudnn.benchmark = True

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

    msn_train, msn_validation, msn_test, original_model, scaler, imputer, n_features = load_dataset_and_orginal_model(
        args)

    train_loader, val_loader, scaler, imputer = create_data_loaders(args, original_model, scaler, msn_train,
                                                                    msn_validation,
                                                                    imputer)
    model = create_model(args, n_features)

    ndcg_test = train_compress_evaluate_and_save_model(model, train_loader, val_loader, msn_validation, msn_test,
                                              scaler, imputer, args, start_epoch, ending_epoch, msglogger )



def train_compress_evaluate_and_save_model(model, train_loader, validation_loader, msn_validation, msn_test,
                                              scaler, imputer, args, start_epoch, ending_epoch, msglogger ):

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.gamma, patience=10)




    if args.compress:
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler,
            (start_epoch-1) )
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    best_model = None
    best_metric = 0
    best_epoch = -1

    df_log = pd.DataFrame(columns=["train_loss", "val_loss", "ndcg@10", "map_1", "map_0"])

    #early_stopping = EarlyStopping(patience= 10)


    for epoch in range(start_epoch, ending_epoch):

        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        if compression_scheduler:

            compression_scheduler.on_epoch_begin(epoch,
                metrics=(val_loss if (epoch != start_epoch) else 10**6))

            if epoch==0:

                print("PERFORMANCE WITHOUT RE-TRAINING ")
                compression_scheduler.mask_all_weights()
                val_loss = valid(validation_loader, model, criterion, args)
                ndcg_val, map_1_val, map_0_val = compute_metrics(model, msn_validation, scaler, imputer)
                current_df = pd.DataFrame([[0., val_loss, ndcg_val, map_1_val, map_0_val]],
                                          columns=["train_loss", "val_loss", "ndcg@10", "map_1", "map_0"])
                df_log = df_log.append(current_df)

                print(current_df)


        train_loss = train_and_compress(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
                                        loggers=[tflogger, pylogger], args=args, val_loader = validation_loader)


        if args.masks_sparsity:
            msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        val_loss = valid(validation_loader, model, criterion, args)
        distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])


        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)
        pd.options.display.float_format = '{:,.4f}'.format

        ndcg_val, map_1_val, map_0_val = compute_metrics(model, msn_validation, scaler, imputer)
        current_df = pd.DataFrame([[train_loss, val_loss, ndcg_val, map_1_val, map_0_val]],
                                  columns=["train_loss", "val_loss", "ndcg@10", "map_1", "map_0"])
        print(current_df)
        df_log = df_log.append(current_df)
        checkpoint_extras = {'ndcg@10': ndcg_val,
                             'map_0': map_0_val,
                             'map_1': map_1_val,
                             'epoch':epoch}

        apputils.save_checkpoint(epoch, len(args.hidden_layers), model, optimizer=optimizer, scheduler=compression_scheduler,
                                 extras=checkpoint_extras, is_best=best_metric < ndcg_val, name=args.name, dir=msglogger.logdir)

        if best_metric < ndcg_val:
            best_model = model
            best_metric = ndcg_val
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
                'ndcg': ndcg_val,
                'map_1': map_1_val,
                'map_0': map_0_val
            }, log_dir=msglogger.logdir, name="best.pth.tar")
        #lr_scheduler.step(val_loss)
        #early_stopping(val_loss, model)

        #if early_stopping.early_stop:
            #print("EARLY STOPPING !!!!")
            #break
    # Finally run results on the test set
    sd = torch.load(os.path.join(msglogger.logdir, "best.pth.tar"))
    model.load_state_dict(sd['state_dict'])
    ndcg_test, map_1_test, map_0_test = compute_metrics(best_model, msn_test, scaler, imputer, cpu=True)
    print("Best model at epoch {}. Ndcg: {:.4f}, map_1: {:.4f}, map_0: {:.4f} ".format(best_epoch, ndcg_test, map_1_test, map_0_test))
    csv_path = os.path.join(msglogger.logdir, "log.csv")
    print("Log file saved to " + csv_path)
    df_log.to_csv(csv_path)

    # write args
    f = open(csv_path, 'a+')
    f.write(str(args))
    f.write("\n")
    f.write("Best model at epoch {} \n. Ndcg: {:.4f}\n map_1: {:.4f}\n map_0: {:.4f} ".format(best_epoch, ndcg_test,
                                                                                              map_1_test,
                                                                                              map_0_test))
    f.close()

    return  ndcg_test



def train_and_compress(train_loader, model, criterion, optimizer, epoch,
                       compression_scheduler, loggers, args, val_loader):
    """Training loop for one epoch."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    state = {k: v for k, v in args._get_kwargs()}
    adjust_learning_rate(optimizer, epoch, args, state )

    bar = Bar('Processing', max=len(train_loader))

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    # Switch to train mode
    model.train()
    model = model.to(args.device)
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        inputs, target = inputs.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)

        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        output = model(inputs)

        loss = criterion(output, target)

        if compression_scheduler:

            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
        # Compute the gradient and do SGD step
        losses.update(loss.data.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        if compression_scheduler:

            compression_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)

        optimizer.step()

        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
            #model = model.to(args.device)

        # measure elapsed time
        batch_time.update(time.time() - end)


        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
            batch=train_step + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,

        )
        bar.next()


    bar.finish()
    return losses.avg


def check_pytorch_version():
    from pkg_resources import parse_version
    if parse_version(torch.__version__) < parse_version('1.0.1'):
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 1.0.1 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 1.0.1 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)




if __name__ == '__main__':
    try:
        check_pytorch_version()

        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
