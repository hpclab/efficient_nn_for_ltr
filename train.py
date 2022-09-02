

import argparse
import random
from datetime import datetime
from progress.bar import Bar as Bar
import time
import pandas as pd

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

from torchsummary import summary

from utils.EarlyStopping import EarlyStopping
from utils import MLP
from utils.train_utils import *
from utils.MLP import *
from rankeval.metrics import NDCG, MAP

import utils.train_parser as train_parser

def main():
    args = train_parser.get_parser().parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    name_dir = args.name + "__" + datestring
    log_dir = os.path.join(args.output_dir, name_dir)
    os.makedirs(log_dir)

    msn_train, msn_validation, msn_test, original_model, scaler, imputer, n_features = load_dataset_and_orginal_model(args)
    train_loader, validation_loader, scaler, imputer = create_data_loaders(args, original_model, scaler, msn_train,
                                                                           msn_validation,
                                                                           imputer)

    model = create_model(args, n_features)


    train_evaluate_and_save_model(model, train_loader, validation_loader, msn_validation, msn_test,
                                          scaler, log_dir=log_dir, imputer=imputer, args = args, state = state)



def train_evaluate_and_save_model(model, train_loader, validation_loader, msn_validation, msn_test, scaler, imputer,  log_dir, args, state):

    #Re-intialize seeds in case of multiple training (grid search or bayesian search)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    torch.cuda.manual_seed_all(args.manualSeed)

    start_epoch = 0
    criterion = nn.MSELoss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor=0.5, patience=6 )
    train_losses = []
    val_losses = []
    df_log = pd.DataFrame(columns=["train_loss", "val_loss", "ndcg@10", "map_1", "map_0"])

    best_model = None
    best_metric = 0
    best_epoch = -1

    for epoch in range(start_epoch, args.epochs):
        model = model.cuda()
        adjust_learning_rate(optimizer, epoch, args, state)
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        train_losses.append(train_loss)
        val_loss, valid_scores = valid(validation_loader, model, criterion, epoch)
        val_losses.append(val_loss)


        #ndcg_val, map_1_val, map_0_val = compute_metrics(model, msn_validation, scaler, imputer=imputer)#msn validation is not imputed n    or scaled
        ndcg_val, map_1_val, map_0_val = compute_metrics_2( msn_validation, valid_scores)
        current_df = pd.DataFrame([[train_loss, val_loss, ndcg_val, map_1_val, map_0_val]],
                                  columns=["train_loss", "val_loss", "ndcg@10", "map_1", "map_0"])
        print(current_df)

        df_log = df_log.append(current_df)

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
            }, log_dir=log_dir, name="best.pth.tar")

        #lr_scheduler.step(val_loss)



    sd = torch.load(os.path.join(log_dir, "best.pth.tar"))
    model.load_state_dict(sd['state_dict'])
    ndcg_test, map_1_test, map_0_test = compute_metrics(best_model, msn_test, scaler, imputer=imputer)


    print("Best model at epoch {}. Ndcg: {:.4f}, map_1: {:.4f}, map_0: {:.4f} ".format(best_epoch, ndcg_test, map_1_test, map_0_test))
    csv_path = os.path.join(log_dir, "log.csv")
    print("Log file saved to " + csv_path)
    df_log.to_csv(csv_path)

    # write args
    f = open(csv_path, 'a+')
    f.write(str(args))
    f.write("\n")
    f.write("Best model at epoch {} \n. Ndcg: {:.4f}\n map_1: {:.4f}\n map_0: {:.4f} ".format(best_epoch, ndcg_test, map_1_test,
                                                                                              map_0_test))
    f.close()
    return ndcg_test




def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))

    for batch_idx, (X, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        X, y = torch.autograd.Variable(X), torch.autograd.Variable(y)

        outputs = model(X)

        loss = criterion(outputs, y)

        losses.update(loss.data.item(), X.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
            batch=batch_idx + 1,
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



def valid(test_loader, model, criterion, epoch):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    bar = Bar('Processing', max=len(test_loader))
    total_outputs = []
    for batch_idx, (X, y) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        X, y = X.cuda(), y.cuda()
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.update(loss.data.item(), X.size(0))
        total_outputs.append(outputs.detach().cpu().numpy())
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
            batch=batch_idx + 1,
            size=len(test_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
        )
        bar.next()
    bar.finish()

    scores = np.concatenate(total_outputs)
    #print(scores.shape)
    return losses.avg, scores

def compute_metrics_2(test_rank_eval_db, pred_scores):
    ndcg = NDCG(cutoff=10, no_relevant_results=1.0, implementation="exp")
    map = MAP()
    map_0 = MAP(no_relevant_results=0)

    ndcg_metric = ndcg.eval(test_rank_eval_db, pred_scores)[0]
    map_1_metric = map.eval(test_rank_eval_db, pred_scores)[0]
    map_0_metric = map_0.eval(test_rank_eval_db, pred_scores)[0]
    return ndcg_metric, map_1_metric, map_0_metric


def compute_metrics(model, test_rank_eval_db, scaler, imputer ,cpu = True):
    ndcg = NDCG(cutoff=10, no_relevant_results=1.0, implementation="exp")
    map = MAP()
    map_0 = MAP(no_relevant_results=0)

    scaled_test = test_rank_eval_db.X
    if imputer:
        scaled_test = imputer.transform(scaled_test)
    if scaler:
        scaled_test = scaler.transform(scaled_test)
    else:
        scaled_test = np.log1p(np.abs(scaled_test))* np.sign(scaled_test)

    if cpu:
        model = model.cpu()
    model.eval()

    pred_scores = model(torch.from_numpy(scaled_test)).detach().cpu().numpy()
    ndcg_metric = ndcg.eval(test_rank_eval_db, pred_scores)[0]
    map_1_metric = map.eval(test_rank_eval_db, pred_scores)[0]
    map_0_metric = map_0.eval(test_rank_eval_db, pred_scores)[0]
    return ndcg_metric, map_1_metric, map_0_metric



def create_model(args, n_features):
    size = len(args.hidden_layers)
    print("Size: ", size)

    model = MLP(input_dim=n_features, hidden_dims=args.hidden_layers, drop_prob=args.drop)
    if args.pretrained_model:
        sd = torch.load(args.pretrained_model)
        print("Loading state dict")
        model.load_state_dict(sd['state_dict'])

    model = model.cuda()
    summary(model, (1, n_features))
    return model



def adjust_learning_rate(optimizer, epoch, args, state):
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__== "__main__":
    main()




