import os
from rankeval.dataset.datasets_fetcher import load_dataset
from rankeval.dataset import Dataset as DatasetRankEval
from rankeval.model import RTEnsemble
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils.DataReader import *

from collections import defaultdict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_dataset_and_orginal_model(args):
    check_prefix = args.dataset_path
    #just for debug
    if not os.path.exists(check_prefix):
        check_prefix = "/data1/letor-datasets/"
    imputer = None
    scaler = None
    if args.dataset_name == "msn10k":
        print("Loading MSN 10k")
        dataset_container = load_dataset(dataset_name=args.dataset_name,
                                         fold='1',
                                         download_if_missing=True,
                                         force_download=False,
                                         with_models=True)
        train_dataset = dataset_container.train_dataset
        validation_dataset = dataset_container.validation_dataset
        test_dataset = dataset_container.test_dataset
        scaler = StandardScaler()
        scaler.fit(train_dataset.X)
    elif args.dataset_name == "msn30k":
        print("Loading MSN 30k")
        dataset_name = "msn30k/Fold1/"
        dataset_prefix = check_prefix + "{}".format(dataset_name)
        rankeval_datasets = {}
        for split in ["train", "vali", "test"]:
            rankeval_datasets[split] = DatasetRankEval.load("{}/{}.txt".format(dataset_prefix, split))
        train_dataset = rankeval_datasets['train']
        validation_dataset = rankeval_datasets['vali']
        test_dataset = rankeval_datasets['test']
        scaler = StandardScaler()
        scaler.fit(train_dataset.X)

    elif args.dataset_name == "istella":
        print("Loading Istella Sample")
        dataset_name = "tiscali/sample/"
        dataset_prefix = check_prefix + "{}".format(dataset_name)
        rankeval_datasets = {}
        for split in ["train", "vali", "test"]:
            rankeval_datasets[split] = DatasetRankEval.load("{}/{}.txt".format(dataset_prefix, split))
        train_dataset = rankeval_datasets['train']
        validation_dataset = rankeval_datasets['vali']
        test_dataset = rankeval_datasets['test']
        val_max = np.max(train_dataset.X)
        imputer = SimpleImputer(missing_values=val_max, strategy='mean')
        imputer.fit(train_dataset.X)

    if args.original_model.endswith("msn"):
        original_model = RTEnsemble(args.original_model_path, name="Original Model", format="LightGBM")

    elif args.original_model.endswith("istella"):
        original_model = RTEnsemble(args.original_model_path, name="Original Model", format="QuickRank")
    else:
        original_model = None

    n_features = train_dataset.X.shape[1]

    return train_dataset, validation_dataset, test_dataset, original_model, scaler, imputer, n_features




def create_data_loaders(args, original_model, scaler, train, validation, imputer):

    if args.dataset_name == "msn30k" or args.dataset_name == "msn10k":
        midpoints = compute_midpoints(train, original_model)

        train_dataset = MSNDataReader(train.X, original_model, scaler)
        batch_creator = MSNBatchCreator(midpoints, original_model, scaler,
                                        n_artificial_samples=int(args.batch_size * args.percentage_of_art_data))
        validation_dataset = MSNDataReader(validation.X, original_model, scaler)

        train_loader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, num_workers=0, collate_fn=batch_creator)
        validation_loader = DataLoader(validation_dataset, args.batch_size,
                                       shuffle=False, num_workers=0)


    elif args.dataset_name == "istella":

        print("IStella dataloaders")
        imputed_X = imputer.transform(train.X)
        scaler = StandardScaler()
        scaler.fit(imputed_X)

        midpoints = compute_midpoints2(imputed_X, original_model, train.X.shape[1])
        train_dataset = IstellaDataReader(train.X, original_model, scaler, imputer)
        batch_creator = BatchCreatorIstella(midpoints, original_model, scaler,
                                            n_artificial_samples=int(args.batch_size * args.percentage_of_art_data))
        validation_dataset = IstellaDataReader(validation.X, original_model, scaler, imputer)
        train_loader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, num_workers=0, collate_fn=batch_creator)
        validation_loader = DataLoader(validation_dataset, args.batch_size,
                                       shuffle=False, num_workers=0)

    return train_loader, validation_loader, scaler, imputer




def compute_midpoints(dataset, model):

    thresholds = defaultdict(list)

    for f, (min_v, max_v) in enumerate(zip(dataset.X.min(axis=0), dataset.X.max(axis=0))):
        thresholds[f].append(min_v)
        thresholds[f].append(max_v)

    for node in np.arange(model.n_nodes):
        if not model.is_leaf_node(node):
            f = model.trees_nodes_feature[node]
            v = model.trees_nodes_value[node]
            thresholds[f].append(v)

    midpoints = np.ndarray(dataset.n_features, dtype=np.object)
    for f in np.arange(dataset.n_features):
        v = np.unique(thresholds[f])
        midpoints[f] = (v[1:] + v[:-1]) / 2

    return midpoints


def compute_midpoints2(X, model, n_features):

    thresholds = defaultdict(list)

    for f, (min_v, max_v) in enumerate(zip(X.min(axis=0), X.max(axis=0))):
        thresholds[f].append(min_v)
        thresholds[f].append(max_v)

    for node in np.arange(model.n_nodes):
        if not model.is_leaf_node(node):
            f = model.trees_nodes_feature[node]
            v = model.trees_nodes_value[node]
            thresholds[f].append(v)

    midpoints = np.ndarray(n_features, dtype=np.object)
    for f in np.arange(n_features):
        v = np.unique(thresholds[f])
        midpoints[f] = (v[1:] + v[:-1]) / 2

    return midpoints


def save_checkpoint(state,  log_dir , name):
    filepath = os.path.join(log_dir, name)
    torch.save(state, filepath)


