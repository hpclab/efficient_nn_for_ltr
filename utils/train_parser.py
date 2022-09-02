import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MLP Training for Learning to Rank',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Datasets
    parser.add_argument("--dataset-name", type=str, default="msn30k", help="Dataset ",
                        choices=['msn10k', 'msn30k', 'istella'])
    parser.add_argument("--dataset-path", type=str, default="/data/letor-datasets/", help="Path to the dataset folder")

    #Original Model
    parser.add_argument("--original-model", type=str, default="LM600_msn",
                        help="Model to approximate with the Multi Layer Perceptron",
                        choices=['LM2500_istella', 'LM600_msn', 'LM800_msn'])
    parser.add_argument("--original-model-path", type=str, default="./best_lgb_msn30kf1_256leaves.txt",
                        help="Path to the pre-trained ensemble of regression trees")

    #Training parameters
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=2500, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate ')
    parser.add_argument('--schedule', type=int, nargs='+', default=[90, 130],
                        help='Decrease learning rate at these epochs.')

    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                        metavar='W', help='weight decay ')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--percentage-of-art-data', type=float, default=1.0, help="Percentage of artificial data per batch")


    #Network Parameters
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[500, 500, 500, 100],
                        help="number of neuron per hidden layers; 4 values for large models  \n 2 values for small models  ")

    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio. Dropout is applied to the first layer')

    #Saving Parameters
    parser.add_argument('--name', type=str,
                        help="The name of the directory where the model will be saved")
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='compression_logs',
                        help='Path to dump logs and checkpoints')

    parser.add_argument("--pretrained-model", type=str, help="Path to the neural network pretrained model")


    return parser