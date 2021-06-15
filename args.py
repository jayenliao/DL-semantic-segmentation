import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='DL-hw7: semantic segmentation')

    # General
    parser.add_argument('-s', '--seed', type=int, default=4028)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-Pt', '--testPATH', type=str, default='./rgb_images(test_set)/')
    parser.add_argument('-Px', '--PATH_x', type=str, default='./rgb_images/')
    parser.add_argument('-Py', '--PATH_y', type=str, default='./WoodScape_ICCV19/semantic_annotations/semantic_annotations/gtLabels/')
    parser.add_argument('-Ps', '--savePATH', type=str, default='./output/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('-Pm', '--modelPATH', type=str, default='')
    parser.add_argument('-FNm', '--modelFN', type=str, default='')
    parser.add_argument('-c', '--CLASSES', type=str, nargs='+', default=['void', 'road', 'lanemarks', 'curb', 'person', 'rider','vehicles', 'bicycle','motorcycle', 'traffic_sign'])
    
    # Preprocessing
    parser.add_argument('-str', '--size_tr', type=float, default=.8)
    parser.add_argument('-sva', '--size_va', type=float, default=.1)
    parser.add_argument('-wtr', '--num_workers_tr', type=int, default=12)
    parser.add_argument('-wev', '--num_workers_ev', type=int, default=4)

    # Model structure
    parser.add_argument('-enc', '--ENCODER', type=str, default='se_resnext50_32x4d')
    parser.add_argument('-encW', '--ENCODER_WEIGHTS', type=str, default='imagenet')
    parser.add_argument('-a', '--ACTIVATION', type=str, default='sigmoid', choices=['sigmoid', 'softmax2d'])

    # Training
    parser.add_argument('-d', '--DEVICE', type=str, default='cuda:1', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], help='Device name')
    parser.add_argument('-o', '--optimizer', type=str, default='Adam', choices=['SGD', 'Momentum', 'AdaGrad', 'Adam'], help='Optimizer')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='No. of epochs')
    parser.add_argument('-tm', '--trained_model', type=str, default='', help='File name of the trained model that is going to keep being trained or to be evaluated. Set an empty string if not using a pretrained model.')
    parser.add_argument('-pfs', '--plot_figsize', nargs='+', type=int, default=[8, 6], help='Figure size of model performance plot. Its length should be 2.')
    parser.add_argument('-de', '--debug', action='store_true')

    return parser