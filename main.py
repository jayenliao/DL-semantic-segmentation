
import smp, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from args import init_arguments
from utils import Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing, create_subfolders, fns2subfolders
from PIL import Image
from datetime import datetime
from torch.utils.data import DataLoader

def main(args):

    # [1] Prepare datasets and dataloaders

    create_subfolders(args.PATH_x)
    create_subfolders(args.PATH_y)
    fns2subfolders(args.PATH_x, args.size_tr, args.size_te, args.seed)
    fns2subfolders(args.PATH_y, args.size_tr, args.size_te, args.seed)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.ENCODER, args.ENCODER_WEIGHTS)
    dataset_nonarg_tr = Dataset(args.PATH_x + 'tr/', args.PATH_y + 'tr/', classes=args.CLASSES, CLASSES=args.CLASSES)

    dataset_tr = Dataset(
        args.PATH_x + 'tr/', args.PATH_y + 'tr/',
        classes=args.CLASSES, CLASSES=args.CLASSES,
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    dataset_va = Dataset(
        args.PATH_x + 'va/', args.PATH_y + 'va/',
        classes=args.CLASSES, CLASSES=args.CLASSES,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    dataset_te = Dataset(
        args.PATH_x + 'te/', args.PATH_y + 'te/',
        classes=args.CLASSES, CLASSES=args.CLASSES,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    loader_tr = DataLoader(dataset_tr, batch_size=args.batch_size, num_workers=args.workers_tr)
    loader_va = DataLoader(dataset_va, batch_size=1, num_workers=args.workers_ev)
    loader_te = DataLoader(dataset_te, batch_size=1, num_workers=args.workers_ev)


    # [2] Create segmentation model with the pretrained encoder and set up for training.
    
    if args.trained_model == '':
        print('Train a new model ...')
        model = smp.FPN(
            encoder_name=args.ENCODER, 
            encoder_weights=args.ENCODER_WEIGHTS, 
            classes=len(args.CLASSES), 
            activation=args.ACTIVATION,
        )
    else:
        model = torch.load(args.savePATH + args.trained_model)

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])

    # Create epoch runners, which are simple loops of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, loss=loss, metrics=metrics, 
        optimizer=optimizer, device=args.DEVICE, verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, 
        device=args.DEVICE, verbose=True,
    )


    # [3] Train model for 40 epochs with validation
    max_score = 0
    dt = datetime.now().strftime('%d-%H-%M-%S')
    folder_name = args.savePATH + args.dt + '_' + '_bs=' + str(args.batch_size) + '_epochs=' + str(args.epochs) + '/'

    for i in range(40):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(loader_tr)
        valid_logs = valid_epoch.run(loader_va)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, folder_name+'best_model.pth')
            print('Model saved!')
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


    # [4] Evaluate model on test set
    model = torch.load('./best_model.pth')
    test_epoch = smp.utils.train.ValidEpoch(model=model, loss=loss, metrics=metrics, device=args.DEVICE)
    logs = test_epoch.run(loader_te)
