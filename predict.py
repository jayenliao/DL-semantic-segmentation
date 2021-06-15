import os, sys, cv2, torch#, smp
import smpgit.segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Dataset, get_validation_augmentation, get_preprocessing_no_mask
from args import init_arguments

def produce_predicted_images(args, best_model=None):
    if not best_model:
        best_model_fn = os.path.join(args.savePATH, args.modelPATH, 'best_model.pth')
        best_model = torch.load(best_model_fn)
        print('The best model', best_model_fn, 'is loaded.')
    best_model.to(args.DEVICE)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.ENCODER, args.ENCODER_WEIGHTS)
    dataset_test = Dataset(
        args.testPATH, None,
        classes=args.CLASSES, CLASSES=args.CLASSES,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing_no_mask(preprocessing_fn)
    )
    loader_test = DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers_ev)
    print('The data loader is prepared for prediction.')

    saveFOLDER = args.modelFN.replace('best_model.pth', '') if args.modelPATH == '' else os.path.join(args.savePATH, args.modelPATH)
    print('saveFOLDER', saveFOLDER)
    best_model.eval()
    with tqdm(loader_test, desc='Output Prediction', file=sys.stdout, disable=False) as iterator:
        for i, x in enumerate(iterator):
            x = x.to(args.DEVICE)
            pr_mask = best_model.predict(x)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            pr_mask = pr_mask.argmax(axis=0)
            pr_mask = cv2.copyMakeBorder(pr_mask, 1, 1, 0, 0, cv2.BORDER_REPLICATE)
            pr_mask = np.array([pr_mask]*3)
            pr_mask = pr_mask.reshape((pr_mask.shape[1], pr_mask.shape[2], 3))
            fn = os.path.join(saveFOLDER, dataset_test.ids[i])
            cv2.imwrite(fn, pr_mask)
    print(f'A total of {len(dataset_test)} images are saved under\n--> {saveFOLDER}.')

if __name__ == '__main__':
    args = init_arguments().parse_args()
    produce_predicted_images(args)
