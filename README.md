# DL-semantic-segmentation

## Data Files

### Dataset Contents

This dataset version consists of 10K images with annotations for 7 tasks.

- RGB images
- Semantic segmentation
- 2D bounding boxes
- Instance segmentation
- Motion segmentation
- Previous images
- CAN information
- Lens soiling data and annotations
- Calibration Information
- Dense polygon points for objects

Coming Soon:

- Fisheye sythetic data with semantic annotations
- Lidar and dGPS scenes

To download data or to learn more details about the data, please go [here](https://github.com/valeoai/WoodScape) and [here](https://competitions.codalab.org/competitions/30993).

### The whole picture of files

- `./privous_images/` contains 8234 png files.

- `./rgb_images/` contains 8234 png files.

- `./rgb_images(test_set)/` contains 1766 png files.

- `./train/`:
    - `./train/gtLabels/` contains 5000 png files (black).
    - `./train/rgbImages/` contains 5000 png files (normal).
    - `./train/rgbLabels/` contains 5000 png files (RGB labels).
    
- `./test/`:
    - `./test/gtLabels/` contains 1000 png files (black).
    - `./test/rgbImages/` contains 1000 png files (normal).
    - `./test/rgbLabels/` contains 1000 png files (RGB labels).
    
- `./WoodScape_ICCV19/`:
    - `~/box_2d_annotations/`:
        - `box_2d_annotation_info.json`
        - `~/box_2d_annotations/` contains 8234 txt files.
        
    - `~/calibration_data/`:
        - `calibration_readme.md`
        - `~/calibration/` contains 8234 json files.
   
    - `~/instance_annotations/`:
        - `~/class_info.json`
        - `~/instance_annotations/` contains 8234 json files.
   
    - `~/motion_annotations/`:
        - `~/motion_annotation_info.json`
        - `~/motion_annotations/`:
            - `~/gtLabels/` contains 8234 png files.
            - `~/rgbLabels` contains 8234 png files.
   
    - `~/semantic_annotations/`:
        - `seg_annotation_info.json`
        - `~/semantic_annotations/`:
            - `~/gtLabels/` contains 8234 png files.
            - `~/rgbLabels` contains 8234 png files.
            
    - `~/soiling_dataset/`:
        - `~/soiling_annotation_info.json`
    
    - `~/vehicle_data/`:
        - `~/vehicle_info/` contains 8234 json files.

To download data or to learn more details about the data, please go [here](https://github.com/valeoai/WoodScape) and [here](https://competitions.codalab.org/competitions/30993).

### Data organization

```
woodscape
│   README.md    
│
└───rgb_images
│   │   00001_[CAM].png
│   │   00002_[CAM].png
|   |   ...
│   │
└───previous_images
│   │   00001_[CAM]_prev.png
│   │   00002_[CAM]_prev.png
|   |   ...
│   │
└───semantic_annotations
        │   rgbLabels
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
        │   gtLabels
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
│   │
└───box_2d_annotations
│   │   00001_[CAM].png
│   │   00002_[CAM].png
|   |   ...
│   │
└───instance_annotations
│   │   00001_[CAM].json
│   │   00002_[CAM].json
|   |   ...
│   │
└───motion_annotations
        │   rgbLabels
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
        │   gtLabels
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
│   │
└───vehicle_data
│   │   00001_[CAM].json
│   │   00002_[CAM].json
|   |   ...
│   │
│   │
└───calibration_data
│   │   00001_[CAM].json
│   │   00002_[CAM].json
|   |   ...
│   │
└───soiling_dataset
        │   rgb_images
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
        │   gt_labels
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
        │   gt_labels
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
```

[CAM] :

- FV --> Front CAM
- RV --> Rear CAM
- MVL --> Mirror Left CAM
- MVR --> Mirror Right CAM

To download data or to learn more details about the data, please go [here](https://github.com/valeoai/WoodScape) and [here](https://competitions.codalab.org/competitions/30993).

## Code

- `args.py` defines the parser of arguments.

- `main.py` defines the main program.

- `predict.py` predict labels for testing images and generate output PNG files.

- `utils.py` defines appropriate formats of dataset and data loader and pipelines of data augmentation.

- `./smpgit/segmentation_models_pytorch/`: socure codes of [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch)

## Usage

```
python3 main.py -enc 'efficientnet-b4' -e 100 -d 'cuda:0'
python3 predict.py -Pm '16-02-58-45efficientnet-b4_bs=8_epochs=30'
```

## Reference

1. Saravanabalagi Ramachandran, John McDonald, and Ganesh Sistu (2021). Woodscape Fisheye Semantic Segmentation for Autonomous Driving | CVPR 2021 OmniCV Workshop Challenge. https://competitions.codalab.org/competitions/30993.

2. Pavel Yakubovskiy (2020). Segmentation Models Pytorch. _GitHub_: GitHub repository. https://github.com/qubvel/segmentation_models.pytorch.

2. Jeremy Jordan (2018). Evaluating image segmentation models. https://www.jeremyjordan.me/evaluating-image-segmentation-models/.

3. Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587.

4. Arunava Chakraborty (2019). PyTorch for Beginners: Semantic Segmentation using torchvision. https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/.

5. Saurabh Kumar (2020). Semantic hand segmentation using Pytorch. _Medium_: towards data science. https://towardsdatascience.com/semantic-hand-segmentation-using-pytorch-3e7a0a0386fa.
