# BinCopyPaste
BinCopyPaste: Several Clicks to build datasets for instance segmentation in bin-picking scenarios

## Introduction
By BinCopyPaste, with only little human-labeling, you can create your own dataset for instance segmentation in dense and occluded bin-picking scenarios. Given templates (object masks) and backgrounds, BinCopyPaste simulates scenes of accumulated bins by pasting objects with random 2d poses onto the background image. It is able to generate a large amount of training data automatically for deep learning-based instance segmentation.
## Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.6 and torchvision that matches the PyTorch installation.
- labelme: use labelme to get the template json files, see https://github.com/wkentaro/labelme for installation and usage.
- detectron2: for training and label visualization, install detectron2 according to https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
## Usage
### prerequisite
- use labelme to label the masks of template objects from sources images, which should generate several .json files.
- create a new folder called "YOUR_NAME". under "YOUR_NAME" folder, create another two folders and name them "tm" and "bg".
- put both the sources imgs and .json files into the "tm" folder.
- put background images into the "bg" folder.
### generate some training data 
```bash
python copy_paste.py --name YOURNAME --temp_file_type png(or else) --left 0.2 --upper 0.2 --right 0.8 --bottom 0.8 --max_tem 50 --min_tem 30 --gen_num_per_base 100
````
#### arguments: 
- **temp_file_type** is the format of your source images 
- **left, right, upper, bottom** is the effective range of the pasted objects, for example, left=0.2 means that the objects should not be pasted closer than 0.2*total_width to the image's left border.
- the number of objects on a background is randomly picked from the range (**min_tem, max_tem**)
- **gen_num_per_base** is the number of data generated from each base. The total amount of data is gen_num_per_base*num_background
#### outputs
- YOURNAME/train
- YOURNAME/meta
### and have a check
```bash
python vis_dataset.py --name YOURNAME 
````
### automatically train a instance segmentation network after data generation
```bash
sh script.sh 
````
don't forget to adjust your own arguments in the script.sh!
also, for adjusting training hyper-parameters, user_config.yaml can be modified.
#### outputs 
- train_output :tensorboard events and models.

### test on real data using trained models
```bash
python test.py --name YOURNAME --model_name YOUR_MODEL_NAME --test_dir YOUR_TEST_DIR
````
