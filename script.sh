#! /bin/bash

python copy_paste.py --name YOURNAME --temp_file_type png --left 0.2 --upper 0.2 --right 0.8 --bottom 0.8 --max_tem 50 --min_tem 30 --gen_num_per_base 100
sleep 5
python save_json.py --name YOURNAME --phase train --oc 0.9
sleep 5
python train_aug.py --name YOURNAME 
