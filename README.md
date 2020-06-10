# RUC-AIM3: Improved TIRG Model for Fashion-IQ Challenge 2020

This is our code for the Fashion-IQ Challenge 2020. Our code is built based on [TIRG](https://github.com/google/tirg).
The Fashion IQ dataset can be downloaded from [here](https://github.com/XiaoxiaoGuo/fashion-iq). 

Make sure the dataset include these files: `./data/images/*.jpg`

## Setup

- python 3
- pytorch 1.1

## Running Models

To run our training: 
```
model='tirg'
img_encoder='efficientnet'
text_encoder='dualenc'
embed_dim=1024
log_dir=results/$model.$img_encoder.$text_encoder.$embed_dim
if [ ! -d "results" ]; then
  mkdir "results"
fi
if [ ! -d "$log_dir" ]; then
  mkdir "$log_dir"
fi

# Training 
CUDA_VISIBLE_DEVICES=0 python main.py \
--model=$model --img_encoder=$img_encoder --text_encoder=$text_encoder \
--embed_dim=$embed_dim --log_dir=$log_dir \
| tee $log_dir/log.$(date "+%Y%m%d%H%M%S")

```

To run our testing: 
```
model='tirg'
img_encoder='efficientnet'
text_encoder='dualenc'
embed_dim=1024
log_dir=results/$model.$img_encoder.$text_encoder.$embed_dim
 
CUDA_VISIBLE_DEVICES=0 python main.py \
--model=$model --img_encoder=$img_encoder --text_encoder=$text_encoder \
--embed_dim=$embed_dim --log_dir=$log_dir \
--is_test --resume_file $log_dir/best_checkpoint.pth \
--return_test_rank

python convert_sims_to_submit.py \
--model=$model --img_encoder=$img_encoder --text_encoder=$text_encoder \
--embed_dim=$embed_dim
```

Or:
```
sh train.sh
sh test.sh
```