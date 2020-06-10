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
