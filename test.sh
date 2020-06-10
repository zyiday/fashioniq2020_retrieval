model='tirg'
img_encoder='efficientnet'
text_encoder='dualenc'
embed_dim=1024
log_dir=results/$model.$img_encoder.$text_encoder.$embed_dim

# Testing 
CUDA_VISIBLE_DEVICES=0 python main.py \
--model=$model --img_encoder=$img_encoder --text_encoder=$text_encoder \
--embed_dim=$embed_dim --log_dir=$log_dir \
--is_test --resume_file $log_dir/best_checkpoint.pth \
--return_test_rank

python convert_sims_to_submit.py \
--model=$model --img_encoder=$img_encoder --text_encoder=$text_encoder \
--embed_dim=$embed_dim --split='test'