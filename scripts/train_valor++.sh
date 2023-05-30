python main_avvp.py \
--seed 87 \
--mode train \
--model MMIL_Net \
--model_name model_VALOR++_reproduce \
--batch_size 64 \
--epochs 60 \
--audio_dir ./data/CLAP/features \
--video_dir ./data/CLIP/features \
--st_dir ./data/feats/r2plus1d_18 \
--label_train ./data/AVVP_train.csv \
--label_val ./data/AVVP_val_pd.csv \
--label_test ./data/AVVP_test_pd.csv \
--optimizer adamw \
--weight_decay 1e-3 \
--scheduler warm_up_cos_anneal \
--warm_up_epoch 10 \
--grad_norm 1.0 \
--lr 3e-4 \
--lr_min 3e-6 \
--beta1 0.5 \
--eps 1e-8 \
--hidden_dim 256 \
--nhead 8 \
--ff_dim 1024 \
--num_layers 4 \
--norm_where post_norm \
--v_pseudo_data_dir ./data/CLIP/segment_pseudo_labels \
--a_pseudo_data_dir ./data/CLAP/segment_pseudo_labels \