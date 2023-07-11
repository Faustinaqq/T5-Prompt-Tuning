CUDA_VISIBLE_DEVICES=5 python main.py --model t5-small  --batch_size 16 --log_file log.txt \
--lr 5e-04 --prompt_len 100 --ft_way prompt --changed 0 --init class_label
