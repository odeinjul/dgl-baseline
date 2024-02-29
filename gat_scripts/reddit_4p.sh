python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/reddit_4p/reddit.json \
    --ip_config ip_config_4p.txt   \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_gat.py --graph_name reddit \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --num_layers 3 --lr 0.003 --sparse_lr 0.01 --feat_dropout 0.1 --attn_dropout 0.1\
    --dgl_sparse --num_hidden 256 --heads 8"
