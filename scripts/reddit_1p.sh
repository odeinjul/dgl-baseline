python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/reddit_4p/reddit.json \
    --ip_config ip_config_1p.txt   \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive.py --graph_name reddit \
    --ip_config ip_config_1p.txt --batch_size 1000 --num_gpus 8 --eval_every 5 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256"
