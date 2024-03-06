python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/mag240_4p_ud/mag240m.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling_only.py --graph_name mag240m \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 0 \
    --fan_out "10,25" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling"