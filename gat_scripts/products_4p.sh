python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_gat.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --lr 0.003 --sparse_lr 0.01\
    --dgl_sparse --num_hidden 256 --heads 8"
