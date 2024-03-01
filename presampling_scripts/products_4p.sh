python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.1 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.2 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.3 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.4 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.5 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.6 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.7 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.8 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.9 --from_top"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.1"

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.2"
    python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.3"

    python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.4"

    python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.5"

    python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.6"

    python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.7"

    python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.8"

    python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json  \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_presampling.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256 --presampling --hot_rate 0.9"
