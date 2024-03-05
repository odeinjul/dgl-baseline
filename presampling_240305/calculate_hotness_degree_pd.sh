
python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_1p/ogbn-products.json  \
    --ip_config ip_config_1p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogbn-products \
    --ip_config ip_config_1p.txt --fan_out "10,25"  "


python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_1p/ogbn-products.json  \
    --ip_config ip_config_1p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogbn-products \
    --ip_config ip_config_1p.txt --fan_out "10,15"  "

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_1p/ogbn-products.json  \
    --ip_config ip_config_1p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogbn-products \
    --ip_config ip_config_1p.txt --fan_out "10,10,10,10"  "

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_1p/ogbn-products.json  \
    --ip_config ip_config_1p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogbn-products \
    --ip_config ip_config_1p.txt --fan_out "5,5,10,15"  "

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_1p/ogbn-products.json  \
    --ip_config ip_config_1p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogbn-products \
    --ip_config ip_config_1p.txt --fan_out "10,10,10,10,10,10"  "

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_1p/ogbn-products.json  \
    --ip_config ip_config_1p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogbn-products \
    --ip_config ip_config_1p.txt --fan_out "5,5,5,5,10,15"  "