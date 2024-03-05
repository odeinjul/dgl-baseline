python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/ogbn-papers100m_4p_ud/ogb-paper100M.json \
    --ip_config ip_config_4p.txt   \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogb-paper100M \
    --ip_config ip_config_4p.txt --fan_out "10,15" "

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/ogbn-papers100m_1p_ud/ogb-paper100M.json \
    --ip_config ip_config_1p.txt   \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogb-paper100M \
    --ip_config ip_config_1p.txt --fan_out "10,25" "

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/ogbn-papers100m_1p_ud/ogb-paper100M.json \
    --ip_config ip_config_1p.txt   \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogb-paper100M \
    --ip_config ip_config_1p.txt --fan_out "5,5,10,15" "

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/ogbn-papers100m_1p_ud/ogb-paper100M.json \
    --ip_config ip_config_1p.txt   \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree.py --graph_name ogb-paper100M \
    --ip_config ip_config_1p.txt --fan_out "5,5,5,5" "