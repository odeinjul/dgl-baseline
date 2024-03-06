python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/datasets/papers100m_ud_4p/ogb-paper100M.json \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 baseline_graphsage.py --graph_name ogb-paper100M \
    --ip_config ip_config_4p.txt --batch_size 1000 --num_gpus 8 --eval_every 1 --num_epochs 20 \
    --fan_out "5,10,15" --lr 0.003 --sparse_lr 0.01 --dropout 0.5 \
    --dgl_sparse --num_layers 3 --num_hidden 256"

    --num_gpus 8 --num_epochs 5 --eval_every 21 --fan_out "10,25" \
    --num_hidden 256 --model_num_hidden 256 --lr 0.003 --dropout 0.5 --dgl_sparse"