python3 ~/workspace/dgl-baseline/tools/launch.py --workspace ~/workspace/dgl-baseline/src \
    --num_trainers 8 \
    --num_servers 1 \
    --num_samplers 1 \
    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json \
    --ip_config ip_config_4p.txt \
    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/baseline_graphsage.py --graph_name ogbn-products \
    --ip_config ip_config_4p.txt --num_gpus 8 --num_epochs 1 --eval_every 1 \
    --num_hidden 256 --model_num_hidden 256 --lr 0.003 --dropout 0.5 --dgl_sparse --fan_out "10,25"" | tee ../logs/2024_03_07_graphsage_papers100M_10_25.log

 


