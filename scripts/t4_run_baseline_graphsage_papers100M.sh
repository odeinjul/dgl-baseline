python3 ~/workspace/dgl-baseline/tools/launch_train.py  --workspace ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 --num_samplers 1 --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/papers100m_ud_4p/ogb-paper100M.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_graphsage.py --num_hidden 128 --graph_name ogb-paper100M --ip_config ip_config4.txt \
    --num_gpus 8 --num_epochs 5 --eval_every 21 --fan_out "10,25" \
    --num_hidden 256 --model_num_hidden 256 --lr 0.003 --dropout 0.5 --dgl_sparse"

python3 ~/workspace/dgl-baseline/tools/launch_train.py  --workspace ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 --num_samplers 1 --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/papers100m_ud_4p/ogb-paper100M.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_graphsage.py --num_hidden 128 --graph_name ogb-paper100M --ip_config ip_config4.txt \
    --num_gpus 8 --num_epochs 5 --eval_every 21 --fan_out "5,10,15" \
    --num_hidden 256 --model_num_hidden 256 --lr 0.003 --dropout 0.5 --dgl_sparse"

python3 ~/workspace/dgl-baseline/tools/launch_train.py  --workspace ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 --num_samplers 1 --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/papers100m_ud_4p/ogb-paper100M.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_graphsage.py --num_hidden 128 --graph_name ogb-paper100M --ip_config ip_config4.txt \
    --num_gpus 8 --num_epochs 5 --eval_every 21 --fan_out "5,5,5,5" \
    --num_hidden 256 --model_num_hidden 256 --lr 0.003 --dropout 0.5 --dgl_sparse"

python3 ~/workspace/dgl-baseline/tools/launch_train.py --workspace ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 --num_samplers 1 --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/papers100m_ud_4p/ogb-paper100M.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_graphsage.py --num_hidden 128 --graph_name ogb-paper100M --ip_config ip_config4.txt \
    --num_gpus 8 --num_epochs 5 --eval_every 21 --fan_out "5,5,5,5,5,5" \
    --num_hidden 256 --model_num_hidden 256 --lr 0.003 --dropout 0.5 --dgl_sparse"