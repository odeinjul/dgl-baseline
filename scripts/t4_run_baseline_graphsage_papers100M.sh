python3 example/launch_train.py --workspace ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/papers100m_ud_4p/ogb-paper100M.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_graphsage.py --num_hidden 128 --graph_name ogb-paper100M --ip_config ip_config4.txt --num_gpus 8 --num_epochs 5 --eval_every 21 --dgl_sparse"