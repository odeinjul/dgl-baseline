python3 ~/workspace/dgl-baseline/tools/launch_train.py --workspace  ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_gat.py --num_hidden 256 --graph_name ogbn-products --ip_config ip_config4.txt --num_gpus 8 --num_epochs 20 --eval_every 1 \
   --num_hidden 256 --model_num_hidden 32 --heads "8,8,1" --feat_dropout 0.1 --attn_dropout 0.1 --lr 0.003 --dropout 0.5 --dgl_sparse --fan_out 10,25"

python3 ~/workspace/dgl-baseline/tools/launch_train.py --workspace  ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_gat.py --num_hidden 256 --graph_name ogbn-products --ip_config ip_config4.txt --num_gpus 8 --num_epochs 20 --eval_every 1 \
   --num_hidden 256 --model_num_hidden 32 --heads "8,8,1" --feat_dropout 0.1 --attn_dropout 0.1 --lr 0.003 --dropout 0.5 --dgl_sparse --fan_out 5,5,5,5"

python3 ~/workspace/dgl-baseline/tools/launch_train.py --workspace  ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 1 \
   --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
   --ip_config ip_config4.txt \
   "~/workspace/embcache_venv/bin/python3 baseline_gat.py --num_hidden 256 --graph_name ogbn-products --ip_config ip_config4.txt --num_gpus 8 --num_epochs 20 --eval_every 1 \
   --num_hidden 256 --model_num_hidden 32 --heads "8,8,1" --feat_dropout 0.1 --attn_dropout 0.1 --lr 0.003 --dropout 0.5 --dgl_sparse --fan_out 5,5,5,5,5,5"

