# python3 ~/workspace/dgl-baseline/tools/launch.py --workspace  ~/workspace/dgl-baseline/src/ \
#    --num_trainers 8 \
#    --num_samplers 1 \
#    --num_servers 1 \
#    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json \
#    --ip_config ip_config_4p.txt \
#    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/baseline_gat.py --graph_name ogbn-products \
#    --ip_config ip_config_4p.txt --num_gpus 8 --num_epochs 20 --eval_every 1 \
#    --num_hidden 256 --model_num_hidden 32 --heads "8,1" --feat_dropout 0.1 --attn_dropout 0.1 --lr 0.003 --dgl_sparse --fan_out "10,25"" | tee ../logs/2024_03_07_gat_products_10_25.log

# python3 ~/workspace/dgl-baseline/tools/launch.py --workspace  ~/workspace/dgl-baseline/src/ \
#    --num_trainers 8 \
#    --num_samplers 1 \
#    --num_servers 1 \
#    --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json \
#    --ip_config ip_config_4p.txt \
#    "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/baseline_gat.py --graph_name ogbn-products \
#    --ip_config ip_config_4p.txt --num_gpus 8 --num_epochs 20 --eval_every 1 \
#    --num_hidden 256 --model_num_hidden 32 --heads "8,8,8,1" --feat_dropout 0.1 --attn_dropout 0.1 --lr 0.003 --dgl_sparse --fan_out "5,5,5,5"" | tee ../logs/2024_03_07_gat_products_5_5_5_5.log

python3 ~/workspace/dgl-baseline/tools/launch.py --workspace  ~/workspace/dgl-baseline/src/ \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 1 \
   --part_config /home/ubuntu/workspace/data/products_4p/ogbn-products.json \
   --ip_config ip_config_4p.txt \
   "/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/baseline_gat.py --graph_name ogbn-products \
   --ip_config ip_config_4p.txt --num_gpus 8 --num_epochs 20 --eval_every 1 \
   --num_hidden 256 --model_num_hidden 32 --heads "8,8,8,8,8,1" --feat_dropout 0.1 --attn_dropout 0.1 --lr 0.003 --dgl_sparse --fan_out "5,5,5,5,5,5"" | tee ../logs/2024_03_07_gat_products_5_5_5_5_5_5.log
