/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine.py --graph_name ogbn-products   \
 --ip_config ip_config_1p.txt --fan_out "10,15" 

/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine.py --graph_name ogbn-products    \
 --ip_config ip_config_1p.txt --fan_out "10,25" 

/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine.py --graph_name ogbn-products    \
 --fan_out "5,5,10,15" 

/home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine.py --graph_name ogbn-products    \
 --ip_config ip_config_1p.txt --fan_out "10,10,10,10" 

 /home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine.py --graph_name ogbn-products    \
 --ip_config ip_config_1p.txt --fan_out "5,5,5,5,10,15" 

 /home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine.py --graph_name ogbn-products    \
 --ip_config ip_config_1p.txt --fan_out "10,10,10,10,10,10" 

 /home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine_final.py \
 --graph_name ogbn-products 

  /home/ubuntu/anaconda3/envs/dglbase/bin/python3 ~/workspace/dgl-baseline/src/train_dist_transductive_hotness_degree_single_machine_final.py \
 --graph_name ogb-paper100M