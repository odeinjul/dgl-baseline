import argparse
import time
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt   

def main(args):
    print("Loading training data")
    save_fn =  os.path.expanduser(f'../src/result/{args.graph_name}_degree_list.pt')
    degree_list = th.load(save_fn)
    print("Result loaded from {}".format(save_fn))
    save_fn = os.path.expanduser(f'../src/result/{args.graph_name}_presampling_heat_{args.fan_out}.pt')
    hotness_list = th.load(save_fn)
    print("Result loaded from {}".format(save_fn))
        
    degree_list = degree_list[hotness_list != 0]
    hotness_list = hotness_list[hotness_list != 0]

    sorted_hotness, sorted_indices = th.sort(hotness_list)
    sorted_degree = degree_list[sorted_indices]

    print(f"Max degree: {th.max(sorted_degree)}, Mean degree: {th.mean(th.Tensor.float(sorted_degree))}, Min degree: {th.min(sorted_degree)}")

    hotness_degree = th.zeros((20, ), dtype=th.float32)
    hotness_count = th.zeros((20, ), dtype=th.float32)
    for i in range(20):
        start_value = i / 20
        end_value = (i + 1) / 20
        indices = th.where((sorted_hotness >= start_value) & (sorted_hotness < end_value))
        hotness_count[i] = (start_value + end_value) / 2
        hotness_degree[i] = th.mean(th.Tensor.float(sorted_degree[indices]))
    
    plt.plot(hotness_count, hotness_degree, label=f'{args.graph_name}, {args.fan_out}')
    
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Average Node Degree')
    plt.xlabel('Vertices Hotness')
    plt.title('Average degree of vertices of different hotness')
    plt.legend()
    save_path = os.path.expanduser(f'../src/result/{args.graph_name}_hotness_degree_{args.fan_out}.pdf')
    plt.savefig(save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--fan_out", type=str, default="10,25")
    args = parser.parse_args()

    print(args)
    main(args)
