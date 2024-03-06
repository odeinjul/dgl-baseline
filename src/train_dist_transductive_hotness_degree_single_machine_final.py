import argparse
import time
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt   

def tmp(graph_name, fan_out):
    print("Loading training data")
    save_fn = os.path.expanduser(f'../src/result/{graph_name}_presampling_heat_{fan_out}.pt')
    hotness_list = th.load(save_fn)
    print("Result loaded from {}".format(save_fn))

    hotness_list = hotness_list[hotness_list != 0]

    # return a cdf of hotness
    x_axis = th.linspace(0, 101, 100)
    hotness_count = th.zeros((101, ), dtype=th.float32)
    for i in range(100):
        start_value = i / 100
        indices = hotness_list[hotness_list < start_value]
        hotness_count[i] = len(indices) / len(hotness_list)
    hotness_count[100] = 1
        

    return hotness_count


def main(args):
    x_axis = th.linspace(0, 1, 101)
    """
    hotness_count = tmp(args.graph_name, "10,15")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 15,10', color="#819fa6")
    
    hotness_count = tmp(args.graph_name, "10,25")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 25,10', color="#c18076")
    
    #hotness_count = tmp(args.graph_name, "5,10,15")
    #plt.plot(x_axis, hotness_degree, label=f'fan_out = {5,10,15}')
    
    hotness_count = tmp(args.graph_name, "5,5,10,15")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 15,10,5,5', color="#3d4a55")
    
    hotness_count = tmp(args.graph_name, "10,10,10,10")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 10,10,10,10', color="#d1b5ab")
    
    hotness_count = tmp(args.graph_name, "5,5,5,5,10,15")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 15,10,5,5,5,5', color='c')
    
    hotness_count = tmp(args.graph_name, "10,10,10,10,10,10")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 10,10,10,10,10,10', color="#1A9050")
    """
    hotness_count = tmp(args.graph_name, "10,15")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 15,10', color="#819fa6")
    
    hotness_count = tmp(args.graph_name, "10,25")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 25,10', color="#c18076")
    
    #hotness_count = tmp(args.graph_name, "5,10,15")
    #plt.plot(x_axis, hotness_degree, label=f'fan_out = {5,10,15}')
    
    hotness_count = tmp(args.graph_name, "5,5,10,15")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 15,10,5,5', color="#3d4a55")
    
    hotness_count = tmp(args.graph_name, "5,5,5,5")
    plt.plot(x_axis, hotness_count, label=f'fan_out = 5,5,5,5', color="#d1b5ab")
    
    
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Access frequency')
    plt.title('Access frequency of all vertices')
    plt.legend()
    save_path = os.path.expanduser(f'../src/result/{args.graph_name}_presampling_final.pdf')
    plt.savefig(save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    args = parser.parse_args()

    print(args)
    main(args)
