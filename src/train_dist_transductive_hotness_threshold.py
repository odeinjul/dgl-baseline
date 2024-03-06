import argparse
import time
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt   

def calculate_throughput(hotness_list, threshold, slice=100, nm=4, nl=8, nt=32):
    sum = 0
    hot, cold = 0, 0
    for i in range(slice):
        start_value = i / slice
        end_value = (i + 1) / slice
        indices = th.where((hotness_list >= start_value) & (hotness_list < end_value))
        hotness = (start_value + end_value) / 2
        count  = len(indices)
        if hotness > threshold: # hot
            single = hotness * nt * ((nm - 1) / nm) + (1 - pow(1-hotness, nl)) * (nm - 1)
            sum += single * count 
            hot += single * count 

        else: # cold
            single = hotness * (1 / 100 if hotness < 1/100 else hotness) * nt * ((nm - 1) / nm) * 2
            sum += single * count
            cold += single * count
    
    return sum, hot, cold
    

def main(args):
    print("Loading training data")
    save_fn =  os.path.expanduser(f'../src/result/{args.graph_name}_degree_list.pt')
    degree_list = th.load(save_fn)
    print("Result loaded from {}".format(save_fn))
    save_fn = os.path.expanduser(f'../src/result/{args.graph_name}_presampling_heat_{args.fan_out}.pt')
    hotness_list = th.load(save_fn)
    print("Result loaded from {}".format(save_fn))
        
    hotness_list = hotness_list[hotness_list != 0]


    throughput = th.zeros((100, ), dtype=th.float32)
    hot_th = th.zeros((100, ), dtype=th.float32)
    cold_th = th.zeros((100, ), dtype=th.float32)
    x_axis = th.linspace(0, 1, 100)
    for i in range(100):
        throughput[i], hot_th[i], cold_th[i] = calculate_throughput(hotness_list, i / 100)
    
    plt.plot(x_axis, throughput, label=f'Total Communication Volume')
    # dashed line
    plt.plot(x_axis, hot_th, label=f'Hot Communication Volume', color='r', linestyle='dashed')
    plt.plot(x_axis, cold_th, label=f'Cold Communication Volume', color='g', linestyle='dashed')
    
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Communication Volume')
    plt.xlabel('Threshold')
    plt.title(f'Communication Volume vs Threshold in {args.graph_name}, fan_out={args.fan_out}')
    plt.legend()
    save_path = os.path.expanduser(f'../src/result/{args.graph_name}_throughput_vs_threshold_{args.fan_out}.pdf')
    plt.savefig(save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--fan_out", type=str, default="10,25")
    args = parser.parse_args()

    print(args)
    main(args)
