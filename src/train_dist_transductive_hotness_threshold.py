import argparse
import time
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt   

def calculate_throughput(hotness_list, nm=4, nl=8, nt=32):
    hotness_tensor = th.tensor(hotness_list, dtype=th.float32)
    hot = th.zeros_like(hotness_tensor)
    cold = th.zeros_like(hotness_tensor)
    
    tmp_cliped = 1 / hotness_tensor
    tmp_cliped[tmp_cliped > 16] = 16
    
    live_cliped = tmp_cliped / (nl * hotness_tensor)
    #live_cliped[live_cliped > 100] = 100
    
    hot = hotness_tensor * nt * ((nm - 1) / nm) + (1 - th.pow(1 - hotness_tensor, nl)) * (nm - 1)
    cold = (hotness_tensor * nt  / live_cliped) * ((nm - 1) / nm) * 2

    return hot, cold

def calculate_throughput_single(hotness_list, nm=1, nl=8, nt=8):
    hotness_tensor = th.tensor(hotness_list, dtype=th.float32)
    hot = th.zeros_like(hotness_tensor)
    cold = th.zeros_like(hotness_tensor)
    
    hotness_tensor_cliped = hotness_tensor
    hotness_tensor_cliped[hotness_tensor_cliped < 1/16] = 1/16
    
    hot = hotness_tensor * nt * 2
    cold = hotness_tensor * hotness_tensor_cliped * nt * 2

    return hot, cold
            
def main(args):
    print("Loading training data")
    
    save_fn = os.path.expanduser(f'../src/result/{args.graph_name}_presampling_heat_{args.fan_out}.pt')
    hotness_list = th.load(save_fn)
    print("Result loaded from {}".format(save_fn))
        
    hotness_list = hotness_list[hotness_list != 0]

    throughput = th.zeros((100, ), dtype=th.float32)
    cold_throughput = th.zeros((100, ), dtype=th.float32)
    hot_throughput = th.zeros((100, ), dtype=th.float32)
    
    if args.single:
        hot_vol, cold_vol = calculate_throughput_single(hotness_list)
    else:
        hot_vol, cold_vol = calculate_throughput(hotness_list)
    
    x_axis = th.linspace(0, 1, 100)
    for i in tqdm.tqdm(range(100)):
        threshold = x_axis[i]
        hot = th.sum(hot_vol[hotness_list > threshold])
        cold = th.sum(cold_vol[hotness_list <= threshold])
        cold_throughput[i] = cold
        hot_throughput[i] = hot
        throughput[i] = hot + cold
    
    plt.plot(x_axis, throughput, label=f'Total Communication Volume')
    # dashed line
    plt.plot(x_axis, hot_throughput, label=f'Hot Communication Volume', color='r', linestyle='dashed')
    plt.plot(x_axis, cold_throughput, label=f'Cold Communication Volume', color='g', linestyle='dashed')
    
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Communication Volume')
    plt.xlabel(f'Threshold nm={1 if args.single else 4}, nl={8}, nt={8 if args.single else 32}')
    plt.title(f'Communication Volume vs Threshold in {args.graph_name}, fan_out={args.fan_out}')
    plt.legend()
    save_path = os.path.expanduser(f'../src/result/{args.graph_name}_throughput_vs_threshold_{args.fan_out}{"_single" if args.single else ""}.pdf')
    plt.savefig(save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--single", action="store_true", help="single threshold")
    args = parser.parse_args()

    print(args)
    main(args)
