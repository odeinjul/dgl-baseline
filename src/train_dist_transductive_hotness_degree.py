import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.distributed import DistEmbedding
from optim import SparseAdam_with_mask
from train_dist import DistSAGE, compute_acc        


def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    arr.uniform_(-1, 1)
    return arr


class DistEmb(nn.Module):
    def __init__(
            self, num_nodes, emb_size, dgl_sparse_emb=False, dev_id="cpu"
    ):
        super().__init__()
        self.dev_id = dev_id
        self.emb_size = emb_size
        self.dgl_sparse_emb = dgl_sparse_emb
        if dgl_sparse_emb:
            self.sparse_emb = DistEmbedding(
                num_nodes, emb_size, name="sage", init_func=initializer
            )
        else:
            self.sparse_emb = th.nn.Embedding(num_nodes, emb_size, sparse=True)
            nn.init.uniform_(self.sparse_emb.weight, -1.0, 1.0)

    def forward(self, idx):
        # embeddings are stored in cpu
        idx = idx.cpu()
        if self.dgl_sparse_emb:
            return self.sparse_emb(idx, device=self.dev_id)
        else:
            return self.sparse_emb(idx).to(self.dev_id)


def load_embs(standalone, emb_layer, g):
    nodes = dgl.distributed.node_split(
        np.arange(g.num_nodes()), g.get_partition_book(), force_even=True
    )
    x = dgl.distributed.DistTensor(
        (
            g.num_nodes(),
            emb_layer.module.emb_size
            if isinstance(emb_layer, th.nn.parallel.DistributedDataParallel)
            else emb_layer.emb_size,
        ),
        th.float32,
        "eval_embs",
        persistent=True,
    )
    num_nodes = nodes.shape[0]
    for i in range((num_nodes + 1023) // 1024):
        idx = nodes[
            i * 1024: (i + 1) * 1024
            if (i + 1) * 1024 < num_nodes
            else num_nodes
        ]
        embeds = emb_layer(idx).cpu()
        x[idx] = embeds

    if not standalone:
        g.barrier()

    return x


def evaluate(
    standalone,
    model,
    emb_layer,
    g,
    labels,
    val_nid,
    test_nid,
    batch_size,
    device,
):
    if not standalone:
        model = model.module
    model.eval()
    emb_layer.eval()
    with th.no_grad():
        inputs = load_embs(standalone, emb_layer, g)
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    emb_layer.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )
    
def presampling(dataloader, num_nodes, num_epochs=1):
    presampling_heat = th.zeros((num_nodes, ), dtype=th.int64)
    sampling_times = 0
    for epoch in range(num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            presampling_heat[input_nodes] += 1
            sampling_times += 1

    if th.distributed.get_backend() == "nccl":
        presampling_heat = presampling_heat.cuda()
        th.distributed.all_reduce(presampling_heat,
                                     th.distributed.ReduceOp.SUM)
        sampling_times = th.tensor([sampling_times], device="cuda")
        th.distributed.all_reduce(sampling_times,
                                     th.distributed.ReduceOp.SUM)
        presampling_heat = presampling_heat / sampling_times
        presampling_heat = presampling_heat.cpu()
    else:
        th.distributed.all_reduce(presampling_heat,
                                     th.distributed.ReduceOp.SUM)
        sampling_times = th.tensor([sampling_times])
        th.distributed.all_reduce(sampling_times,
                                     th.distributed.ReduceOp.SUM)
        presampling_heat = presampling_heat / sampling_times
    import matplotlib.pyplot as plt   
    import os                              
    presampling_temp = presampling_heat[blocks[0].ndata[dgl.NID]["_N"][blocks[0].srcnodes()]]
    
    # iterate each epoch and get presampling heat list
    temp_list = []
    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        temp_list.append(presampling_heat[input_nodes])
    result_tensor = th.cat(temp_list, dim=0)

    print(len(result_tensor))
    sorted = th.sort(result_tensor)[0]
    sorted = sorted[sorted != 0]
    cumulative_prob = np.arange(len(sorted)) / float(len(sorted))
    plt.plot(sorted, cumulative_prob, label=f'{args.graph_name}, {args.fan_out}')
    plt.xlabel(f'Access frequency')
    plt.title('Access frequency of all vertices')
    plt.legend()
    save_path = os.path.expanduser(f'./result/{args.graph_name}_presampling_{args.fan_out}.pdf')
    
    if th.distributed.get_rank() == 0:
        plt.savefig(save_path)

    if th.distributed.get_rank() == 0:
        print(
        "Presampling done, max: {:.3f} min: {:.3f} avg: {:.3f}"
        .format(
            th.max(presampling_heat).item(),
            th.min(presampling_heat).item(),
            th.mean(presampling_heat).item()))
        save_fn =  os.path.expanduser(f'./result/{args.graph_name}_presampling_heat_{args.fan_out}.pt')
        th.save(presampling_heat, save_fn)
        print("Result saved to {}".format(save_fn))
        
    return presampling_heat

 
def run(args, device, data, group=None):
    import os
    train_nid, val_nid, test_nid, n_classes, g = data
    degree_list = g.in_degrees()
    if th.distributed.get_rank() == 0:
        save_fn =  os.path.expanduser(f'./result/{args.graph_name}_degree_list.pt')
        th.save(degree_list, save_fn)
        print("Result saved to {}".format(save_fn))
    """
    if th.distributed.get_rank() == 0:
        print("Loading training data")
        save_fn =  os.path.expanduser(f'./result/{args.graph_name}_degree_list.pt')
        degree_list = th.load(save_fn)
        print("Result loaded from {}".format(save_fn))
        save_fn = os.path.expanduser(f'./result/{args.graph_name}_presampling_heat_{args.fan_out}.pt')
        hotness_list = th.load(save_fn)
        print("Result loaded from {}".format(save_fn))
        
        # delete hotness = 0
        degree_list = degree_list[hotness_list != 0]
        hotness_list = hotness_list[hotness_list != 0]
        print (len(degree_list), len(hotness_list))

        sorted_degree, sorted_indices = th.sort(degree_list)
        sorted_hotness = degree_list[sorted_indices]
       
        avg_hotness = th.zeros((20, ), dtype=th.float32)
        degree_count = th.zeros((20, ), dtype=th.float32)
        
        # divide 20 bins from degree.min to degree.max
        step = (th.max(sorted_degree) - th.min(sorted_degree)) / 20
        for i in range(20):
            start_value = th.min(sorted_degree) + i * step
            end_value = start_value + step
            indices = th.where((sorted_degree >= start_value) & (sorted_degree < end_value))
            avg_hotness[i] = th.mean(th.Tensor.float(sorted_hotness[indices]))
            degree_count[i] = (start_value + end_value) / 2
            

       
        # make a plot, the x-axis is hotness, the y-axis is average degree
        import matplotlib.pyplot as plt
        plt.plot(degree_count, avg_hotness, label=f'{args.graph_name}, {args.fan_out}')
        # set plot x axis from 0, 0.1, 0.2, ..., 0.9, 1.0
        plt.xticks(np.arange(th.min(sorted_degree), th.max(sorted_degree), step*2))
        plt.ylabel('Average Hotness')
        plt.xlabel('Vertices Degree')
        plt.title('Average hotness of vertices of different hotness')
        plt.legend()
        save_path = os.path.expanduser(f'./result/{args.graph_name}_degree_hotness_{args.fan_out}.pdf')
        plt.savefig(save_path)
    """
    
def main(args):
    dgl.distributed.initialize(args.ip_config)
    if args.presampling_am:
        group_ranks_0 = range(8)
        group_ranks_1 = range(8, 16)
        group_ranks_2 = range(16, 24)
        group_ranks_3 = range(24, 32)
    if not args.standalone:
        th.distributed.init_process_group(backend="gloo")
        if args.presampling_am:
            group_0 = th.distributed.new_group(ranks=group_ranks_0)
            group_1 = th.distributed.new_group(ranks=group_ranks_1)
            group_2 = th.distributed.new_group(ranks=group_ranks_2)
            group_3 = th.distributed.new_group(ranks=group_ranks_3)
            if th.distributed.get_rank() in group_ranks_0:
                group = group_0
            elif th.distributed.get_rank() in group_ranks_1:
                group = group_1
            elif th.distributed.get_rank() in group_ranks_2:
                group = group_2
            elif th.distributed.get_rank() in group_ranks_3:
                group = group_3
    g = dgl.distributed.DistGraph(
            args.graph_name,
            part_config=args.part_config
        )
    print("rank:", g.rank())

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(
        g.ndata["train_mask"], pb, force_even=True
    )
    val_nid = dgl.distributed.node_split(
        g.ndata["val_mask"], pb, force_even=True
    )
    test_nid = dgl.distributed.node_split(
        g.ndata["test_mask"], pb, force_even=True
    )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
        "(local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        th.cuda.set_device(device)
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    data = train_nid, val_nid, test_nid, n_classes, g
    if args.presampling_am:
        run(args, device, data, group=group)
    else:
        run(args, device, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument("--n_classes", type=int, help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--dgl_sparse",
        action="store_true",
        help="Whether to use DGL sparse embedding",
    )
    parser.add_argument(
        "--sparse_lr", type=float, default=1e-2, help="sparse lr rate"
    )
    parser.add_argument(
        "--presampling", action="store_true"
    )
    parser.add_argument(
        "--presampling_am", action="store_true"
    )
    parser.add_argument(
        "--hot_rate", type=float, default=0.5
    )
    parser.add_argument(
        "--from_top", action="store_true"
    )
    args = parser.parse_args()

    print(args)
    main(args)
