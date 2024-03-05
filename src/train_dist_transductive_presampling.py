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

def presampling_accross_machine(dataloader, num_nodes, num_epochs=1, group=None):
    presampling_heat_am = th.zeros((num_nodes, ), dtype=th.int64)
    for epoch in range(num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            presampling_heat_am[input_nodes] += 1

    if th.distributed.get_backend() == "nccl":
        presampling_heat_am = presampling_heat_am.cuda()
        th.distributed.all_reduce(presampling_heat_am,
                                     th.distributed.ReduceOp.SUM, group=group)
        presampling_heat_am = (presampling_heat_am > 0).to(th.int64)
        th.distributed.all_reduce(presampling_heat_am,
                                     th.distributed.ReduceOp.SUM)
        presampling_heat_am = presampling_heat_am / 8
        presampling_heat_am = presampling_heat_am.cpu()
    else:
        th.distributed.all_reduce(presampling_heat_am,
                                     th.distributed.ReduceOp.SUM, group=group)
        presampling_heat_am = (presampling_heat_am > 0).to(th.int64)
        th.distributed.all_reduce(presampling_heat_am,
                                     th.distributed.ReduceOp.SUM)
        presampling_heat_am = presampling_heat_am / 8

    sorted = th.sort(presampling_heat_am)[0]
    presampling_hotness = presampling(dataloader, num_nodes)
    if th.distributed.get_rank() == 0:
        print(f"total: {len(presampling_heat_am)}")
        for i in range(5):
            print(f'presampling_accross_machine = {i}: {len(presampling_heat_am[presampling_heat_am == i])}, rate: {len(presampling_heat_am[presampling_heat_am == i]) / len(presampling_heat_am)}')
        quantiles = th.quantile(sorted.float(), th.linspace(0, 1, steps=11), 0)
        print('------')
        for i in range(len(quantiles) - 1):
            start_value = quantiles[i].item()
            end_value = quantiles[i + 1].item()
            print(f'Batch {step}, Bin {i + 1}: {start_value} - {end_value}')
        print('------')

        import matplotlib.pyplot as plt   
        import os  
        for i in range(1, 5):                
            plt.clf()
            print(f"{i}, {len(presampling_hotness[presampling_heat_am == i])}")            
            presampling_temp = presampling_hotness[presampling_heat_am == i]
            sorted = th.sort(presampling_temp)[0]
            sorted = sorted[sorted != 0]
            cumulative_prob = np.arange(len(sorted)) / float(len(sorted))
            plt.plot(sorted, cumulative_prob, label=f'{args.graph_name}')
            plt.xlabel('Access frequency')
            plt.title(f'Access frequency of vertices accessed across {i} machine(s)')
            plt.legend()
            save_path = os.path.expanduser(f'./result/{args.graph_name}_heat_am_{i}.png')
            plt.savefig(save_path)
        return presampling_heat_am

    
def run(args, device, data, group=None):
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g = data
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    if args.presampling:      
        hot_rate = args.hot_rate
        hotness_list = presampling(dataloader, g.num_nodes())
    if args.presampling_am:
        hotness_list = presampling_accross_machine(dataloader, g.num_nodes(), group=group)
    
    emb_layer = DistEmb(
        g.num_nodes(),
        args.num_hidden,
        dgl_sparse_emb=args.dgl_sparse,
        dev_id=device,
    )
    model = DistSAGE(
        args.num_hidden,
        256,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[dev_id], output_device=dev_id
            )
            if not args.dgl_sparse:
                emb_layer = th.nn.parallel.DistributedDataParallel(emb_layer)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.dgl_sparse:
        if args.presampling:
            emb_optimizer = SparseAdam_with_mask(
                [emb_layer.sparse_emb], lr=args.sparse_lr
            )
        else:
            emb_optimizer = dgl.distributed.optim.SparseAdam(
                [emb_layer.sparse_emb], lr=args.sparse_lr
            )
        print("optimize DGL sparse embedding:", emb_layer.sparse_emb)
    elif args.standalone:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print("optimize Pytorch sparse embedding:", emb_layer.sparse_emb)
    else:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.module.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print(
            "optimize Pytorch sparse embedding:",
            emb_layer.module.sparse_emb
        )

    iter_tput = []
    epoch = 0
    log_path = f'./result/graphsage_{args.graph_name}_{args.hot_rate}_from_{"top" if args.from_top else "bottom"}.log'
    res_path = f'./result/graphsage_{args.graph_name}_{args.hot_rate}_from_{"top" if args.from_top else "bottom"}.txt'
    epoch_list = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        emb_update_time = 0
        num_seeds = 0
        num_inputs = 0

        with model.join():
            step_time = []
            tic = time.time()
            tic_step = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # calculate quantiles
                if args.presampling:      
                    hotness_list_temp = hotness_list[input_nodes]
                    sorted = th.sort(hotness_list_temp)[0]
                    if args.from_top:
                        quantiles = th.quantile(sorted.float(), 1-hot_rate, 0)
                    else:
                        quantiles = th.quantile(sorted.float(), hot_rate, 0)
                        
                th.cuda.synchronize()
                sample_time += time.time() - tic_step
                
                load_begin = time.time()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                blocks = [block.to(device) for block in blocks]
                batch_labels = g.ndata["labels"][seeds].long().to(device)
                batch_inputs = emb_layer(input_nodes)
                th.cuda.synchronize()
                load_time += time.time() - load_begin

                forward_start = time.time()
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                th.cuda.synchronize()
                forward_time += time.time() - forward_start

                backward_begin = time.time()
                emb_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                th.cuda.synchronize()
                backward_time += time.time() - backward_begin

                update_start = time.time()
                optimizer.step()
                th.cuda.synchronize()
                update_time += time.time() - update_start

                emb_update_start = time.time()
                if args.presampling:
                    #emb_optimizer.step(hotness_list.to(device), args.hot_rate)
                    emb_optimizer.step(hotness_list.to(device), quantiles.to(device), from_top=args.from_top)
                else:
                    emb_optimizer.step()
                th.cuda.synchronize()
                emb_update_time += time.time() - emb_update_start

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    if th.distributed.get_rank() == 0:
                        print(
                            "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                            "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                            "{:.1f} MB | time {:.3f} s".format(
                                g.rank(),
                                epoch,
                                step,
                                loss.item(),
                                acc.item(),
                                np.mean(iter_tput[3:]),
                                gpu_mem_alloc,
                                np.sum(step_time[-args.log_every:]),
                            )
                        )
                        with open(log_path, 'a') as f:
                            f.write("Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB | time {:.3f} s\n"
                                .format(
                                    g.rank(),
                                    epoch,
                                    step,
                                    loss.item(),
                                    acc.item(),
                                    np.mean(iter_tput[3:]),
                                    gpu_mem_alloc,
                                    np.sum(step_time[-args.log_every:])
                                ))
                tic_step = time.time()

        toc = time.time()
        epoch_list.append(toc - tic)
        epoch += 1

        if th.distributed.get_rank() == 0:
            timetable = ("=====================\n"
                        "Part {}, Epoch Time(s): {:.4f}\n"
                        "Sampling Time(s): {:.4f}\n"
                        "Loading Time(s): {:.4f}\n"
                        "Forward Time(s): {:.4f}\n"
                        "Backward Time(s): {:.4f}\n"
                        "Update Time(s): {:.4f}\n"
                        "Emb Update Time(s): {:.4f}\n"
                        "#seeds: {}\n"
                        "#inputs: {}\n"
                        "=====================\n".format(
                        th.distributed.get_rank(),
                        toc - tic,
                        sample_time,
                        load_time,
                        forward_time,
                        backward_time,
                        update_time,
                        emb_update_time,
                        num_seeds,
                        num_inputs,
            ))
            print(timetable)
            with open(log_path, 'a') as f:
                    f.write(timetable)
                

        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            val_acc, test_acc = evaluate(
                args.standalone,
                model,
                emb_layer,
                g,
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )
        
            val_acc_avg = th.tensor([val_acc], device="cuda")
            test_acc_avg = th.tensor([test_acc], device="cuda")
            th.distributed.all_reduce(val_acc_avg)
            th.distributed.all_reduce(test_acc_avg)
            val_acc = val_acc_avg.item() / th.distributed.get_world_size()
            test_acc = test_acc_avg.item() / th.distributed.get_world_size()

            if th.distributed.get_rank() == 0:
                print(th.distributed.get_world_size())
                print(
                    "Part {}, Val Acc avg {:.4f}, Test Acc avg {:.4f}, time: {:.4f}".format
                    (
                        g.rank(), val_acc, test_acc, time.time() - start
                    )
                )
                with open(log_path, 'a') as f:
                    f.write("Part {}, Epoch {}, Val Acc avg {:.4f}, Test Acc avg {:.4f}\n".format
                    (
                        g.rank(), epoch, val_acc, test_acc
                    ))
                with open(res_path, 'a') as f:
                    f.write("Epoch {}, Val Acc avg {:.4f}, Test Acc avg {:.4f}\n".format
                    (
                        epoch, val_acc, test_acc
                    ))
    avg_epoch = np.mean(epoch_list[2:])
    throughput = len(train_nid) / avg_epoch
    if th.distributed.get_rank() == 0:
        with open(res_path, 'a') as f:
            f.write("Avg_epoch: {}, throughput: {}, len(train_nid): {}\n".format
            (
                avg_epoch, throughput, len(train_nid)
            ))




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
