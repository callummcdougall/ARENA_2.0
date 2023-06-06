import collections
import inspect
import threading

import torch.distributed

from torch.distributed import ReduceOp


class Dist:
    def __init__(self, world_size=3):
        self.distributed = torch.distributed
        self.world_size = world_size
        self.source_dest_tensors = {i: {j: {'to_read': [], 'to_write': []} for j in range(self.world_size)} for i in range(self.world_size)}
        self.source_dest_tensors_lock = threading.Lock()
        self.barrier_lock = threading.Barrier(self.world_size)
        self.proc_barrier = threading.Barrier(self.world_size)

        self.reads = collections.defaultdict(list)
        self.writes = collections.defaultdict(list)

    def with_rank(self, rank):
        return DistWithRank(rank, self.source_dest_tensors, self.source_dest_tensors_lock, self.barrier_lock, self.proc_barrier, self.reads, self.writes)

    def get_world_size(self):
        return self.world_size

class DistWithRank(Dist):
    def __init__(self, rank, source_dest_tensors, source_dest_tensors_lock, barrier, proc_barrier, reads, writes):
        super().__init__()
        self.rank = rank
        self.source_dest_tensors = source_dest_tensors
        self.source_dest_tensors_lock = source_dest_tensors_lock
        self.barrier_lock = barrier
        self.proc_barrier = proc_barrier

        self.reads = reads
        self.writes = writes

    def __enter__(self):
        self.replace_torch()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_torch()
        for i in range(self.world_size):
            for j in range(self.world_size):
                # if len(self.source_dest_tensors[i][j]['to_read']) > 0 and len(self.source_dest_tensors[j][i]['to_write']) > 0:  # we don't worry about this case because of the lock
                assert len(self.source_dest_tensors[i][j]['to_read']) == 0, f"Incomplete sends: {self.source_dest_tensors[i][j]['to_read']}"
                assert len(self.source_dest_tensors[i][j]['to_write']) == 0, f"Incomplete recvs: {self.source_dest_tensors[i][j]['to_write']}"

    def write_to(self, tensor, rank):
        self.writes[self.rank].append(rank)
        with self.source_dest_tensors_lock:
            if len(self.source_dest_tensors[rank][self.rank]['to_read']) > 0:
                assert self.source_dest_tensors[rank][self.rank]['to_read'][0].shape == tensor.shape, f"Shape mismatch: {self.source_dest_tensors[rank][self.rank]['to_read'][0].shape} != {tensor.shape}"
                assert self.source_dest_tensors[rank][self.rank]['to_read'][0].dtype == tensor.dtype, f"Dtype mismatch: {self.source_dest_tensors[rank][self.rank]['to_read'][0].dtype} != {tensor.dtype}"
                tensor[:] = self.source_dest_tensors[rank][self.rank]['to_read'].pop(0)[:]
                print(f"Rank {self.rank} read tensor {tensor} from rank {rank}")
            else:
                self.source_dest_tensors[self.rank][rank]['to_write'].append(tensor)
    def read_from(self, tensor, rank):
        self.reads[self.rank].append(rank)
        with self.source_dest_tensors_lock:
            if len(self.source_dest_tensors[rank][self.rank]['to_write']) > 0:
                assert self.source_dest_tensors[rank][self.rank]['to_write'][0].shape == tensor.shape, f"Shape mismatch: {self.source_dest_tensors[rank][self.rank]['to_write'][0].shape} != {tensor.shape}"
                assert self.source_dest_tensors[rank][self.rank]['to_write'][0].dtype == tensor.dtype, f"Dtype mismatch: {self.source_dest_tensors[rank][self.rank]['to_write'][0].dtype} != {tensor.dtype}"
                tensor[:] = self.source_dest_tensors[rank][self.rank]['to_write'].pop(0)[:]
            else:
                self.source_dest_tensors[rank][self.rank]['to_read'].append(tensor)


    def is_available(self):
        return True

    def init_process_group(self, *args, **kwargs):
        print("using fake dist class - skipping init_process_group. remember to get your ranks with get_rank(), the rank you passed was ignored!")

    def is_initialized(self):
        return True

    def is_mpi_available(self):
        return False

    def is_nccl_available(self):
        return False

    def is_gloo_available(self):
        return False

    def is_torchelastic_launched(self):
        return False

    def get_backend(self, group=None):
        if group is not None:
            raise NotImplementedError("groups are not implemented")
        return 'fake'

    def get_rank(self, group=None):
        if group is not None:
            raise NotImplementedError("groups are not implemented")
        return self.rank

    def get_world_size(self, group=None):
        if group is not None:
            raise NotImplementedError("groups are not implemented")
        return self.world_size

    def new_group(self, *args, **kwargs):
        raise NotImplementedError("groups are not implemented")

    def get_group_rank(self, *args, **kwargs):
        raise NotImplementedError("groups are not implemented")

    def get_global_rank(self, *args, **kwargs):
        raise NotImplementedError("groups are not implemented")

    def get_process_group_ranks(self, *args, **kwargs):
        raise NotImplementedError("groups are not implemented")

    def send(self, tensor, rank, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")
        self.write_to(tensor, rank)

    def recv(self, tensor, rank, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")
        self.read_from(tensor, rank)

    def broadcast(self, tensor, rank):
        if self.rank == rank:
            for i in range(self.world_size):
                if i != self.rank:
                    self.write_to(tensor, i)
        else:
            self.read_from(tensor, rank)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")

        if self.rank == dst:
            res = tensor.clone()
        if op == ReduceOp.SUM:
            if self.rank == dst:
                for i in range(self.world_size):
                    if i != self.rank:
                        self.read_from(res, i)
                        tensor += res
            else:
                self.write_to(tensor, dst)
        elif op == ReduceOp.PRODUCT:
            if self.rank == dst:
                for i in range(self.world_size):
                    if i != self.rank:
                        self.read_from(res, i)
                        tensor *= res
        elif op == ReduceOp.MAX:
            if self.rank == dst:
                for i in range(self.world_size):
                    if i != self.rank:
                        self.read_from(res, i)
                        tensor = torch.max(tensor, res)
        elif op == ReduceOp.MIN:
            if self.rank == dst:
                for i in range(self.world_size):
                    if i != self.rank:
                        self.read_from(res, i)
                        tensor = torch.min(tensor, res)
        else:
            raise NotImplementedError("only sum, product, max and min are implemented")

    def all_reduce(self, tensor, op=ReduceOp.SUM, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")

        self.reduce(tensor, 0, op=op, group=group)
        self.broadcast(tensor, 0)

    def all_gather(self, tensor_list, tensor, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")

        for i in range(self.world_size):
            if i != self.rank:
                self.write_to(tensor, i)
        for i in range(self.world_size):
            if i != self.rank:
                self.read_from(tensor_list[i], i)
        tensor_list[self.rank] = tensor

    def gather(self, tensor, gather_list, dst, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")

        if self.rank == dst:
            for i in range(self.world_size):
                if i != self.rank:
                    self.read_from(gather_list[i], i)
            gather_list[self.rank] = tensor
        else:
            self.write_to(tensor, dst)

    def scatter(self, scatter_list, tensor, src, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")

        if self.rank == src:
            for i in range(self.world_size):
                if i != self.rank:
                    self.write_to(scatter_list[i], i)
            scatter_list[self.rank] = tensor
        else:
            self.read_from(tensor, src)

    def barrier(self, group=None):
        if group not in (None, torch.distributed.group.WORLD):
            raise NotImplementedError("groups are not implemented")

        self.proc_barrier.wait()
    def replace_torch(self):
        self._is_available = torch.distributed.is_available
        torch.distributed.is_available = self._is_available
        self._init_process_group = torch.distributed.init_process_group
        torch.distributed.init_process_group = self.init_process_group
        self._is_initialized = torch.distributed.is_initialized
        torch.distributed.is_initialized = self.is_initialized
        self._is_mpi_available = torch.distributed.is_mpi_available
        torch.distributed.is_mpi_available = self.is_mpi_available
        self._is_nccl_available = torch.distributed.is_nccl_available
        torch.distributed.is_nccl_available = self.is_nccl_available
        self._is_gloo_available = torch.distributed.is_gloo_available
        torch.distributed.is_gloo_available = self.is_gloo_available
        self._is_torchelastic_launched = torch.distributed.is_torchelastic_launched
        torch.distributed.is_torchelastic_launched = self.is_torchelastic_launched
        self._get_backend = torch.distributed.get_backend
        torch.distributed.get_backend = self.get_backend
        self._get_rank = torch.distributed.get_rank
        torch.distributed.get_rank = self.get_rank
        self._get_world_size = torch.distributed.get_world_size
        torch.distributed.get_world_size = self.get_world_size
        self._new_group = torch.distributed.new_group
        torch.distributed.new_group = self.new_group
        self._get_group_rank = torch.distributed.get_group_rank
        torch.distributed.get_group_rank = self.get_group_rank
        self._get_world_size = torch.distributed.get_world_size
        torch.distributed.get_world_size = self.get_world_size
        self._new_group = torch.distributed.new_group
        torch.distributed.new_group = self.new_group
        self._get_group_rank = torch.distributed.get_group_rank
        torch.distributed.get_group_rank = self.get_group_rank
        self._get_global_rank = torch.distributed.get_global_rank
        torch.distributed.get_global_rank = self.get_global_rank
        self._get_process_group_ranks = torch.distributed.get_process_group_ranks
        torch.distributed.get_process_group_ranks = self._get_process_group_ranks
        self._send = torch.distributed.send
        torch.distributed.send = self.send
        self._recv = torch.distributed.recv
        torch.distributed.recv = self.recv
        self._broadcast = torch.distributed.broadcast
        torch.distributed.broadcast = self.broadcast
        self._reduce = torch.distributed.reduce
        torch.distributed.reduce = self.reduce
        self._all_reduce = torch.distributed.all_reduce
        torch.distributed.all_reduce = self.all_reduce
        self._all_gather = torch.distributed.all_gather
        torch.distributed.all_gather = self.all_gather
        self._gather = torch.distributed.gather
        torch.distributed.gather = self.gather
        self._scatter = torch.distributed.scatter
        torch.distributed.scatter = self.scatter
        self._barrier = torch.distributed.barrier
        torch.distributed.barrier = self.barrier

    def restore_torch(self):
        self.barrier_lock.wait()
        torch.distributed.is_available = self._is_available
        torch.distributed.init_process_group = self._init_process_group
        torch.distributed.is_initialized = self._is_initialized
        torch.distributed.is_mpi_available = self._is_mpi_available
        torch.distributed.is_nccl_available = self._is_nccl_available
        torch.distributed.is_gloo_available = self._is_gloo_available
        torch.distributed.is_torchelastic_launched = self._is_torchelastic_launched
        torch.distributed.get_backend = self._get_backend
        torch.distributed.get_rank = self._get_rank
        torch.distributed.get_world_size = self._get_world_size
        torch.distributed.new_group = self._new_group
        torch.distributed.get_group_rank = self._get_group_rank
        torch.distributed.get_global_rank = self._get_global_rank
        torch.distributed.get_process_group_ranks = self._get_process_group_ranks
        torch.distributed.send = self._send
        torch.distributed.recv = self._recv
        torch.distributed.broadcast = self._broadcast
        torch.distributed.reduce = self._reduce
        torch.distributed.all_reduce = self._all_reduce
        torch.distributed.all_gather = self._all_gather
        torch.distributed.gather = self._gather
        torch.distributed.scatter = self._scatter
        torch.distributed.barrier = self._barrier

def test_scaffold(test_func):
    fake_dist = Dist()
    threads = []
    def run_test(rank):
        with fake_dist.with_rank(rank):
            test_func()
    for rank in range(fake_dist.world_size):
        t = threading.Thread(target=run_test, args=(rank,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    return fake_dist


def test_broadcast(broadcast_impl):
    fake_dist = test_scaffold(broadcast_impl)
    assert all(len(fake_dist.reads[i]) == 1 for i in range(fake_dist.world_size) if i != 0)
    assert all(len(fake_dist.writes[i]) == fake_dist.world_size - 1 for i in range(fake_dist.world_size) if i == 0)
