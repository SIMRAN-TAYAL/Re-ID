# ========== 3. RandomIdentitySampler ==========

import random
from torch.utils.data import Sampler



class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.index_dic = {}

        # Build dictionary: pid -> list of indices
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic.setdefault(pid, []).append(index)

        self.pids = list(self.index_dic.keys())
        self.num_pids_per_batch = batch_size // num_instances

    def __iter__(self):
        # Randomize the order of PIDs
        random.shuffle(self.pids)

        # For each small chunk of PIDs, create a batch
        for i in range(0, len(self.pids), self.num_pids_per_batch):
            selected_pids = self.pids[i:i + self.num_pids_per_batch]
            batch = []
            for pid in selected_pids:
                idxs = self.index_dic[pid]
                # pick num_instances for each PID
                if len(idxs) < self.num_instances:
                    idxs = random.choices(idxs, k=self.num_instances)
                else:
                    idxs = random.sample(idxs, self.num_instances)
                batch.extend(idxs)
            yield from batch  # directly yield indices instead of building huge list

    def __len__(self):
        return len(self.data_source)