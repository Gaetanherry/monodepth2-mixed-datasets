import math
import torch
from torch.utils.data.sampler import RandomSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.smallest_dataset_size = min([len(cur_dataset) for cur_dataset in dataset.datasets])
        assert self.batch_size % self.number_of_datasets == 0, "'batch_size' must be a multiple of the number of datasets"

    def __len__(self):
        return self.batch_size * math.ceil(self.smallest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        # push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        # With naive mixing we are forced to restrict size to smallest dataset
        epoch_samples = self.smallest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, self.batch_size):
            for _ in range(0, self.batch_size, self.number_of_datasets):
                cur_samples = []
                for i in range(self.number_of_datasets):
                    try:
                        cur_sample = sampler_iterators[i].__next__()
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_sample = sampler_iterators[i].__next__()
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
