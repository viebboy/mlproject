"""
data.py: data related tools for mlproject
-----------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-11
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache License 2.0


"""

from __future__ import annotations
import dill
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
import joblib
import os
import numpy as np
import random
from tqdm import tqdm

from mlproject.constants import (
    DISABLE_WARNING,
    NB_PARALLEL_JOBS,
    LOG_LEVEL,
)


@logger.catch
def write_cache(dataset, start_idx: int, stop_idx:int, cache_file: dict):
    bin_file = cache_file['binary_file']
    idx_file = cache_file['index_file']
    record = BinaryBlob(binary_file=bin_file, index_file=idx_file, mode='w')
    if LOG_LEVEL == 'DEBUG':
        for i, idx in tqdm(enumerate(range(start_idx, stop_idx))):
            record.write_index(i, dataset[idx])
    else:
        for i, idx in enumerate(range(start_idx, stop_idx)):
            record.write_index(i, dataset[idx])

    record.close()
    logger.debug(f'complete writing cache file: {bin_file}')


@logger.catch
def write_cache_pickle_safe(dataset_class, dataset_params, start_idx: int, stop_idx:int, cache_file: dict):
    dataset = dataset_class(**dataset_params)

    bin_file = cache_file['binary_file']
    idx_file = cache_file['index_file']
    record = BinaryBlob(binary_file=bin_file, index_file=idx_file, mode='w')
    if LOG_LEVEL == 'DEBUG':
        for i, idx in tqdm(enumerate(range(start_idx, stop_idx))):
            record.write_index(i, dataset[idx])
    else:
        for i, idx in enumerate(range(start_idx, stop_idx)):
            record.write_index(i, dataset[idx])

    record.close()
    logger.debug(f'complete writing cache file: {bin_file}')


class BinaryBlob(TorchDataset):
    """
    abstraction for binary blob storage
    """
    def __init__(self, binary_file: str, index_file: str, mode='r'):
        assert mode in ['r', 'w']
        self._mode = 'write' if mode == 'w' else 'read'

        if mode == 'w':
            # writing mode
            self._fid = open(binary_file, 'wb')
            self._idx_fid = open(index_file, 'w')
            self._indices = set()
        else:
            assert os.path.exists(binary_file)
            assert os.path.exists(index_file)

            # read index file
            with open(index_file, 'r') as fid:
                content = fid.read().split('\n')[:-1]

            self._index_content = {}
            self._indices = set()
            for row in content:
                sample_idx, byte_pos, byte_length, need_conversion = row.split(',')
                self._index_content[int(sample_idx)] = (
                    int(byte_pos),
                    int(byte_length),
                    bool(int(need_conversion)),
                )
                self._indices.add(int(sample_idx))

            # open binary file
            self._fid = open(binary_file, 'rb')
            self._fid.seek(0, 0)
            self._idx_fid = None

            # sorted indices
            self._sorted_indices = list(self._indices)
            self._sorted_indices.sort()

        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def __getitem__(self, i: int):
        if self._mode == 'write':
            raise RuntimeError('__getitem__ is not supported when BinaryBlob is opened in write mode')

        if idx >= len(self):
            raise RuntimeError(f'index {i} is out of range: [0 - {len(self)})')
        idx = self._sorted_indices[i]
        return self.read_index(idx)

    def __len__(self):
        if self._mode == 'write':
            raise RuntimeError('__len__ is not supported when BinaryBlob is opened in write mode')
        return len(self._sorted_indices)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write_index(self, index: int, content):
        assert isinstance(index, int)
        if self._mode == 'write':
            # allow writing
            try:
                # check if index existence
                if index in self._indices:
                    raise RuntimeError(f'Given index={index} has been occuppied. Cannot write')

                # convert to byte string
                if not isinstance(content, bytes):
                    content = dill.dumps(content)
                    # flag to mark whether serialization/deserialization is
                    # needed
                    converted = 1
                else:
                    converted = 0

                # log position before writing
                current_pos = self._fid.tell()
                # write byte string
                self._fid.write(content)
                # write metadata information
                self._idx_fid.write(f'{index},{current_pos},{len(content)},{converted}\n')

                # keep track of index
                self._indices.add(index)

            except Exception as error:
                self.close()
                raise error
        else:
            # raise error
            self.close()
            raise RuntimeError('BinaryBlob was opened in reading mode. No writing allowed')

    def read_index(self, index: int):
        assert isinstance(index, int)
        assert index >= 0

        if self._mode == 'read':
            if index not in self._indices:
                self.close()
                raise RuntimeError(f'Given index={index} does not exist in BinaryBlob')

            # pos is the starting position we need to seek
            target_pos, length, need_conversion = self._index_content[index]
            # we need to compute seek parameter
            delta = target_pos - self._fid.tell()
            # seek to target_pos
            self._fid.seek(delta, 1)

            # read `length` number of bytes
            item = self._fid.read(length)
            # deserialize if needed
            if need_conversion:
                item = dill.loads(item)
            return item
        else:
            self.close()
            raise RuntimeError('BinaryBlob was opened in writing mode. No reading allowed')

    def close(self):
        if self._fid is not None:
            self._fid.close()
        if self._idx_fid is not None:
            self._idx_fid.close()


class CacheDataset(TorchDataset):
    """
    Wrapper to cache a given dataset in BinaryBlob formats
    This will save samples generated from input dataset to disk
    Random augmentation should not be implemented in __getitem__ of the input dataset because
    randomly augmented samples will be reused due to the caching behavior

    To properly implement random augmentation, inheritance from this class should be used
    and random augmentation should be applied after calling __getitem__ of CacheDataset
    """
    def __init__(
        self,
        dataset,
        cache_prefix: str,
        nb_shard: 32,
    ):
        if not DISABLE_WARNING:
            msg = [
                'CacheDataset will save samples generated from ',
                f'the input dataset {dataset.__class__.__name__} to disk. '
                f'Make sure that random augmentation is done after __getitem__ of CacheDataset'
            ]
            for m in msg:
                logger.warning(m)

        # write blob files if needed
        if not self._has_cache(dataset, cache_prefix, nb_shard):
            self._write_cache(dataset, cache_prefix, nb_shard)

        self._read_cache(cache_prefix, nb_shard)
        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def _has_cache(self, dataset, cache_prefix: str, nb_shard: int):
        cache_files, exist = self._get_cache_files(cache_prefix, nb_shard)
        if not exist:
            return False

        # even if we have cached files, we should check if the cached length
        # matches with the lenght of input dataset
        total_cached_samples = 0
        for item in cache_files:
            idx_file = item['index_file']
            with open(idx_file, 'r') as fid:
                total_cached_samples += len(fid.read().split('\n')[:-1])

        if total_cached_samples == len(dataset):
            return True
        else:
            logger.warning(
                (
                    f'mismatched between total number of cached samples ({total_cached_samples}) ',
                    f'and total number of samples in the input dataset ({len(dataset)})'
                 )
            )
            return False

    def __getitem__(self, i: int):
        record_idx, item_idx = self._indices[i]

        # get data for the frame
        item = self._records[record_idx].read_index(item_idx)

        return item

    def close(self):
        for record in self._records:
            record.close()

    def __len__(self):
        return len(self._indices)

    def _get_cache_files(self, cache_prefix, nb_shard):
        cache_files = []
        exist = True
        for i in range(nb_shard):
            bin_file = cache_prefix + '{:09d}.bin'.format(i)
            idx_file = cache_prefix + '{:09d}.idx'.format(i)
            cache_files.append(
                {
                    'binary_file': bin_file,
                    'index_file': idx_file,
                }
            )
            if exist and not os.path.exists(bin_file):
                exist = False

            if exist and not os.path.exists(idx_file):
                exist = False

        return cache_files, exist


    def _read_cache(self, cache_prefix, nb_shard):
        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)
        self._records = []
        self._indices = []

        for f_idx, cache_file in enumerate(cache_files):
            bin_file = cache_file['binary_file']
            idx_file = cache_file['index_file']
            self._records.append(
                BinaryBlob(
                    binary_file=bin_file,
                    index_file=idx_file,
                    mode='r'
                )
            )

            # find the number of samples
            with open(idx_file, 'r') as fid:
                indices = fid.read().split('\n')[:-1]

            # create index to retrieve item
            for i in range(len(indices)):
                self._indices.append((f_idx, i))

        logger.info('complete reading cache files')


    def _write_cache(self, dataset, cache_prefix, nb_shard):
        nb_sample = len(dataset)

        # now split into different shards to perform parallel write
        start_indices, stop_indices = [], []
        shard_size = int(np.ceil(nb_sample / nb_shard))

        for i in range(nb_shard):
            start_indices.append(i * shard_size)
            stop_indices.append(min((i + 1) * shard_size, nb_sample))

        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)

        try:
            joblib.Parallel(n_jobs=NB_PARALLEL_JOBS, backend='loky')(
                joblib.delayed(write_cache)(
                    dataset,
                    start_idx,
                    stop_idx,
                    cache_file,
                )
                for start_idx, stop_idx, cache_file in zip(
                    start_indices,
                    stop_indices,
                    cache_files,
                )
            )
        except Exception:
            for start_idx, stop_idx, cache_file in zip(start_indices, stop_indices, cache_files):
                write_cache(dataset, start_idx, stop_idx, cache_file)

        logger.info('complete writing cache files')


class CacheIterator(TorchDataset):
    """
    Wrapper to cache a given iterator in BinaryBlob formats
    This will save samples generated from input iterator to disk
    """
    def __init__(
        self,
        iterator,
        cache_prefix: str,
        nb_shard: 32,
    ):

        # write blob files if needed
        if not self._has_cache(cache_prefix, nb_shard):
            self._write_cache(iterator, cache_prefix, nb_shard)

        self._read_cache(cache_prefix, nb_shard)
        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def _has_cache(self, cache_prefix: str, nb_shard: int):
        cache_files, exist = self._get_cache_files(cache_prefix, nb_shard)
        return exist

    def __getitem__(self, i: int):
        record_idx = i % len(self._records)

        # get data for the frame
        item = self._records[record_idx].read_index(i)

        return item

    def close(self):
        for record in self._records:
            record.close()

    def __len__(self):
        return self.nb_sample

    def _get_cache_files(self, cache_prefix, nb_shard):
        cache_files = []
        exist = True
        for i in range(nb_shard):
            bin_file = cache_prefix + '{:09d}.bin'.format(i)
            idx_file = cache_prefix + '{:09d}.idx'.format(i)
            cache_files.append(
                {
                    'binary_file': bin_file,
                    'index_file': idx_file,
                }
            )
            if exist and not os.path.exists(bin_file):
                exist = False

            if exist and not os.path.exists(idx_file):
                exist = False

        return cache_files, exist


    def _read_cache(self, cache_prefix, nb_shard):
        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)
        self._records = []
        self.nb_sample = 0

        for f_idx, cache_file in enumerate(cache_files):
            bin_file = cache_file['binary_file']
            idx_file = cache_file['index_file']
            self._records.append(
                BinaryBlob(
                    binary_file=bin_file,
                    index_file=idx_file,
                    mode='r'
                )
            )

            # find the number of samples
            with open(idx_file, 'r') as fid:
                indices = fid.read().split('\n')[:-1]
            self.nb_sample += len(indices)

        logger.info('complete reading cache files')


    def _write_cache(self, iterator, cache_prefix, nb_shard):
        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)
        records = [BinaryBlob(cache_file['binary_file'], cache_file['index_file'], 'w') for cache_file in cache_files]
        count = 0

        if LOG_LEVEL == 'DEBUG':
            loop = tqdm(iterator)
        else:
            loop = iterator

        for sample in loop:
            shard_idx = count % nb_shard
            records[shard_idx].write_index(count, sample)
            count += 1

        for record in records:
            record.close()

        logger.info('complete writing cache files')


class PickleSafeCacheIterator(TorchDataset):
    """
    Alternative version of CacheIterator that receives iterator class name and params separately
    This is supposed to be used with those iterators that cannot be pickled
    """
    def __init__(
        self,
        iterator_class,
        iterator_params,
        cache_prefix: str,
        nb_shard: 32,
    ):
        # write blob files if needed
        if not self._has_cache(cache_prefix, nb_shard):
            self._write_cache(iterator_class, iterator_params, cache_prefix, nb_shard)

        self._read_cache(cache_prefix, nb_shard)
        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def _has_cache(self, cache_prefix: str, nb_shard: int):
        _, exist = self._get_cache_files(cache_prefix, nb_shard)
        return exist

    def __getitem__(self, i: int):
        record_idx = i % len(self._records)

        # get data for the frame
        item = self._records[record_idx].read_index(i)

        return item

    def close(self):
        for record in self._records:
            record.close()

    def __len__(self):
        return self.nb_sample

    def _get_cache_files(self, cache_prefix, nb_shard):
        cache_files = []
        exist = True
        for i in range(nb_shard):
            bin_file = cache_prefix + '{:09d}.bin'.format(i)
            idx_file = cache_prefix + '{:09d}.idx'.format(i)
            cache_files.append(
                {
                    'binary_file': bin_file,
                    'index_file': idx_file,
                }
            )
            if exist and not os.path.exists(bin_file):
                exist = False

            if exist and not os.path.exists(idx_file):
                exist = False

        return cache_files, exist


    def _read_cache(self, cache_prefix, nb_shard):
        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)
        self._records = []
        self.nb_sample = 0

        for f_idx, cache_file in enumerate(cache_files):
            bin_file = cache_file['binary_file']
            idx_file = cache_file['index_file']
            self._records.append(
                BinaryBlob(
                    binary_file=bin_file,
                    index_file=idx_file,
                    mode='r'
                )
            )

            # find the number of samples
            with open(idx_file, 'r') as fid:
                indices = fid.read().split('\n')[:-1]
            self.nb_sample += len(indices)

        logger.info('complete reading cache files')


    def _write_cache(self, iterator_class, iterator_params, cache_prefix, nb_shard):
        iterator = iterator_class(iterator_params)

        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)
        records = [BinaryBlob(cache_file['binary_file'], cache_file['index_file'], 'w') for cache_file in cache_files]
        count = 0

        if LOG_LEVEL == 'DEBUG':
            loop = tqdm(iterator)
        else:
            loop = iterator

        for sample in loop:
            shard_idx = count % nb_shard
            records[shard_idx].write_index(count, sample)
            count += 1

        for record in records:
            record.close()

        logger.info('complete writing cache files')

class PickleSafeCacheDataset(TorchDataset):
    """
    Alternative version of CacheDataset that receives dataset class name and params separately
    This is supposed to be used with those dataset objects that cannot be pickled
    """
    def __init__(
        self,
        dataset_class,
        dataset_params,
        cache_prefix: str,
        nb_shard: 32,
    ):
        if not DISABLE_WARNING:
            msg = [
                'CacheDataset will save samples generated from ',
                f'the input dataset {dataset_class} to disk. '
                f'Make sure that random augmentation is done after __getitem__ of CacheDataset'
            ]
            for m in msg:
                logger.warning(m)

        # write blob files if needed
        if not self._has_cache(dataset_class, dataset_params, cache_prefix, nb_shard):
            self._write_cache(dataset_class, dataset_params, cache_prefix, nb_shard)

        self._read_cache(cache_prefix, nb_shard)
        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def _has_cache(self, dataset_class, dataset_params, cache_prefix: str, nb_shard: int):
        self._nb_sample = len(dataset_class(**dataset_params))

        cache_files, exist = self._get_cache_files(cache_prefix, nb_shard)
        if not exist:
            return False

        # even if we have cached files, we should check if the cached length
        # matches with the lenght of input dataset
        total_cached_samples = 0
        for item in cache_files:
            idx_file = item['index_file']
            with open(idx_file, 'r') as fid:
                total_cached_samples += len(fid.read().split('\n')[:-1])

        if total_cached_samples == self._nb_sample:
            return True
        else:
            logger.warning(
                (
                    f'mismatched between total number of cached samples ({total_cached_samples}) ',
                    f'and total number of samples in the input dataset ({self._nb_sample})'
                 )
            )
            return False

    def __getitem__(self, i: int):
        record_idx, item_idx = self._indices[i]

        # get data for the frame
        item = self._records[record_idx].read_index(item_idx)

        return item

    def close(self):
        for record in self._records:
            record.close()

    def __len__(self):
        return len(self._indices)

    def _get_cache_files(self, cache_prefix, nb_shard):
        cache_files = []
        exist = True
        for i in range(nb_shard):
            bin_file = cache_prefix + '{:09d}.bin'.format(i)
            idx_file = cache_prefix + '{:09d}.idx'.format(i)
            cache_files.append(
                {
                    'binary_file': bin_file,
                    'index_file': idx_file,
                }
            )
            if exist and not os.path.exists(bin_file):
                exist = False

            if exist and not os.path.exists(idx_file):
                exist = False

        return cache_files, exist


    def _read_cache(self, cache_prefix, nb_shard):
        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)
        self._records = []
        self._indices = []

        for f_idx, cache_file in enumerate(cache_files):
            bin_file = cache_file['binary_file']
            idx_file = cache_file['index_file']
            self._records.append(
                BinaryBlob(
                    binary_file=bin_file,
                    index_file=idx_file,
                    mode='r'
                )
            )

            # find the number of samples
            with open(idx_file, 'r') as fid:
                indices = fid.read().split('\n')[:-1]

            # create index to retrieve item
            for i in range(len(indices)):
                self._indices.append((f_idx, i))

        logger.info('complete reading cache files')


    def _write_cache(self, dataset_class, dataset_params, cache_prefix, nb_shard):

        # now split into different shards to perform parallel write
        start_indices, stop_indices = [], []
        shard_size = int(np.ceil(self._nb_sample / nb_shard))

        for i in range(nb_shard):
            start_indices.append(i * shard_size)
            stop_indices.append(min((i + 1) * shard_size, self._nb_sample))

        cache_files, _ = self._get_cache_files(cache_prefix, nb_shard)

        joblib.Parallel(n_jobs=NB_PARALLEL_JOBS, backend='loky')(
            joblib.delayed(write_cache_pickle_safe)(
                dataset_class,
                dataset_params,
                start_idx,
                stop_idx,
                cache_file,
            )
            for start_idx, stop_idx, cache_file in zip(
                start_indices,
                stop_indices,
                cache_files,
            )
        )

        logger.info('complete writing cache files')


class ConcatDataset(TorchDataset):
    """
    Wrapper to concatenate many dataset objects
    """
    def __init__(self, *datasets):
        self._datasets = []
        self._start_indices = []

        start_index = 0
        for dataset in datasets:
            assert hasattr(dataset, '__getitem__')
            assert hasattr(dataset, '__len__')
            self._datasets.append(dataset)
            self._start_indices.append(start_index)
            start_index += len(dataset)

        self._total_samples = start_index
        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def __len__(self):
        return self._total_samples

    def __getitem__(self, i: int):
        # go backward to find which dataset to use based on the start index
        for dataset_idx, boundary in enumerate(self._start_indices[::-1]):
            if i >= boundary:
                break
        # remember to access dataset index backward
        return self._datasets[-dataset_idx][i - boundary]


class ForwardValidationDataset:
    """
    Dataset wrapper that splits a time-series dataset for forward-validation
    """
    def __init__(self, dataset, nb_fold, nb_test_fold, test_fold_index, prefix):
        assert nb_test_fold < nb_fold
        assert test_fold_index < nb_test_fold
        assert prefix in ['train', 'test']

        total_len = len(dataset)
        fold_size = int(np.ceil(total_len / nb_fold))
        if prefix == 'train':
            self.start_index = 0
            self.stop_index = (nb_fold - nb_test_fold + test_fold_index) * fold_size
        else:
            self.start_index = (nb_fold - nb_test_fold + test_fold_index) * fold_size
            self.stop_index = min(total_len, self.start_index + fold_size)

        self._dataset = dataset
        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def __len__(self):
        return self.stop_index - self.start_index

    def __getitem__(self, i):
        return self._dataset[self.start_index + i]


class PickleSafeForwardValidationDataset(ForwardValidationDataset):
    def __init__(
        self,
        dataset_class,
        dataset_params,
        nb_fold,
        nb_test_fold,
        test_fold_index,
        prefix
    ):
        dataset = dataset_class(**dataset_params)
        super().__init__(dataset, nb_fold, nb_test_fold, test_fold_index, prefix)


class ClassificationSampler:
    def __init__(self, dataset, class_percentage: dict, label_getter=None):
        # compute sample indices from each class
        class_indices = self._compute_class_indices(dataset, label_getter)

        # make sure percentage is right
        total = 0
        for lb in class_indices:
            assert lb in class_percentage
            total += class_percentage[lb]
        assert total == 1, 'the total percentage of all classes must be 1'

        # compute the target length for each class
        target_len = self._compute_target_length(class_percentage, len(dataset))

        # now perform sampling
        self._dataset = dataset
        self._target_len = target_len
        self._class_indices = class_indices

        # perform initial sampling
        self._sampling()
        # create a mask array to check whether a sample has been accessed or not
        # when all samples have been accesssed, reset the sampling
        self._access_mask = [False for _ in range(len(dataset))]
        # counter is to avoid checking the status of the mask all the time
        self._access_counter = 0

        self._cur_index = -1

    def _compute_class_indices(self, dataset, label_getter):
        class_indices = {}
        for i in range(len(dataset)):
            if label_getter is None:
                # if not specified, assume the 2nd element is label
                lb = dataset[i][1]
            else:
                lb = label_getter(dataset[i])
            if lb not in class_indices:
                class_indices[lb] = [i,]
            else:
                class_indices[lb].append(i)

        return class_indices

    def _compute_target_length(self, target_percentage, total_len):
        target_len = {}
        labels = list(target_percentage.keys())
        allocated = 0
        for lb in labels[:-1]:
            target_len[lb] = int(target_percentage[lb] * total_len)
            allocated += target_len[lb]

        # the last one should be computed by subtraction
        target_len[labels[-1]] = total_len - allocated

        for lb, value in target_len.items():
            if value <= 0:
                raise RuntimeError(
                    f'the given class_percentage results in class {lb} having {value} samples'
                )

        return target_len

    def _sampling(self):
        logger.debug('compute sampling strategy')
        if not hasattr(self, '_sampler'):
            self._sampler = {lb: [] for lb in self._target_len}

        sampler = {}
        for lb, target_size in self._target_len.items():
            actual_size = len(self._class_indices[lb])
            if target_size >= actual_size:
                # target length is larger than actual length
                # requires oversampling
                nb_repeat = int(np.floor(target_size / actual_size))
                sampler[lb] = []
                # repeat entire index set
                for _ in range(nb_repeat):
                    sampler[lb].extend(self._class_indices[lb])

                # compute how much more do we need
                leftover = target_size - len(sampler[lb])
                if leftover > 0:
                    sampler[lb].extend(random.sample(self._class_indices[lb], leftover))
            else:
                # target length is less than actual length
                # need to perform sampling
                old_indices = set(self._sampler[lb])
                leftover_indices = [i for i in self._class_indices[lb] if i not in old_indices]
                if len(leftover_indices) >= target_size:
                    sampler[lb] = random.sample(leftover_indices, target_size)
                else:
                    sampler[lb] = leftover_indices
                    # sampling from the old list
                    sampler[lb].extend(random.sample(self._sampler[lb], target_size - len(leftover_indices)))

        self._sampler = sampler
        self._index_map = []
        labels = list(self._target_len.keys())
        counter = {lb: 0 for lb in labels}
        while True:
            for lb in labels:
                self._index_map.append((lb, counter[lb]))
                counter[lb] += 1
                if counter[lb] >= len(self._sampler[lb]):
                    labels.remove(lb)
            if len(labels) == 0:
                break

        assert len(self._index_map) == len(self._dataset)

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        if self._access_counter >= len(self):
            total_accessed = np.sum(self._access_mask)
            if total_accessed == len(self):
                # all items have been accessed, reset sampling
                self._sampling()
                self._access_counter = 0
                self._access_mask = [False for _ in range(len(self))]
            else:
                self._access_counter = total_accessed

        self._access_mask[i] = True
        self._access_counter += 1

        lb, idx = self._index_map[i]
        original_idx = self._sampler[lb][idx]
        return self._dataset[original_idx]


class PickleSafeClassificationSampler(ClassificationSampler):
    def __init__(self, dataset_class, dataset_params, class_percentage, label_getter=None):
        dataset = dataset_class(**dataset_params)
        super().__init__(dataset, class_percentage, label_getter)
