# mlproject.data Module

There are a couple of convenient abstraction for data handling under `mlproject.data`. 

This doc explains the working mechanisms and use-cases for each abstraction.

## mlproject.data.BinaryBlob
-----------------------------

This is datastructure is used to store indexed data in binary format.

For example, if you have a dataset that consists of millions of images, the cost of loading and decoding individual images during training can be daunting, which can be a major overhead in data prep.

Because data in BinaryBlob is saved in a contiguous disk segment, we avoid random read. 

The example below demonstrate how to write data into a BinaryBlob and read it back:

```python
from mlproject.data import BinaryBlob
import numpy as np

blob = BinaryBob(
    binary_file='random_data.binary', 
    index_file='random_data.idx', 
    mode='w',
)

# create a list of random numpy array
data = [np.random.rand(256, 256, 3) for _ in range(100)]

# now write this list to blob
for idx, sample in enumerate(data):
    blob.write_index(idx, sample)
    
# we need to close the blob to complete the writing process
blob.close()

# now we open the blob and read data back
recon_blob = BinaryBob(
    binary_file='random_data.binary', 
    index_file='random_data.idx', 
    mode='r', # remember to open in read mode, otherwise the data files will be deleted!
)

# verify
for idx, original in enumerate(data):
    reconstruct = recon_blob.read_index(idx)
    np.testing.assert_allclose(original, reconstruct)

# we could also loop through a BinaryBlob
for original, reconstruct in zip(data, recon_blob):
    np.testing.assert_allclose(original, reconstruct)

recon_blob.close()
```

In the above example, we open a `BinaryBlob` in writing mode and provide 2 paths: the filename of the binary file where data is saved and the index file where the metadata is saved.
When we complete the writing, we should call close() to finalize the writing.

Note that the indices that we used to write the data doesn't need to be in any order or value range, as long as they are integers.

For example, one could write data into 3 indices: -1, 3, 10 and later read them back using these indices. 

To read back a blob, we need to open it in read mode (mode='r').

To access the data in a BinaryBlob, we could:
- use `read_index(k)` to access data at a given index k. 
  This index must belong to one of the values that were used to write. 
- use indexing operator `[]`. In this case, the index must be non-negative index and it must be between [0, len(blob)] 
- use loop, i.e., `for sample in blob:`

Note that indices used in the indexing operator `[]` will correspond to sorted indices that were used in writing the blob.

That is, if we used -1, 3, 5 to write sample a, b, and c into a blob called x, then `a=x[0]`, `b=x[1]` and `c=x[2]`. 
We also have `a=x.read_index(-1)`, `b=x.read_index(3)`, and `c=x.read_index(5)`

If you already have a dataset instance and don't want to handle writing BinaryBlob files and implementing an interface for them, you could take advantage of `mlproject.data.CacheDataset` described below:


## mlproject.data.ConcatDataset
--------------------------------

This provides a dataset interface for a list of datasets. For example, if you have 5 dataset objects: d1, d2, d3, d4, d5.

Each provides dataset interfaces (`__getitem__` and `__len__`).

If we want to create a dataset object that has `K = len(d1) + len(d2) + len(d3) + len(d4) + len(d5)` samples according to that order,

we could simply wrap them into `ConcatDataset` as `dataset = ConcatDataset(d1, d2, d3, d4, d5)`.

In general, give a list of dataset (`dataset_list`), we could combine them and provide dataset interfaces by wrapping them as `ConcatDataset(*dataset_list)`


## mlproject.data.CacheDataset
------------------------------

This is a convenient class that allows us to wrap cache a dataset into BinaryBlob without having to handle the anything. 

Given a dataset that supports indexing operator `__getitem__()` and `__len__()`, you can create a binary cached version of this dataset by wrapping it under `mlproject.data.CacheDataset` as follows:

```python
from mlproject.data import CacheDataset

cache_prefix = 'path_to_cache_file/my_dataset'
nb_shard = 8 # here we want to split into 8 different blobs
cached_dataset = CacheDataset(your_dataset, cache_prefix, nb_shard=nb_shard)
```

The first time you construct a `CacheDataset` object, data will be written (in parallel) to the different binary blobs. 

Once there are cached files, the next time you create CacheDataset, it automatically loads these cached files.

`CacheDataset` also provide dataset interfaces. That is `__getitem__()`, `__len__()` and iteration. 

## mlproject.data.CacheIterator
-------------------------------

This abstraction is similar to `mlproject.data.CacheDataset`. It can be used to cache an iterator into BinaryBlob files and provide dataset interfaces. 


## mlproject.data.PickleSafeCacheDataset
----------------------------------------

This is a pickle-safe version of `mlproject.data.CacheDataset`. 

Instead of passing a dataset object to perform caching, users could provide the name of the dataset class and its parameters.

The construction of the dataset instance will be done during caching. 

Because we want to perform parallel caching (writing multiple BinaryBlob files at the same time), the dataset object must be serialized to pass to different processes.

Explicitly passing dataset object might lead to failure in serialization when writing cache files in parallel.

This interface should be used when issues appeared during the caching process.

## mlproject.data.PickleSafeCacheIterator
-----------------------------------------

This is similar to `mlproject.data.CacheIterator`, but pickle-safe. Similar idea to `mlproject.data.PickleSafeCacheDataset`. 


## mlproject.data.ForwardValidationDataset
------------------------------------------

This class allows us to split a time-series dataset for forward validation purposes. 

A time-series dataset is the one with samples ordered in chronological order.

That is, sample i (dataset[i]) semantically happens before sample i+1 (dataset[i+1]). 
