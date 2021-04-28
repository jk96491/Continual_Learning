import torch
from random import shuffle
from .wrapper import Subclass, AppendName, Permutation


def SplitGen(train_dataset, val_dataset, accu_dataset_len ,pre_count, first_split_sz=2, other_split_sz=2, rand_split=False, remap_class=True):
    '''
    Generate the dataset splits based on the labels.
    :param train_dataset: (torch.utils.data.dataset)
    :param val_dataset: (torch.utils.data.dataset)
    :param first_split_sz: (int)
    :param other_split_sz: (int)
    :param rand_split: (bool) Randomize the set of label in each split
    :param remap_class: (bool) Ex: remap classes in a split from [2,4,6 ...] to [0,1,2 ...]
    :return: train_loaders {task_name:loader}, val_loaders {task_name:loader}, out_dim {task_name:num_classes}
    '''
    assert train_dataset.number_classes == val_dataset.number_classes,'Train/Val has different number of classes'
    num_classes = train_dataset.number_classes

    train_dataset.labels += accu_dataset_len
    for i in range(len(train_dataset.dataset.train_labels)):
        train_dataset.dataset.train_labels[i] += accu_dataset_len

    val_dataset.labels += accu_dataset_len

    for i in range(len(val_dataset.dataset.test_labels)):
        val_dataset.dataset.test_labels[i] += accu_dataset_len

    # Calculate the boundary index of classes for splits
    # Ex: [0,2,4,6,8,10] or [0,50,60,70,80,90,100]
    split_boundaries = [0, first_split_sz]
    while split_boundaries[-1]<num_classes:
        split_boundaries.append(split_boundaries[-1]+other_split_sz)
    print('split_boundaries:',split_boundaries)
    assert split_boundaries[-1]==num_classes,'Invalid split size'

    # Assign classes to each splits
    # Create the dict: {split_name1:[2,6,7], split_name2:[0,3,9], ...}
    if not rand_split:
        class_lists = {str(i):list(range(split_boundaries[i-1] + accu_dataset_len,split_boundaries[i] + accu_dataset_len)) for i in range(1,len(split_boundaries))}
    else:
        randseq = torch.randperm(num_classes)
        class_lists = {str(i):randseq[list(range(split_boundaries[i-1] + accu_dataset_len,split_boundaries[i] + accu_dataset_len))].tolist() for i in range(1, len(split_boundaries))}
    print(class_lists)

    # Generate the dicts of splits
    # Ex: {split_name1:dataset_split1, split_name2:dataset_split2, ...}
    train_dataset_splits = {}
    val_dataset_splits = {}
    task_output_space = {}
    for name,class_list in class_lists.items():
        name_int = int(name) + pre_count
        train_dataset_splits[name] = AppendName(Subclass(train_dataset, class_list, remap_class), str(name_int))
        val_dataset_splits[name] = AppendName(Subclass(val_dataset, class_list, remap_class), str(name_int))
        task_output_space[name] = len(class_list)

    pre_count = len(split_boundaries) + pre_count - 1

    return train_dataset_splits, val_dataset_splits, task_output_space, pre_count


def PermutedGen(train_dataset, val_dataset, n_permute, remap_class=False):
    sample, _ = train_dataset[0]
    n = sample.numel()
    train_datasets = {}
    val_datasets = {}
    task_output_space = {}
    for i in range(1,n_permute+1):
        rand_ind = list(range(n))
        shuffle(rand_ind)
        name = str(i)
        if i==1:  # First task has no permutation
            train_datasets[name] = AppendName(train_dataset, name)
            val_datasets[name] = AppendName(val_dataset, name)
        else:
            # For incremental class scenario, use remap_class=True
            first_class_ind = (i-1)*train_dataset.number_classes if remap_class else 0
            train_datasets[name] = AppendName(Permutation(train_dataset, rand_ind), name, first_class_ind=first_class_ind)
            val_datasets[name] = AppendName(Permutation(val_dataset, rand_ind), name, first_class_ind=first_class_ind)
        task_output_space[name] = train_dataset.number_classes

    return train_datasets, val_datasets, task_output_space