import os
import torch
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen
import agents
from random import shuffle
from collections import OrderedDict


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Prepare dataloaders

    train_dataset_splits = {}
    val_dataset_splits = {}
    task_output_space = {}
    pre_count = 0
    accu_dataset_len = 0
    for i in range(len(args.dataset)):
        train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset[i]](args.dataroot, args.train_aug)
        cur_pre_count = 0
        if args.n_permutation > 0:
            cur_train_dataset_splits, cur_val_dataset_splits, cur_task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                 args.n_permutation,
                                                                                 remap_class=not args.no_class_remap)
        else:
            cur_train_dataset_splits, cur_val_dataset_splits, cur_task_output_space, cur_pre_count = SplitGen(train_dataset, val_dataset, accu_dataset_len, pre_count,
                                                                              first_split_sz=args.first_split_size,
                                                                         other_split_sz=args.other_split_size,
                                                                              rand_split=args.rand_split,
                                                                              remap_class=not args.no_class_remap)

        for index in range(len(cur_train_dataset_splits)):
            train_dataset_splits[str(index + pre_count + 1)] = cur_train_dataset_splits[str(index + pre_count + 1)]
            val_dataset_splits[str(index + pre_count + 1)] = cur_val_dataset_splits[str(index + pre_count + 1)]
            task_output_space[str(index + pre_count + 1)] = cur_task_output_space[str(index + pre_count+ 1)]

        pre_count = cur_pre_count
        accu_dataset_len += train_dataset.number_classes


    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                    'model_type':args.model_type, 'model_name': args.model_name, 'model_weights':args.model_weights,
                    'out_dim':{'All':args.force_out_dim} if args.force_out_dim>0 else task_output_space,
                    'optimizer':args.optimizer,
                    'print_freq':args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef}
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    print(agent.model)
    print('#parameter of model:',agent.count_parameter())

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:',task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        task_names = ['All']
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

        acc_table['All'] = {}
        acc_table['All']['All'] = agent.validation(val_loader)

    else:  # Incremental learning
        # Feed data to agent and evaluate agent's performance
        for i in range(len(task_names)):
            train_name = task_names[i]
            print('======================',train_name,'=======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                      batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            if args.incremental_class:
                agent.add_valid_output_dim(task_output_space[train_name])

            # Learn
            agent.learn_batch(train_loader, val_loader)

            # Evaluate
            acc_table[train_name] = OrderedDict()
            for j in range(i+1):
                val_name = task_names[j]
                print('validation split name:', val_name)
                val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
                val_loader = torch.utils.data.DataLoader(val_data,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.workers)
                acc_table[val_name][train_name] = agent.validation(val_loader)

    return acc_table, task_names