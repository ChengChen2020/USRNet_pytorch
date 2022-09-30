import os
import json
from collections import OrderedDict


def parse(opt_path, is_train=True):

    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['is_train'] = is_train

    path_task = os.path.join(opt['log_path']['root'], opt['task'])
    if is_train:
        opt['log_path']['models'] = os.path.join(path_task, 'models')
        opt['log_path']['images'] = os.path.join(path_task, 'images')
    else:  # test
        opt['log_path']['images'] = os.path.join(path_task, 'test_images')

    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase
        if 'dataroot_H' in dataset and dataset['dataroot_H'] is not None:
            dataset['dataroot_H'] = os.path.expanduser(dataset['dataroot_H'])
        if 'dataroot_L' in dataset and dataset['dataroot_L'] is not None:
            dataset['dataroot_L'] = os.path.expanduser(dataset['dataroot_L'])

    return opt


if __name__ == '__main__':
    parse('config.json', is_train=True)
