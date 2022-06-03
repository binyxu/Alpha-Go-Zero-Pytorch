from __future__ import print_function
from msilib import Directory
import pickle
import torch
import os
from policy_value_net_pytorch import PolicyValueNet
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure, RandPlayer
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import time
from collections import OrderedDict

def run():
    directory = 'weight/'
    for files in os.listdir(directory):
        load_file = directory + files
        save_file = directory + 'weight/numpy_' + files
        my_model = torch.load(load_file)
        params = []
        for key in my_model:
            if 'fc' in key and 'weight' in key:
                params.append(my_model[key].numpy().T)
            elif 'conv' in key and 'weight' in key:
                params.append(my_model[key].numpy()[:,:,::-1,::-1].copy())
            else:
                params.append(my_model[key].numpy())
        pickle.dump(params, open(save_file, 'wb'), protocol=2)

    # param_theano = pickle.load(open('best_policy_6_6_4.model', 'rb'))
    # keys = ['conv1.weight' ,'conv1.bias' ,'conv2.weight' ,'conv2.bias' ,'conv3.weight' ,'conv3.bias'  
    #     ,'act_conv1.weight' ,'act_conv1.bias' ,'act_fc1.weight' ,'act_fc1.bias'     
    #     ,'val_conv1.weight' ,'val_conv1.bias' ,'val_fc1.weight' ,'val_fc1.bias' ,'val_fc2.weight' ,'val_fc2.bias']
    # param_pytorch = OrderedDict()
    # for key, value in zip(keys, param_theano):
    #     if 'fc' in key and 'weight' in key:
    #         param_pytorch[key] = torch.FloatTensor(value.T)
    #     elif 'conv' in key and 'weight' in key:
    #         param_pytorch[key] = torch.FloatTensor(value[:,:,::-1,::-1].copy())
    #     else:
    #         param_pytorch[key] = torch.FloatTensor(value)

if __name__ == '__main__':
    run()
