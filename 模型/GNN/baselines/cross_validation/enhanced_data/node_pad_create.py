import pickle
import torch

with open('no_bug_round_2/nodes_without_no.pkl', mode='rb') as nodes_file:
    nodes_list = pickle.load(nodes_file)

pads_list = []

for nodes in nodes_list:
    one_nums = 0
    for node in nodes:
        if node != '[PAD]':
            one_nums += 1
    pads = [1] * one_nums
    pads.extend([0] * (150- one_nums))

    pads_list.append(torch.tensor(pads))


with open('no_bug_round_2/nodes_pad.pkl', mode='wb') as nodes_pads:
    pickle.dump(pads_list, nodes_pads)

