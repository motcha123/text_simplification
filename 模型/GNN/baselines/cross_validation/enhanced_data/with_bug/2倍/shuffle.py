import pickle
import sys

import torch
import random

random.seed(42)

shuffled_index = random.sample(list(range(3231)), 3231)
print(shuffled_index)
bug_list = [68, 69, 107, 333, 692, 736, 764, 826, 926, 1270, 1585, 1725, 1781, 1927, 1975, 2033, 2043, 2061, 2101, 2151, 2193, 2321, 2517, 2571, 2597, 2619, 2681, 2711, 2759, 3021, 3143]
print(len(bug_list))

shuffled_list = []

with open('enhanced_complex_sentence.txt', mode='r', encoding='utf-8') as file:
    complex_sentences = file.readlines()
    print(len(complex_sentences))

with open('enhanced_simple_sentence.txt', mode='r', encoding='utf-8') as file:
    simple_sentences = file.readlines()
    print(len(simple_sentences))

new_complex_sentences = []
new_simple_sentences = []
for index in range(len(complex_sentences)):
    if index in bug_list:
        continue
    else:
        new_complex_sentences.append(complex_sentences[index])
        new_simple_sentences.append(simple_sentences[index])

complex_sentences = new_complex_sentences
simple_sentences = new_simple_sentences

with open('amr_seq_list.pkl', mode='rb') as file:
    amr_seq_list = pickle.load(file)
    print(len(amr_seq_list))

with open('edge_index_list.pkl', mode='rb') as file:
    edge_index_list = pickle.load(file)
    print(len(edge_index_list))

with open('mat_list.pkl', mode='rb') as file:
    mat_list = pickle.load(file)
    print(len(mat_list))

with open('node_pad_list.pkl', mode='rb') as file:
    node_pad_list = pickle.load(file)
    print(len(node_pad_list))

complex_sentence_list = []
simple_sentence_list = []

new_amr_seq_list = []
new_edge_index_list = []
new_mat_list = []
new_node_pad_list = []

for index in shuffled_index:
    new_amr_seq_list.append(amr_seq_list[index])
    new_edge_index_list.append(edge_index_list[index])
    new_mat_list.append(mat_list[index])
    new_node_pad_list.append(node_pad_list[index])

    complex_sentence = complex_sentences[index]
    complex_sentence = complex_sentence.replace('\n', '')
    simple_sentence = simple_sentences[index]
    simple_sentence = simple_sentence.replace('\n', '')
    complex_sentence_list.append(complex_sentence)
    simple_sentence_list.append(simple_sentence)

with open('shuffle/amr_seq_list.pkl', mode='wb') as file:
    pickle.dump(new_amr_seq_list, file)

with open('shuffle/edge_index_list.pkl', mode='wb') as file:
    pickle.dump(new_edge_index_list, file)

with open('shuffle/mat_list.pkl', mode='wb') as file:
    pickle.dump(new_mat_list, file)

with open('shuffle/node_pad_list.pkl', mode='wb') as file:
    pickle.dump(new_node_pad_list, file)

with open('shuffle/complex_sentence_list.pkl', mode='wb') as file:
    pickle.dump(complex_sentence_list, file)

with open('shuffle/simple_sentence_list.pkl', mode='wb') as file:
    pickle.dump(simple_sentence_list, file)

