import pickle

bug_list = [61, 267, 268, 317, 382, 544, 794, 804, 1906, 2257, 2308, 2411, 2414, 2528, 2951, 3124, 3166, 3167, 3168, 3398, 3794, 3820, 4219, 4220, 4528, 4530, 4548, 4549, 4662, 4664, 4753, 4807, 4810, 4927, 4928, 4930, 5068, 5072, 5074, 5097, 5142, 5245, 5368, 5542, 5543, 5644, 5689, 5792, 5794, 5890, 6483, 6642, 6643, 6994, 7437, 7472, 7473, 7542, 7545]
print(len(bug_list))

with open('with_bug/amr_seq.pkl', mode='rb') as file:
    bug_amr_seq_list = pickle.load(file)

# with open('with_bug/edge_index_list.pkl', model='rb') as file:
#     bug_edge = pickle.load(file)

with open('with_bug/enhanced_complex_sentence.txt', mode='r', encoding='utf-8') as file:
    bug_complex_sentences = file.readlines()

with open('with_bug/enhanced_complex_tokenize.pkl', mode='rb') as file:
    bug_complex_tokenize = pickle.load(file)

with open('with_bug/enhanced_simple_sentence.txt', mode='r', encoding='utf-8') as file:
    bug_simple_sentences = file.readlines()

# with open('with_bug/nodes_without_no.pkl', model='rb') as file:
#     bug_nodes = pickle.load(file)

no_bug_amr_seq = open('no_bug/amr_seq_list.pkl', mode='wb')
# no_bug_edge = open('no_bug/edge.pkl', model='wb')
no_bug_complex_tokenize = open('no_bug/complex_tokenize.pkl', mode='wb')
# no_bug_nodes = open('no_bug/nodes.pkl', model='wb')
no_bug_complex_sentences = open('no_bug/complex_sentences.txt', mode='a', encoding='utf-8')
no_bug_simple_sentences = open('no_bug/simple_sentences.txt', mode='a', encoding='utf-8')

no_bug_amr_seq_list = []
# no_bug_edge_list = []
no_bug_complex_tokenize_list = []
# no_bug_nodes_list = []

for index in range(len(bug_amr_seq_list)):
    if index in bug_list:
        continue
    no_bug_amr_seq_list.append(bug_amr_seq_list[index])
    # no_bug_edge_list.append(bug_edge[index])
    no_bug_complex_tokenize_list.append(bug_complex_tokenize[index])
    # no_bug_nodes_list.append(bug_nodes[index])

    no_bug_complex_sentences.write(bug_complex_sentences[index])
    no_bug_simple_sentences.write(bug_simple_sentences[index])

pickle.dump(no_bug_amr_seq_list, no_bug_amr_seq)
# pickle.dump(no_bug_edge_list, no_bug_edge)
pickle.dump(no_bug_complex_tokenize_list, no_bug_complex_tokenize)
# pickle.dump(no_bug_nodes_list, no_bug_nodes)
