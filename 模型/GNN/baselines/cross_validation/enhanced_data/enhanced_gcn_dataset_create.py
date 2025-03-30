import copy
import sys
from hanlp_restful import HanLPClient
import time
import pickle
import os
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch.nn import Linear
import re
import hanlp
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn


def collate_fn(batch):
    # 自定义转换操作
    # 这里以处理不定长序列为例
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    return data, target


# 这个类通过获取一个中文句子，返回其GCN的整合特征
class graph_emb(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()

        self.conv1 = GCNConv(d_model, d_model)  # 定义输入特征数量和输出特征数量
        self.conv2 = GCNConv(d_model, d_model)
        self.conv3 = GCNConv(d_model, d_model)
        self.linear_layer = Linear(d_model, d_model)

    def forward(self, graph_x, graph_edge):

        _h = graph_x
        h = self.conv1(graph_x, graph_edge)  # 输入特征和邻接矩阵
        h = h.relu()
        out = self.linear_layer(h)
        out = global_mean_pool(out, batch=None)

        return h, out

# 关闭警告，主要是torch的
import warnings
warnings.filterwarnings("ignore")

# amr模型
amr_model = hanlp.load('MRP2020_AMR_ENG_ZHO_XLM_BASE')

def get_graph_sequence(tokenize_result):
    """
    输入一句话，返回其AMR的序列化表示
    :param sentence: text
    :return:
    """
    amr = amr_model(tokenize_result)
    str_amr = str(amr)
    # print(str_amr)
    # amr_2 = amr_model(tokenize_result, output_amr=False)
    # # 有的AMR里面缺少结点，怎么回事？
    # for item in amr_2['nodes']:
    #     print(item['label'])
    # for item in amr_2.items():
    #     print(item)
    str_amr = re.sub('-\d\d', '', str_amr)
    str_amr_lines = str_amr.split('\n')

    for index in range(len(str_amr_lines)):
        raw_data = copy.deepcopy(str_amr_lines[index])
        raw_data = raw_data.replace(')', '')
        if index == 0:
            node_index = raw_data.split('/')[0][1:].strip()
            str_amr_lines[index] = str_amr_lines[index].replace(node_index, '', 1)
            str_amr_lines[index] = str_amr_lines[index].replace(' / ', '', 1)
        else:
            if '/' in raw_data:
                find = re.findall('\((.* / )', str_amr_lines[index])
                str_amr_lines[index] = str_amr_lines[index].replace(find[0], '')
            else:
                node_index = raw_data.split(' ')[-1]
                str_amr_lines[index] = str_amr_lines[index].replace(node_index, '', 1)
                str_amr_lines[index] = str_amr_lines[index].replace(' / ', '', 1)
    str_amr = '\n'.join(str_amr_lines)
    str_amr = str_amr.replace('\n', '')
    str_amr = str_amr.replace(' ', '')

    # print(str_amr)

    return str_amr


def get_graph_edge_new(tokenize_result):
    """
    输入一个句子，返回AMR的结点和边
    :param sentence: text
    :return:

    可以先按照空格的多少，记录一个“缩进等级”
    词，是左号后紧跟的
    边，是冒号后紧跟的，上一个比其缩进少的，[(编号, 边, 结点, 缩进等级), (编号, 边, 结点, 缩进等级)] 从第二个结点开始，向前寻找最近的，缩进少1的结点，构建新的边和结点

    """
    amr = amr_model(tokenize_result)
    str_amr = str(amr)
    str_amr = re.sub('-\d\d', '', str_amr)
    # print(str_amr)
    str_amr_lines = str_amr.split('\n')

    # 先记录缩进
    spaces_list = []
    for index in range(len(str_amr_lines)):
        if index == 0:
            spaces_list.append(0)
        else:
            if '/' not in str_amr_lines[index]:
                spaces_list.append(str_amr_lines[index-1].split(':')[0].count(' '))
            else:
                spaces_list.append(str_amr_lines[index].split(':')[0].count(' '))

    result = []  # [(编号, 边, 结点, 缩进等级), (编号, 边, 结点, 缩进等级)]
    additional_edges = []  # [source, target, edge_type]
    for index in range(len(str_amr_lines)):
        jump_flag = 0
        if ')' in str_amr_lines[index]:
            str_amr_lines[index] = str_amr_lines[index].replace(')', '')
        raw_data = copy.deepcopy(str_amr_lines[index])
        if index == 0:
            node_name = raw_data.split('/')[1].strip()
            node_index = raw_data.split('/')[0][1:].strip()
            edge_type = None
        else:
            if '/' in raw_data:
                data = str_amr_lines[index].split('/')
                node_index = (data[0].split('(')[-1].strip())
                node_name = (data[1].strip())
                edge_type = re.findall(':(.*?) ', raw_data)[0]
            else:
                source = node_index + '-' + node_name
                target = raw_data.split(' ')[-1]
                edge_type = re.findall(':(.*?) ', raw_data)[0]
                additional_edges.append([source, target, node_index + '+' + edge_type])
                jump_flag = 1
        if jump_flag == 0:
            retraction_num = spaces_list[index]
            result.append((node_index, edge_type, node_name, retraction_num))

    # 补全 target 的名字
    for idx in range(len(additional_edges)):
        for r in result:
            if additional_edges[idx][1] == r[0]:
                additional_edges[idx][1] = r[0] + '-' + r[2]

    nodes = []
    nodes_without_no = []
    edge_index_source = []
    edge_index_target = []
    additional_edges_index = 0
    for index in range(len(result)):
        no, edge, node, spaces = result[index][0], result[index][1], result[index][2], result[index][3]
        try:
            additional_source = additional_edges[additional_edges_index][0]
            additional_target = additional_edges[additional_edges_index][1]
            additional_edge_type = additional_edges[additional_edges_index][2]
        except:
            additional_source = 'tyuiolp;'
            additional_target = 'tyuiolp;'
            additional_edge_type = 'tyuiolp;'
        if index == 0:
            nodes.append(str(no) + '-' + node)
            nodes_without_no.append(node)
        else:
            for index_before in range(index - 1, -1, -1):
                if result[index_before][3] < spaces:
                    nodes.append(str(no) + '-' + edge)
                    nodes_without_no.append(edge)
                    edge_index_source.append(str(result[index_before][0]) + '-' + result[index_before][2])
                    edge_index_target.append(str(no) + '-' + edge)

                    nodes.append(str(no) + '-' + node)
                    nodes_without_no.append(node)
                    edge_index_source.append(str(no) + '-' + edge)
                    edge_index_target.append(str(no) + '-' + node)
                    break
        if additional_source == (no + '-' + node):
            nodes.append(additional_edge_type)
            nodes_without_no.append(additional_edge_type.split('+')[1])
            edge_index_source.append(additional_source)
            edge_index_target.append(additional_edge_type)
            edge_index_source.append(additional_edge_type)
            edge_index_target.append(additional_target)
            additional_edges_index += 1


    # return nodes, edge_index_source, edge_index_target  # 没转成编号之前
    # print(nodes_without_no)
    # print(edge_index_source)
    # print(edge_index_target)
    # 将边的对应关系变成nodes中的索引
    for ind in range(len(edge_index_source)):
        edge_index_source[ind] = nodes.index(edge_index_source[ind])
        edge_index_target[ind] = nodes.index(edge_index_target[ind])
    return nodes_without_no, edge_index_source, edge_index_target


def alignment_one_sentence(amr_seq_ids:list, nodes_without_no:list, bert_tokenizer):
    """
    对齐单个句子
    将结点列表映射为bert得到的向量
    :param amr_seq_ids: 序列化的 AMR 经过 bert分词器 得到的 索引列表
    :param nodes_without_no: 结点列表，原文
    :return: 返回需要进行乘法操作的矩阵
    """

    # 用一个字典记录句子中的单词是否被匹配过了
    used_dict = {}
    for index in range(len(amr_seq_ids)):
        used_dict[index] = 0

    # 对每一个词，都构建一个 长度为 len(nodes_without_no) 的向量，如果这个词没有贡献，那么向量就全为0
    pos_list = []
    pos_weight = []
    node_idx = []

    ind_of_nodes = 0
    pos_now = 0
    match_num = 0
    for node in nodes_without_no:
        # 结点分词结果
        tokenized_nodes = bert_tokenizer.encode(node)[1:-1]

        # 先映射[PAD]
        if tokenized_nodes == 0:
            pos_list.append([len(amr_seq_ids)-1])  # 哪个amr序列的词对结点中目前索引的词有贡献
            pos_weight.append(1)  # 权重
            node_idx.append(ind_of_nodes)  # 被映射到结点列表的第几个词
        else:
            for index in range(len(amr_seq_ids)):
                if (amr_seq_ids[index:index + len(tokenized_nodes)] == tokenized_nodes) and (used_dict[index] == 0):
                    for i in range(index, index + len(tokenized_nodes)):
                        used_dict[i] = 1
                    pos = list(range(index, index + len(tokenized_nodes)))  # pos 是在这个索引的词对目前的结点有贡献
                    # print('结点:', tokenized_nodes, '位置:', pos)
                    # 从 len(amr_seq_ids) 个词 变成 len(nodes_without_no) 个词
                    # 那么，变化的矩阵应该是 len(nodes_without_no) * len(amr_seq_ids) 形状
                    percentage = 1 / len(pos)

                    pos_list.append(pos)  # 哪个amr序列的词对结点中目前索引的词有贡献
                    pos_weight.append(percentage)  # 权重
                    node_idx.append(ind_of_nodes)  # 被映射到结点列表的第几个词

                    pos_now = pos[-1] + 1

                    match_num += 1

                    # print('匹配：', node)

                    break

                else:
                    if used_dict[index] == 0:
                        pos_list.append([index])  # 哪个amr序列的词对结点中目前索引的词有贡献
                        pos_weight.append(0)  # 权重
                        node_idx.append(None)  # 被映射到结点列表的第几个词
                        used_dict[index] = 1

        ind_of_nodes += 1

    return match_num, pos_list, pos_weight, node_idx


if __name__ == '__main__':
    file_names = ['no_bug/complex_sentences.txt']

    with open('no_bug/complex_tokenize.pkl', mode='rb') as file:
        t_list = pickle.load(file)

    with open('no_bug/nodes_washed.pkl', mode='rb') as file2:
        nodes_list = pickle.load(file2)

    with open('no_bug/amr_seq_list.pkl', mode='rb') as file3:
        amr_list = pickle.load(file3)

    with open('no_bug/edge_index_list.pkl', mode='rb') as file3:
        edge_index_list = pickle.load(file3)

    mat_list = []
    for file_name in file_names:

        train_file_path = file_name
        f = open(train_file_path, mode='r', encoding='utf-8').readlines()

        nodes_num_list = []  # 最大140
        edge_index_source_num_list = []  # 最大142
        edge_index_target_num_list = []  # 最大142
        sentence_length_list = []

        max_nodes_num = 150
        max_edges_num = 150

        max_length = 512  # amr_seq 最大长度

        bug_index = []

        for index in range(len(f)):
            try:
                print(file_name, index)

                sentence = f[index]
                sentence = sentence.replace('\n', '')

                tokenize_result = t_list[index]

                # 序列化amr
                amr_seq = [amr_list[index]]

                nodes_without_no = nodes_list[index]

                bert_tokenizer = BertTokenizer.from_pretrained('../../../../downloaded_models/mc_bert_tokenizer')
                pretrained_model = BertModel.from_pretrained('../../../../downloaded_models/mc_bert_model')

                data_sentence_c = bert_tokenizer.batch_encode_plus(amr_seq,
                                                              max_length=max_length,
                                                              truncation=True,
                                                              return_tensors='pt',
                                                              padding='max_length')

                ids = data_sentence_c['input_ids']

                # 分词后的索引
                amr_ids = ids[0].tolist()

                # 编码句子
                src_amr_seq = pretrained_model(torch.tensor(amr_ids).unsqueeze(0)).last_hidden_state

                match_num, pos_list, pos_weight, node_idx = alignment_one_sentence(amr_ids, nodes_without_no, bert_tokenizer)

                list_for_tensor_this_sentence = []
                for i in range(len(ids)):
                    for pos, weight, idx in zip(pos_list, pos_weight, node_idx):
                        if idx is None:
                            list_for_tensor_this_sentence.append([0] * max_nodes_num)
                            continue
                        else:
                            for _ in pos:
                                raw_list = [0] * max_nodes_num
                                raw_list[idx] = weight
                                list_for_tensor_this_sentence.append(raw_list)

                # pad
                while len(list_for_tensor_this_sentence) < max_length:
                    list_for_tensor_this_sentence.append([0] * max_nodes_num)

                mat = torch.tensor(list_for_tensor_this_sentence, dtype=torch.float32)  # 这个要保存
                mat = mat.t()
                edge_index = edge_index_list[index]

                graph_model = graph_emb()
                graph_model(mat @ src_amr_seq, edge_index)

                mat_list.append(mat)

                with open('mat_list.pkl', mode='wb') as file_1:
                    pickle.dump(mat_list, file_1)
            except:
                bug_index.append([index])
                continue

        print(bug_index)

# bugs = [[169], [170], [171], [172], [173], [672], [997], [1828], [1829], [2047], [2049], [2405], [3018], [3719], [4278], [4279], [4614], [5528], [5583], [5917], [6374], [7354], [7428], [7888]]
