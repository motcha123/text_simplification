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
from tqdm import tqdm


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
        # # amr模型
        # self.amr_model = hanlp.load('MRP2020_AMR_ENG_ZHO_XLM_BASE')
        # # 粗粒度分词器
        # self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

        self.conv1 = GCNConv(d_model, d_model)  # 定义输入特征数量和输出特征数量
        self.conv2 = GCNConv(d_model, d_model)
        self.conv3 = GCNConv(d_model, d_model)
        self.linear_layer = Linear(d_model, d_model)

        self.norm1 = LayerNorm(d_model=d_model)
        self.norm2 = LayerNorm(d_model=d_model)
        self.norm3 = LayerNorm(d_model=d_model)

        self.temp = SAGEConv(d_model, d_model)

    def forward(self, graph_x, graph_edge):

        _h = graph_x
        h = self.conv1(graph_x, graph_edge)  # 输入特征和邻接矩阵
        h = h.relu()
        # h = h + _h
        # _h = h
        h = self.conv2(h, graph_edge)
        h = h.relu()
        # h = h + _h
        # _h = h
        h = self.conv3(h, graph_edge)
        h = h.relu()
        h = h + _h
        out = self.linear_layer(h)
        out = global_mean_pool(out, batch=None)

        return h, out

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):  # eps 是数值稳定性，一般是一个很小的常数
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x-mean) / torch.sqrt(var+self.eps)
        out = self.gamma * out + self.beta
        return out




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

sub_dict = {'perspective':'perspective',
             'quant':'数字',
             'day':'天',
             'mode':'语气',
             'arg1':'受事',
             'op4':'并列',
             'instrument':'工具',
             'manner':'方式',
             'decade':'年代',
             'time':'时间',
             'part':'部分',
             'snt1':'句子',
             'snt2':'句子',
             'op3':'并列',
             'topic':'话题',
             'dcopy':'dcopy',
             'value':'值',
             'frequency':'频率',
             'polarity':'极性',
             'poss':'领属',
             'arg2':'间接宾语',
             'arg3':'出发点',
             'domain':'属性',
             'condition':'条件',
             'tense':'时态',
             'duration':'时长',
             'ord':'序数',
             'degree':'程度',
             'name':'名称',
             'year':'年',
             'arg0':'施事',
             'location':'处所',
             'unit':'量词',
             'aspect':'体',
             'purpose':'目的',
             'cost':'花费',
             'arg4':'终点',
             'op6':'并列',
             'age':'年龄',
             'op1':'并列',
             'cunit':'量词',
             'range':'跨度',
             'smood':'',
             'compared-to':'参照物',
             'op5':'并列',
             'cause':'起因',
             'direction':'方向',
             'op2':'并列',
             'example':'例子',
             'li':'数字举例',
             'beneficiary':'受益者',
             'month':'月',
             'and':'并列'}

if __name__ == '__main__':
    HanLP = HanLPClient('https://www.hanlp.com/api', auth='Nzc2N0BiYnMuaGFubHAuY29tOlNCQ3haaEEwc09WRDJyc2c=',
                        language='zh')
    file_names = ['2倍/enhanced_complex_sentence.txt']

    bug_save_file = open('2倍数据增强/before_shuffle/bug.txt', mode='a', encoding='utf-8')

    for file_name in file_names:
        mat_list = []
        edge_index_list = []
        amr_seq_list = []
        node_pads_list = []

        train_file_path = file_name
        f = open(train_file_path, mode='r', encoding='utf-8').readlines()

        nodes_num_list = []  # 最大140
        edge_index_source_num_list = []  # 最大142
        edge_index_target_num_list = []  # 最大142
        sentence_length_list = []

        max_nodes_num = 150
        max_edges_num = 150

        max_length = 512  # amr_seq 最大长度

        bug_list = []

        for index in tqdm(range(len(f))):
            try:
                # print(file_name, index)

                sentence = f[index]
                sentence = sentence.replace('\n', '')

                tokenize_result = HanLP.tokenize(sentence, coarse=True)
                tokenize_result = tokenize_result[0]
                time.sleep(1.2)

                # 序列化amr
                amr_seq = [get_graph_sequence(tokenize_result)]

                for key in list(sub_dict.keys()):
                    if key in amr_seq[0]:
                        amr_seq[0] = amr_seq[0].replace(key, sub_dict[key])

                # amr_seq 需要替换

                nodes_without_no, edge_index_source, edge_index_target = get_graph_edge_new(tokenize_result)

                for i in range(len(nodes_without_no)):
                    if '-of' in nodes_without_no[i]:
                        left_part = nodes_without_no[i].split('-of')[0]
                        if left_part in list(sub_dict.keys()):
                            new_left_part = sub_dict[left_part]
                            nodes_without_no[i] = new_left_part + '-of'
                    elif nodes_without_no[i] in list(sub_dict.keys()):
                        nodes_without_no[i] = sub_dict[nodes_without_no[i]]

                # nodes_without_no 需要替换

                node_pads = [1] * len(nodes_without_no)
                while len(nodes_without_no) < max_nodes_num:
                    nodes_without_no.append('[PAD]')
                    node_pads.append(0)

                if len(nodes_without_no) > 150:
                    continue

                pad_idx = 149
                while len(edge_index_source) < max_edges_num:
                    edge_index_source.append(pad_idx)
                    edge_index_target.append(pad_idx)
                    pad_idx -= 1

                if len(edge_index_source) > 150:
                    continue

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
                edge_index = torch.tensor([edge_index_source, edge_index_target], dtype=torch.int64)  # 这个要保存
                amr_seq = amr_seq[0]  # 这个要保存

                # graph_model = graph_emb()
                # graph_model(mat @ src_amr_seq, edge_index)

                mat_list.append(mat)
                edge_index_list.append(edge_index)
                amr_seq_list.append(amr_seq)
                node_pads_list.append(node_pads)

                with open('2倍数据增强/before_shuffle/mat_list.pkl', mode='wb') as file_1:
                    pickle.dump(mat_list, file_1)
                with open('2倍数据增强/before_shuffle/edge_index_list.pkl', mode='wb') as file_2:
                    pickle.dump(edge_index_list, file_2)
                with open('2倍数据增强/before_shuffle/amr_seq_list.pkl', mode='wb') as file_3:
                    pickle.dump(amr_seq_list, file_3)
                with open('2倍数据增强/before_shuffle/node_pad_list.pkl', mode='wb') as file_1:
                    pickle.dump(node_pads_list, file_1)
            except:
                bug_save_file.write(str(index) + '\n')
                bug_list.append(index)
                print('bug', index)
                continue

        print(bug_list)
        # [68, 69, 107, 333, 692, 736, 764, 826, 926, 1270, 1585, 1725, 1781, 1927, 1975, 2033, 2043, 2061, 2101, 2151, 2193, 2321, 2517, 2571, 2597, 2619, 2681, 2711, 2759, 3021, 3143]
