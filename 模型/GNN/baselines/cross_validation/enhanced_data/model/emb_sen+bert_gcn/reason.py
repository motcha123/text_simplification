import pickle
import math, torch
import sys, os
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gc

torch.manual_seed(42)


class graph_emb(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()

        self.conv1 = GCNConv(d_model, d_model)  # 定义输入特征数量和输出特征数量
        self.conv2 = GCNConv(d_model, d_model)
        self.conv3 = GCNConv(d_model, d_model)
        self.dr1 = nn.Dropout(0.2)
        self.dr2 = nn.Dropout(0.2)
        self.dr3 = nn.Dropout(0.2)

    def forward(self, graph_x, graph_edge):
        h_1 = graph_x
        h = self.conv1(graph_x, graph_edge)  # 输入特征和邻接矩阵
        h = h.relu()
        h_2 = h + h_1
        h_2 = self.dr1(h_2)
        h = self.conv2(h_2, graph_edge)
        h = h.relu()
        h_3 = h + h_2
        h_3 = self.dr2(h_3)
        h = self.conv2(h_3, graph_edge)
        h = h + h_3
        h = self.dr3(h)

        return h


tokenizer = BertTokenizer.from_pretrained('mc_bert_tokenizer')


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        torch.cuda.empty_cache()
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class bert_gcn_trans(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.empty_cache()
        self.pretrained_model = BertModel.from_pretrained('mc_bert_model').to('cuda')

        # gcn 模型
        self.gcn_model = graph_emb()
        self.gcn_model = self.gcn_model.to('cuda')

        # transformer模型
        self.transformer = nn.Transformer(d_model=768, num_encoder_layers=6, num_decoder_layers=6,
                                          dim_feedforward=768 * 4,
                                          batch_first=True, activation=F.relu, device='cuda')

        self.embedding = nn.Embedding(21128, 768).to('cuda')

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(768, dropout=0.2).to('cuda')

        self.predictor = nn.Linear(768, 21128, device='cuda')

        self.fuse_linear_1 = nn.Linear(768, 2048, device='cuda')
        self.fuse_linear_2 = nn.Linear(2048, 768, device='cuda')

    def forward(self, src, tgt, map_mats, srcs_amr_seq, edge_index_list, amr_attention_mask, nodes_padding_list):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        tgt_key_padding_mask = self.get_key_padding_mask(tgt)

        tgt_mask = tgt_mask.cuda(0)
        tgt_key_padding_mask = tgt_key_padding_mask.cuda(0)
        src_key_padding_mask = self.get_key_padding_mask(torch.concat([src, nodes_padding_list], dim=1))
        src_key_padding_mask = src_key_padding_mask.cuda(0)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        srcs_amr_seq = self.pretrained_model(input_ids=srcs_amr_seq,
                                             attention_mask=amr_attention_mask).last_hidden_state

        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        nodes = self.gcn_model(map_mats @ srcs_amr_seq, edge_index_list.permute(1, 0, 2))

        input_emb = torch.concat([src, nodes], dim=1)
        input_emb = self.fuse_linear_1(input_emb)
        input_emb = self.fuse_linear_2(input_emb)

        out = self.transformer(input_emb, tgt, tgt_mask=tgt_mask, memory_mask=None,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=None,
                               tgt_is_causal=True, memory_is_causal=False)

        return out

    def get_key_padding_mask(self, tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 0] = -torch.inf
        return key_padding_mask


if __name__ == '__main__':
    max_len = 512

    complex_sentence_file_path = 'complex_sentence_list.pkl'
    simple_sentence_file_path = 'simple_sentence_list.pkl'

    with open(complex_sentence_file_path, mode='rb') as complex_sentence_file:
        complex_sentences = pickle.load(complex_sentence_file)[2880:]

    with open(simple_sentence_file_path, mode='rb') as simple_sentence_file:
        simple_sentences = pickle.load(simple_sentence_file)[2880:]

    # 序列化的AMR
    with open('amr_seq_list.pkl', mode='rb') as file:
        seq_amr = pickle.load(file)[2880:]

    # 用于映射的矩阵
    with open('mat_list.pkl', mode='rb') as file:
        map_mats = pickle.load(file)[2880:]

    # 用于GCN计算的边
    with open('edge_index_list.pkl', mode='rb') as file:
        edge_index_list = pickle.load(file)[2880:]

    # 用于pad结点
    with open('node_pad_list.pkl', mode='rb') as file:
        node_pad_list = pickle.load(file)[2880:]

    for i in range(len(node_pad_list)):
        node_pad_list[i] = torch.tensor(node_pad_list[i])

    data_sentence_c = tokenizer.batch_encode_plus(complex_sentences,
                                                  max_length=max_len,
                                                  truncation=True,
                                                  return_tensors='pt',
                                                  padding='max_length')

    data_sentence_AMR_seq = tokenizer.batch_encode_plus(seq_amr,
                                                        max_length=max_len,
                                                        truncation=True,
                                                        return_tensors='pt',
                                                        padding='max_length')

    data_sentence_s = tokenizer.batch_encode_plus(simple_sentences,
                                                  max_length=max_len,
                                                  truncation=True,
                                                  return_tensors='pt',
                                                  padding='max_length')

    srcs = data_sentence_c['input_ids']  # 困难句子分词索引
    srcs_amr_seq = data_sentence_AMR_seq['input_ids']  # 序列化的amr对应的分词索引
    tgts = data_sentence_s['input_ids'][:, :-1]  # 简单句子的分词索引
    tgt_y = data_sentence_s['input_ids'][:, 1:]

    amr_attention_mask = data_sentence_AMR_seq['attention_mask']


    class Mydataset(Dataset):
        def __init__(self):
            # 用于transformer的
            self.srcs_sentence = srcs
            self.tgts = tgts
            self.tgt_y = tgt_y

            # 用于GCN的
            self.map_mats = map_mats
            self.srcs_amr_seq = srcs_amr_seq
            self.edge_index_list = edge_index_list
            self.amr_attention_mask = amr_attention_mask
            self.node_pad_list = node_pad_list

        def __len__(self):
            return len(srcs)

        def __getitem__(self, item):
            return (self.srcs_sentence[item], self.tgts[item], self.map_mats[item], self.srcs_amr_seq[item],
                    self.edge_index_list[item], self.amr_attention_mask[item], self.tgt_y[item],
                    self.node_pad_list[item])


    dataset = Mydataset()
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    model = bert_gcn_trans()
    model.load_state_dict(torch.load('emb_sen+bert_gcn_3layer.pth'))

    model.eval()

    save_path = 'result_emb_sen+bert_gcn.txt'
    save_file = open(save_path, mode='w', encoding='utf-8')

    with torch.no_grad():
        # 生成最多max_len个词
        for idx, data in enumerate(dataloader):
            generated = [101]
            srcs_sentence, tgts, map_mats, srcs_amr_seq, edge_index_list, amr_attention_mask, tgt_y, node_pad_list = data
            ori = srcs_sentence.tolist()
            ori = ori[0]
            sentence_origin = tokenizer.decode(ori)
            # print('sentence_origin', sentence_origin)  # 原始句子
            srcs_sentence = srcs_sentence.cuda(0)
            map_mats = map_mats.cuda(0)
            srcs_amr_seq = srcs_amr_seq.cuda(0)
            edge_index_list = edge_index_list.cuda(0)
            amr_attention_mask = amr_attention_mask.cuda(0)
            node_pad_list = node_pad_list.cuda(0)

            for i in range(511):
                sentence = tokenizer.decode(generated)
                tgts = torch.tensor([generated], dtype=torch.long)
                tgts = tgts.cuda(0)

                input_tensor = tgts
                out = model(srcs_sentence, input_tensor, map_mats, srcs_amr_seq, edge_index_list, amr_attention_mask,
                            node_pad_list)
                out = model.predictor(out)
                predicted_idx = torch.argmax(out[:, -1], dim=-1).item()
                generated.append(predicted_idx)

                # 如果生成结束标记，则停止生成
                if predicted_idx == 102:
                    break
            save_file.write(sentence+'\n')
            save_file.flush()
            # print(sentence)
        print('ok')

