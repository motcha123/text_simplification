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
        self.embedding = nn.Embedding(21128, 768, device='cuda')

        # gcn 模型
        self.gcn_model = graph_emb()
        self.gcn_model = self.gcn_model.to('cuda')

        # transformer模型
        self.transformer = nn.Transformer(d_model=768, num_encoder_layers=6, num_decoder_layers=6,
                                          dim_feedforward=768*4,
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
        srcs_amr_seq = self.embedding(srcs_amr_seq)

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
        complex_sentences = pickle.load(complex_sentence_file)[:2560]

    with open(simple_sentence_file_path, mode='rb') as simple_sentence_file:
        simple_sentences = pickle.load(simple_sentence_file)[:2560]

    # 序列化的AMR
    with open('amr_seq_list.pkl', mode='rb') as file:
        seq_amr = pickle.load(file)[:2560]

    # 用于映射的矩阵
    with open('mat_list.pkl', mode='rb') as file:
        map_mats = pickle.load(file)[:2560]

    # 用于GCN计算的边
    with open('edge_index_list.pkl', mode='rb') as file:
        edge_index_list = pickle.load(file)[:2560]

    # 用于pad结点
    with open('node_pad_list.pkl', mode='rb') as file:
        node_pad_list = pickle.load(file)[:2560]

    for i in range(len(node_pad_list)):
        node_pad_list[i] = torch.tensor(node_pad_list[i])

    # print(len(complex_sentences))
    # print(len(simple_sentences))
    # print(len(seq_amr))
    # print(len(map_mats))
    # print(len(edge_index_list))
    # print(len(node_pad_list))
    # sys.exit()

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
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    model = bert_gcn_trans()

    # model.load_state_dict(torch.load('sentence+emb_gcn_layer3.pth'))

    criteria = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    step = 0

    loss_path = 'loss_emb_sen+emb_gcn.txt'
    loss_file = open(loss_path, mode='w')
    loss_file.close()
    loss_file = open(loss_path, mode='a')

    for i in range(300):
        loss_ave = []
        for idx, data in enumerate(dataloader):
            srcs_sentence, tgts, map_mats, srcs_amr_seq, edge_index_list, amr_attention_mask, tgt_y, node_pads = data
            srcs_sentence = srcs_sentence.cuda(0)
            tgts = tgts.cuda(0)
            map_mats = map_mats.cuda(0)
            srcs_amr_seq = srcs_amr_seq.cuda(0)
            edge_index_list = edge_index_list.cuda(0)
            amr_attention_mask = amr_attention_mask.cuda(0)
            node_pads = node_pads.cuda(0)

            n_tokens = (tgts != 0).sum()

            optimizer.zero_grad()
            # 进行transformer的计算
            out = model(srcs_sentence, tgts, map_mats, srcs_amr_seq, edge_index_list,
                        amr_attention_mask, node_pads)
            # 将结果送给最后的线性层进行预测
            out = model.predictor(out)
            # print(torch.argmax(out, dim=-1))

            tgt_y = tgt_y.cuda(0)
            loss = criteria(out.reshape(-1, 21128), tgt_y.reshape(-1)) / n_tokens
            # print(loss)
            loss_ave.append(loss.item())

            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            step += 1

        torch.save(model.state_dict(), 'sentence+emb_gcn_layer3.pth')  # 保存模型参数
        loss_file.write(str(loss.item()) + '\n')
        loss_file.flush()
        # print('模型保存！！！')
        print(i, sum(loss_ave) / len(loss_ave))

