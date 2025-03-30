import sys
import math
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

torch.manual_seed(42)

device = 'cuda'

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
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class CopyTaskModel(nn.Module):

    def __init__(self, d_model=768):
        super(CopyTaskModel, self).__init__()

        # 定义Transformer。超参是我拍脑袋想的
        self.transformer = nn.Transformer(d_model=768, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                                          batch_first=True, activation=F.relu, device=device, dropout=0.3)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(768, 21128, device=device)

        self.embedding = nn.Embedding(21128, 768)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = CopyTaskModel.get_key_padding_mask(src)
        tgt_key_padding_mask = CopyTaskModel.get_key_padding_mask(tgt)

        if device == 'cuda':
            tgt_mask = tgt_mask.cuda(0)
            src_key_padding_mask = src_key_padding_mask.cuda(0)
            tgt_key_padding_mask = tgt_key_padding_mask.cuda(0)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        if device == 'cuda':
            src = src.cuda(0)
            tgt = tgt.cuda(0)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 0] = -torch.inf
        return key_padding_mask

with open('complex_sentence_list.pkl', mode='rb') as file:
    complex_sentences = pickle.load(file)[:2560]

with open('simple_sentence_list.pkl', mode='rb') as file:
    simple_sentences = pickle.load(file)[:2560]

data_sentence_c = tokenizer.batch_encode_plus(complex_sentences,
                                              max_length=512,
                                              truncation=True,
                                              return_tensors='pt',
                                              padding='max_length')

data_sentence_s = tokenizer.batch_encode_plus(simple_sentences,
                                              max_length=512,
                                              truncation=True,
                                              return_tensors='pt',
                                              padding='max_length')

srcs = sentence_input_ids_c = data_sentence_c['input_ids']
tgts = sentence_input_ids_s = data_sentence_s['input_ids']

model = CopyTaskModel()
criteria = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
# model.load_state_dict(torch.load('only_trans.pth'))

class Mydataset(Dataset):
    def __init__(self):
        self.srcs = srcs
        self.tgts = tgts

    def __len__(self):
        return len(srcs)

    def __getitem__(self, item):
        return self.srcs[item], self.tgts[item]

dataset = Mydataset()
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

step = 0

# record_path = 'loss_record.txt'
# with open(record_path, encoding='utf-8', mode='w') as file:
#     pass

# record_file = open(record_path, encoding='utf-8', mode='a')

for i in range(300):
    loss_list = []
    for idx, data in enumerate(dataloader):
        src, tgt_o = data
        tgt = tgt_o[:, :-1]
        tgt_y = tgt_o[:, 1:]
        n_tokens = (tgt_o != 0).sum()

        optimizer.zero_grad()
        # 进行transformer的计算
        out = model(src, tgt)
        # 将结果送给最后的线性层进行预测
        out = model.predictor(out)

        if device == 'cuda':
            tgt_y = tgt_y.cuda(0)
        loss = criteria(out.reshape(-1, 21128), tgt_y.reshape(-1)) / n_tokens
        loss_list.append(loss.item())

        step += 1
        # if step % 100 == 0:
        #     record_file.write(str(loss.item()))
        #     record_file.write('\n')
        #     record_file.flush()

        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
    loss_ave = sum(loss_list) / len(loss_list)
    print(i, loss_ave)
    torch.save(model.state_dict(), 'only_trans.pth')  # 保存模型参数
print("完成")


