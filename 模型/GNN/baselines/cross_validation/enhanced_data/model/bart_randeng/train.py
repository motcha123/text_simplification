import sys
import pickle
import torch
import datasets
from rouge_chinese import Rouge
import numpy as np
import json
from typing import List, Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, BertTokenizer,
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          BartForConditionalGeneration)

# pip install transformers[torch]

# 最大输入长度
max_input_length = 512
# 最大输出长度
max_target_length = 512
learning_rate = 1e-04
# 读取数据

with open('complex_sentence_list.pkl', mode='rb') as file:
    complex_sentences = pickle.load(file)

with open('simple_sentence_list.pkl', mode='rb') as file:
    simple_sentences = pickle.load(file)

simplify_dataset = {'data': []}
for i in range(len(complex_sentences)):
    simplify_dataset["data"].append({"title": simple_sentences[i], "content": complex_sentences[i]})

with open('simplify_dataset.json', mode='w', encoding='utf-8') as file:
    json.dump(simplify_dataset, file)

# dataset = load_dataset('json', data_files=r'D:\edge_download\nlpcc2017_clean.json', field='data')
dataset = load_dataset('json', data_files='simplify_dataset.json', field='data')

# 加载tokenizer,中文bart使用bert的tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bart_tokenizer")
# tokenizer = BertTokenizer.from_pretrained("bart_randeng/tokenizer")
tokenizer = AutoTokenizer.from_pretrained("bart_randeng/tokenizer")


# print(tokenizer.decode([-100], skip_special_tokens=True))

# 调整数据格式
def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }


# 将原始数据中的content和title转换为document和summary
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
print(dataset)
# 划分数据集
train_dataset, valid_dataset = dataset.train_test_split(train_size=2560, shuffle=False, seed=42).values()
valid_dataset, test_dataset = valid_dataset.train_test_split(train_size=320, shuffle=False, seed=42).values()
datasets = datasets.DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})
print(datasets["train"][2])
print(datasets["validation"][2])
print(datasets["test"][2])
print("数据转换完毕")


def preprocess_function(examples):
    """
    document作为输入，summary作为标签
    """
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = datasets
tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=True,
                                            remove_columns=["document", "summary", "id"])


# print(tokenized_datasets["train"][2].keys())
# print(tokenized_datasets["train"][2])

def collate_fn(features: Dict):
    batch_input_ids = [torch.LongTensor(feature["input_ids"]) for feature in features]
    batch_attention_mask = [torch.LongTensor(feature["attention_mask"]) for feature in features]
    batch_labels = [torch.LongTensor(feature["labels"]) for feature in features]

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }


# 构建DataLoader来验证collate_fn
dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, batch_size=4, collate_fn=collate_fn)
batch = next(iter(dataloader))
# print(batch)

print("开始模型训练")

model = AutoModelForSeq2SeqLM.from_pretrained("bart_randeng/model")
# 参数量
print(sum(i.numel() for i in model.parameters()) / 10000)

# output = model(**batch) # 验证前向传播
# print(output)
print("加载预训练模型")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 将id解码为文字
    predictions_list = []
    labels_list = []
    for i in range(len(predictions)):
        pred = predictions[i][np.where(predictions[i] != -100)]
        pred = pred.tolist()
        pred = tokenizer.decode(pred, skip_special_tokens=True)
        pred = ' '.join(pred)
        predictions_list.append(pred)

        label = labels[i][np.where(labels[i] != -100)]
        label = label.tolist()
        label = tokenizer.decode(label, skip_special_tokens=True)
        label = ' '.join(label)
        labels_list.append(label)

    # predictions = predictions[np.where(predictions != -100)]
    # labels = labels[np.where(labels != -100)]
    # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # # decoded_preds = tokenizer.decode(predictions, skip_special_tokens=True)
    # # 替换标签中的-100
    # # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)

    # 去掉解码后的空格
    decoded_preds = predictions_list
    decoded_labels = labels_list
    # 分词计算rouge
    # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
    # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
    # 计算rouge
    rouge = Rouge()
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}
    result = {key: value * 100 for key, value in result.items()}
    return result


# 设置训练参数
args = Seq2SeqTrainingArguments(
    output_dir="results_bart_randeng",  # 模型保存路径
    num_train_epochs=300,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=learning_rate,
    warmup_steps=100,
    weight_decay=0.001,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=500,
    save_total_limit=1,
    generation_max_length=512,  # 生成的最大长度
    generation_num_beams=1,  # beam search
    load_best_model_at_end=True,
    metric_for_best_model="rouge-l",
    logging_strategy='epoch',
    save_strategy='epoch',
    eval_strategy='epoch'
)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=collate_fn,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# # 打印验证集上的结果
# print(trainer.evaluate(tokenized_datasets["validation"]))
# # 打印测试集上的结果
# print(trainer.evaluate(tokenized_datasets["test"]))
# # 保存最优模型
# trainer.save_model("results/best")

