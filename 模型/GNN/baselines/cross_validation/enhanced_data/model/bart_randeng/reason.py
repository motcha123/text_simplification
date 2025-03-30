import sys
import pickle
import torch
import datasets
import lawrouge
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


tokenizer=AutoTokenizer.from_pretrained("bart_randeng/tokenizer")


model = AutoModelForSeq2SeqLM.from_pretrained("results_bart_randeng/checkpoint-19200/model")

from transformers import pipeline

with open('complex_sentence_list.pkl', mode='rb') as file:
    complex_sentences = pickle.load(file)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device='cuda')

save_path = 'bart_randeng_result.txt'
save_file = open(save_path, mode='w', encoding='utf-8')

for sentence in complex_sentences[2880:]:
    result = pipe(sentence, max_length=512)
    save_file.write(result[0]['generated_text']+'\n')
    save_file.flush()
    # print(result[0]['generated_text'])
print('完成')