import sys
import sentencepiece
import pickle
import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer
#  T5Tokenizer,


tokenizer = AutoTokenizer.from_pretrained("t5_tokenizer")

model = AutoModelForSeq2SeqLM.from_pretrained("t5_mt_checkpoints/checkpoint-2080/model")


from transformers import pipeline

with open('complex_sentence_list.pkl', mode='rb') as file:
    complex_sentences = pickle.load(file)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device='cuda')

save_path = 't5_mt_result.txt'
save_file = open(save_path, mode='w', encoding='utf-8')

for sentence in complex_sentences[2880:]:
    sentence = "文本简化：" + sentence
    result = pipe(sentence, max_length=512)
    save_file.write(result[0]['generated_text']+'\n')
    save_file.flush()
    # print(result[0]['generated_text'])
print('完成')