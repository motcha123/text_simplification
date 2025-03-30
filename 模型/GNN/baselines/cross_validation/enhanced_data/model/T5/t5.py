import sys
import sentencepiece
import pickle
import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer
#  T5Tokenizer,


tokenizer = AutoTokenizer.from_pretrained("t5_tokenizer")

with open('complex_sentence_list.pkl', mode='rb') as file:
    complex_sentences = pickle.load(file)

with open('simple_sentence_list.pkl', mode='rb') as file:
    simple_sentences = pickle.load(file)

dataset = {'train':[], 'validation':[], 'test':[]}
for i in range(len(complex_sentences)):
    contents = "文本简化：" + complex_sentences[i]
    inputs = tokenizer(contents, max_length=512, truncation=True)
    labels = tokenizer(simple_sentences[i], max_length=512, truncation=True)
    inputs['labels'] = labels['input_ids']

    if i < 2560:
        dataset['train'].append(inputs)
    elif i < 2880:
        dataset['validation'].append(inputs)
    else:
        dataset['test'].append(inputs)


model = AutoModelForSeq2SeqLM.from_pretrained("t5_model")
# model = T5ForConditionalGeneration.from_pretrained("t5_model")

# 统计参数量，以万为单位
print(sum(i.numel() for i in model.parameters()) / 10000)

import numpy as np
from rouge_chinese import Rouge

rouge = Rouge()

def compute_metric(evalPred):
    predictions, labels = evalPred
    decode_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decode_preds = [" ".join(p) for p in decode_preds]
    decode_labels = [" ".join(l) for l in decode_labels]
    scores = rouge.get_scores(decode_preds, decode_labels, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }


args = Seq2SeqTrainingArguments(
    output_dir="./summary",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="rouge-l",
    predict_with_generate=True,
    num_train_epochs=200,
    save_total_limit=3,
)

trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metric,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)

trainer.train()

from transformers import pipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device='cuda')
result = pipe("文本简化：血友病(Hemophilia)是一组因遗传性凝血活酶生成障碍引起的出血性疾病，临床上分为血友病A、血友病B和遗传性XI因子缺乏症三型，其中以血友病A，即凝血因子VIII缺乏症最为常见。", max_length=512)
print(result)