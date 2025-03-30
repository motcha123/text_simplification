# 用粗粒度分词计算 BLEU

import pickle
import sys
import hanlp
from nltk.translate.bleu_score import sentence_bleu

# 粗粒度分词器
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

with open('complex_sentence_list.pkl', mode='rb') as f:
    complex_sentences = pickle.load(f)[2880:]

with open('simple_sentence_list.pkl', mode='rb') as f:
    simple_sentences = pickle.load(f)[2880:]

with open('result_model_B.txt', mode='r', encoding='utf-8') as f:
    result = f.readlines()

scores = []
for index in range(len(result)):
    reference = simple_sentences[index]
    reference = tok(reference)

    candidate = result[index]
    candidate = candidate.replace('[CLS]', '')
    candidate = candidate.replace('[SEP]', '')
    candidate = candidate.replace('\n', '')
    candidate = candidate.replace(' ', '')
    candidate = tok(candidate)

    score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))

    scores.append(score)

print('bleu', sum(scores)/len(scores))


