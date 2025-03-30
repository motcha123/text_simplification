import sys

from rouge_chinese import Rouge
import hanlp
import pickle

# 粗粒度分词器
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

rouge = Rouge()

with open('complex_sentence_list.pkl', mode='rb') as f:
    complex_sentences = pickle.load(f)

with open('simple_sentence_list.pkl', mode='rb') as f:
    simple_sentences = pickle.load(f)

with open('result_model_B.txt', mode='r', encoding='utf-8') as f:
    result = f.readlines()

rouge_1_list = []
rouge_2_list = []
rouge_l_list = []
for index in range(len(result)):
    reference = simple_sentences[index+2880]
    reference = tok(reference)
    reference = ' '.join(reference)

    out = result[index]
    out = out.replace('[CLS]', '')
    out = out.replace('[SEP]', '')
    out = out.replace('\n', '')
    out = out.replace(' ', '')
    out = tok(out)
    out = ' '.join(out)

    rouge = Rouge()
    score = rouge.get_scores(out, reference)

    rouge_1_list.append(score[0]['rouge-1']['f'])
    rouge_2_list.append(score[0]['rouge-2']['f'])
    rouge_l_list.append(score[0]['rouge-l']['f'])

print('rouge_1_ave:', sum(rouge_1_list)/len(rouge_1_list))
print('rouge_2_ave:', sum(rouge_2_list)/len(rouge_2_list))
print('rouge_l_ave:', sum(rouge_l_list)/len(rouge_l_list))

