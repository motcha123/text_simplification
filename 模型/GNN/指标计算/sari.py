# 用粗粒度分词计算 SARI

import sys
import hanlp
import evaluate
import pickle

# 粗粒度分词器
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

sari = evaluate.load("sari")

with open('complex_sentence_list.pkl', mode='rb') as f:
    complex_sentences = pickle.load(f)[2880:]  #[2560:]

with open('simple_sentence_list.pkl', mode='rb') as f:
    simple_sentences = pickle.load(f)[2880:]

with open('result_model_B.txt', mode='r', encoding='utf-8') as f:
    result = f.readlines()

# sources=["Some people really enjoy windowshopping."]
# predictions=["Some birds like the windows."]
#
# references=[["Some people enjoy shopping.",
#              "People to to browse the stores."]]
#
# sari_score = sari.compute(sources=sources,
#                           predictions=predictions,
#                           references=references)

scores = []
for index in range(len(result)):
    reference = tok(simple_sentences[index])

    source = tok(complex_sentences[index])

    prediction = result[index]
    prediction = prediction.replace('[CLS]', '')
    prediction = prediction.replace('[SEP]', '')
    prediction = prediction.replace('\n', '')
    prediction = prediction.replace(' ', '')
    prediction = tok(prediction)

    source = ' '.join(source)
    prediction = ' '.join(prediction)
    reference = ' '.join(reference)

    sari_score = sari.compute(sources=[source],
                              predictions=[prediction],
                              references=[[reference]])

    scores.append(sari_score['sari'])

print(sum(scores)/len(scores))
