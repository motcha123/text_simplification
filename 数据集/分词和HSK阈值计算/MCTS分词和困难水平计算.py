# https://github.com/hankcs/HanLP
# C:\Users\54330\AppData\Roaming\hanlp

import hanlp
from difficulty_dict_construct import diff_dict_cons

# 多任务模型
# HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)  # 世界最大中文语料库
# print(HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。']))

diff_sen_file = 'sentences.txt'  #传入需要计算句子难度的txt文件，一行一个句子
diff_sentences = open(diff_sen_file, 'r', encoding='utf-8').readlines()

# 单任务模型，精度更高
HanLP = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
    .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
    .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
    .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok')\
    .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')

diff_dict = diff_dict_cons()

score = 0
sen_num = 0
# 目前的句子平均hsk得分计算策略：不在hsk词汇表中的词记0分，只用句子中的hsk词汇计算单个句子的平均单词得分
# 在所有句子上得出的平均分为 3.199951537130917
for sen in diff_sentences:
    for words in HanLP(sen.strip())["tok"]:
        sentence_score = 0
        len_words = len(words)
        for w in words:
            word_score = diff_dict.get(w, 0)
            sentence_score += diff_dict.get(w, 0)
            if word_score == 0:
                len_words -= 1
        if len_words == 0:
            continue
        sentence_score_ave_for_words = sentence_score / len_words  # 该句子的平均得分
        score += sentence_score_ave_for_words
    sen_num += 1
    print('sen_num:', sen_num, 'score:', sentence_score_ave_for_words)

print('句子平均得分:', score/len(diff_sentences))  # 所有句子的平均得分
