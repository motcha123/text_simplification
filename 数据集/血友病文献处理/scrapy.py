import sys
import pandas as pd
import re
import hanlp

# 1. 预处理：去除空格和参考文献
# 2. 选难句：根据HSK等级初步筛选困难句子
# 3. 然后再人工筛选一遍确保句子足够困难，然后为每个句子进行人工简化


def diff_dict_cons():
    """
    用于构建一个字典，key是词汇，value是对应的hsk等级
    :return: hsk等级词汇字典
    """
    f_1 = '../HSK词汇/hsk3.0/1级.txt'
    f_2 = '../HSK词汇/hsk3.0/2级.txt'
    f_3 = '../HSK词汇/hsk3.0/3级.txt'
    f_4 = '../HSK词汇/hsk3.0/4级.txt'
    f_5 = '../HSK词汇/hsk3.0/5级.txt'
    f_6 = '../HSK词汇/hsk3.0/6级.txt'
    f_7 = '../HSK词汇/hsk3.0/7-9级.txt'
    words_1 = open(f_1, 'r', encoding='utf-8').readlines()
    words_2 = open(f_2, 'r', encoding='utf-8').readlines()
    words_3 = open(f_3, 'r', encoding='utf-8').readlines()
    words_4 = open(f_4, 'r', encoding='utf-8').readlines()
    words_5 = open(f_5, 'r', encoding='utf-8').readlines()
    words_6 = open(f_6, 'r', encoding='utf-8').readlines()
    words_7 = open(f_7, 'r', encoding='utf-8').readlines()

    diff_dict = {}
    for word in words_1:
        diff_dict[word.strip()] = 1
    for word in words_2:
        diff_dict[word.strip()] = 2
    for word in words_3:
        diff_dict[word.strip()] = 3
    for word in words_4:
        diff_dict[word.strip()] = 4
    for word in words_5:
        diff_dict[word.strip()] = 5
    for word in words_6:
        diff_dict[word.strip()] = 6
    for word in words_7:
        diff_dict[word.strip()] = 7

    return diff_dict



def diff_cal(HanLP, diff_dict, sentence_list):
    """
    用于计算某个句子的平均hsk得分
    :return:
    """

    score = 0
    sen_num = 0
    diff_sentences_num = 0
    # 目前的句子平均hsk得分计算策略：不在hsk词汇表中的词记0分，只用句子中的hsk词汇计算单个句子的平均单词得分
    # 在所有句子上得出的平均分为 3.199951537130917
    sentence_score_ave_for_words = 0
    for sen in sentence_list:
        sen = str(sen)
        for words in HanLP(sen.strip())["tok"]:
            sentence_score = 0
            len_words = len(words)
            for w in words:
                word_score = diff_dict.get(w, 0)
                sentence_score += diff_dict.get(w, 0)
                if word_score == 0:
                    len_words -= 1
            try:
                sentence_score_ave_for_words = sentence_score / len_words  # 该句子的平均得分
            except ZeroDivisionError:
                sentence_score_ave_for_words = 0
            if sentence_score_ave_for_words > 3.199951537130917:
                diff_sentences_num += 1
            score += sentence_score_ave_for_words
        sen_num += 1
        print('sen_num:', sen_num, 'score:', sentence_score_ave_for_words)
        #print('diff_sentences_num', diff_sentences_num)


if __name__ == '__main__':

    # 以下是计算每个句子的平均hsk难度
    # 目前的策略：只计算hsk词表中有的词的平均难度，否则不计入计算
    file_ready = r'E:\毕业论文\test.xlsx'
    df = pd.read_excel(file_ready)
    sentence_list = df.iloc[:, 0]
    print(sentence_list)

    # 单任务分词模型，精度更高
    HanLP = hanlp.pipeline() \
        .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
        .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
        .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
        .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
        .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok') \
        .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')

    diff_dict = diff_dict_cons()

    diff_cal(HanLP=HanLP, diff_dict=diff_dict, sentence_list=sentence_list)

