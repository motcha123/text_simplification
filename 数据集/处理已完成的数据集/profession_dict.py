import pickle

import pandas as pd
import hanlp


def dataset_origin_dict_construct():
    """
    统计原始数据集的词频，并保存为一个字典
    :return:
    """
    excel = "dataset.xlsx"

    sentences_all = pd.read_excel(excel)
    sentences_simple = sentences_all['简化结果']
    sentences_complex = sentences_all['复杂句子']

    word_all_dict = {}

    # 粗粒度分词器
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

    for sentences in [sentences_simple, sentences_complex]:
        for sentence in sentences:
            tokenize_result = tok([sentence])[0]
            for word in tokenize_result:
                if word in word_all_dict.keys():
                    word_all_dict[word] += 1
                else:
                    word_all_dict[word] = 1

    # 保存原生的数据集词频字典
    dict_save = open('dataset_origin_dict', 'wb')
    pickle.dump(word_all_dict, dict_save)
    dict_save.close()


def dataset_origin_dict_construct_class():
    """
    分别统计原始数据集复杂句、简单句的词频，并分别保存为两个字典
    :return:
    """
    excel = "dataset.xlsx"

    sentences_all = pd.read_excel(excel)
    sentences_simple = sentences_all['简化结果']
    sentences_complex = sentences_all['复杂句子']

    # 粗粒度分词器
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

    word_all_dict = {}
    for sentences in [sentences_simple]:
        for sentence in sentences:
            tokenize_result = tok([sentence])[0]
            for word in tokenize_result:
                if word in word_all_dict.keys():
                    word_all_dict[word] += 1
                else:
                    word_all_dict[word] = 1

    dict_save = open('dataset_origin_dict_simple', 'wb')
    pickle.dump(word_all_dict, dict_save)
    dict_save.close()

    word_all_dict = {}
    for sentences in [sentences_complex]:
        for sentence in sentences:
            tokenize_result = tok([sentence])[0]
            for word in tokenize_result:
                if word in word_all_dict.keys():
                    word_all_dict[word] += 1
                else:
                    word_all_dict[word] = 1
    dict_save = open('dataset_origin_dict_complex', 'wb')
    pickle.dump(word_all_dict, dict_save)
    dict_save.close()


if __name__ == '__main__':
    # 构建并保存数据集原生词频词典
    # dataset_origin_dict_construct()

    # 构建并保存复杂句子和简单句子的词频词典
    # dataset_origin_dict_construct_class()

    # 读取数据集原生词频词典
    # data = open('dataset_origin_dict', 'rb')
    # dataset_origin_dict = pickle.load(data)
    # data.close()

    data = open('dataset_origin_dict_complex', 'rb')
    dataset_origin_dict_complex = pickle.load(data)
    data.close()

    data = open('dataset_origin_dict_simple', 'rb')
    dataset_origin_dict_simple = pickle.load(data)
    data.close()

    # 国家词频字典
    national_words = pd.read_excel("word_frequency_nation.xlsx")
    words = national_words['词语'].tolist()

    for a, b in dataset_origin_dict_simple.items():
        if a not in words:
            print(a, b)
