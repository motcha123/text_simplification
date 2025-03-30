"""
本文件用于为一个数据集计算各种指标、分布
并输出各种指标的雷达图、分布图
数据和将分别被保存在两个文件夹中
数据被保存为一个 excel，图被保存为 jpeg
"""

import sys
import pandas as pd
from matplotlib import pyplot as plt
from difficulty_dict_construct import diff_dict_cons
import spacy
import hanlp
import math
import re


class Analyzer:

    def __init__(self, name, save_path, complex_sentences, simple_sentences_list):
        self.name = name  # 给数据集起个名字，将用于存储文件夹的命名
        self.save_path = save_path  # 存储路径
        self.complex_sentences = complex_sentences  # 复杂句子
        self.simple_sentences_list = simple_sentences_list  # 简单句子列表，这是一个二维列表
        self.sentences_num = len(self.complex_sentences)  # 复杂句子个数
        self.simple_sentences_group_num = len(self.simple_sentences_list)  # 有几组简单句子

        # 用于存储结果的df结构
        self.save_df = pd.DataFrame(columns=[])

        # 粗粒度分词器
        self.tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

    def length_cal(self):
        """
        传入数据集，返回句子字数比值分布
        """
        length_ratio_list = []
        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]
            length_ratio_group_list = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                length_ratio = len(sentence_s) / len(sentence_c)
                length_ratio_group_list.append(length_ratio)
            group_mean = sum(length_ratio_group_list) / len(length_ratio_group_list)
            length_ratio_list.append(group_mean)

        mean_value = sum(length_ratio_list) / len(length_ratio_list)
        print(self.name, '句子字数比值', mean_value)
        # plt.hist(length_ratio_list)
        # plt.show()

        return length_ratio_list, mean_value

    def hsk_mean_level(self):
        """
        计算平均词汇难度比值，不在hsk词表中的不参与计算
        """
        # 单任务模型，精度更高
        HanLP = hanlp.pipeline() \
            .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
            .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
            .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
            .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
            .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok') \
            .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')

        diff_dict = diff_dict_cons()

        word_level_ratio_list = []

        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]
            score_c = 0
            word_num_c = 0
            tokenized_result = HanLP(sentence_c.strip())["tok"][0]
            for word in tokenized_result:
                if word in diff_dict.keys():
                    score_c += diff_dict[word]
                    word_num_c += 1
            if score_c != 0:
                sentence_c_hsk_level = score_c / word_num_c
            else:
                continue

            hsk_ratio_group_list = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                score_s = 0
                word_num_s = 0
                tokenized_result = HanLP(sentence_s.strip())["tok"][0]
                for word in tokenized_result:
                    if word in diff_dict.keys():
                        score_s += diff_dict[word]
                        word_num_s += 1
                if word_num_s != 0:
                    sentence_s_hsk_level = score_s / word_num_s
                    hsk_ratio_group_list.append(sentence_s_hsk_level)
            if len(hsk_ratio_group_list) == 0:
                continue
            else:
                sentence_s_mean_hsk = sum(hsk_ratio_group_list) / len(hsk_ratio_group_list)
                word_level_ratio_list.append(sentence_s_mean_hsk / sentence_c_hsk_level)

        mean_value = sum(word_level_ratio_list) / len(word_level_ratio_list)
        print(self.name, 'hsk等级', mean_value)
        # plt.hist(word_level_ratio_list)
        # plt.show()

    def tree_depth_cal(self):
        """
        用于计算依赖树深度比值
        """

        nlp = spacy.load("zh_core_web_trf")

        tree_depth_ratio_list = []

        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]

            doc = nlp(sentence_c)
            max_depth_c = 0
            for token in doc:
                ancestors = [ancestors for ancestors in token.ancestors]
                depth = len(ancestors)
                if depth > max_depth_c:
                    max_depth_c = depth

            tree_depth_group_list = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                doc = nlp(sentence_s)
                max_depth_s = 0
                for token in doc:
                    ancestors = [ancestors for ancestors in token.ancestors]
                    depth = len(ancestors)
                    if depth > max_depth_s:
                        max_depth_s = depth
                tree_depth_group_list.append(max_depth_s)

            mean_simple_depth = sum(tree_depth_group_list) / len(tree_depth_group_list)

            tree_depth_ratio_list.append(mean_simple_depth / max_depth_c)

        mean_value = sum(tree_depth_ratio_list) / len(tree_depth_ratio_list)
        print(self.name, '依赖树深度', mean_value)
        # plt.hist(tree_depth_ratio_list)
        # plt.show()

    def word_statics(self):
        """
        统计词汇的添加、删除和重排序的比例
        """
        delete_list = []
        add_list = []
        reorder_list = []
        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]
            tokenize_result_c = self.tok([sentence_c])

            while '，' in tokenize_result_c[0]:
                tokenize_result_c[0].remove('，')
            while '。' in tokenize_result_c[0]:
                tokenize_result_c[0].remove('。')
            while '、' in tokenize_result_c[0]:
                tokenize_result_c[0].remove('、')

            word_sentence_c_num = len(tokenize_result_c[0])  # 复杂句子的词汇数量

            delete_group_list = []
            add_group_list = []
            reorder_group_list = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                tokenize_result_s = self.tok([sentence_s])

                while '，' in tokenize_result_s[0]:
                    tokenize_result_s[0].remove('，')
                while '。' in tokenize_result_s[0]:
                    tokenize_result_s[0].remove('。')
                while '、' in tokenize_result_s[0]:
                    tokenize_result_s[0].remove('、')

                delete_num = 0
                add_num = 0
                reorder_num = 0

                # 删除数量
                for word_c in tokenize_result_c[0]:
                    if word_c not in tokenize_result_s[0]:
                        delete_num += 1
                # 删除比例
                delete_ratio = delete_num / word_sentence_c_num
                delete_group_list.append(delete_ratio)

                # 增加数量
                for word_s in tokenize_result_s[0]:
                    if word_s not in tokenize_result_c[0]:
                        add_num += 1
                # 增加比例
                word_add_ratio = add_num / word_sentence_c_num
                add_group_list.append(word_add_ratio)

                # 重排数量
                for word_c in tokenize_result_c[0]:
                    if word_c in tokenize_result_s[0] and tokenize_result_c[0].index(word_c) != tokenize_result_s[0].index(
                            word_c):
                        reorder_num += 1
                # 重排比例
                word_reorder_ratio = reorder_num / word_sentence_c_num
                reorder_group_list.append(word_reorder_ratio)

            delete_list.append(sum(delete_group_list) / len(delete_group_list))
            add_list.append(sum(add_group_list) / len(add_group_list))
            reorder_list.append(sum(reorder_group_list) / len(reorder_group_list))

        mean_delete_num = sum(delete_list) / len(delete_list)
        mean_add_num = sum(add_list) / len(add_list)
        mean_reorder_num = sum(reorder_list) / len(reorder_list)

        print(self.name, 'add_mean', mean_add_num)
        print(self.name, 'delete_mean', mean_delete_num)
        print(self.name, 'reorder_mean', mean_reorder_num)

        # plt.hist(delete_list)
        # plt.show()
        # plt.hist(add_list)
        # plt.show()
        # plt.hist(reorder_list)
        # plt.show()

    def familiar_words(self):
        """
        统计词汇熟悉度，用词频代替，只统计在词频表中的词汇
        """
        # 读取词频表
        word_frequency_pd = pd.read_excel("word_frequency_nation.xlsx")
        word_frequency_list = word_frequency_pd['词语'].tolist()

        # 构建词频字典
        word_frequency_pd = pd.read_excel('word_frequency_nation.xlsx')
        word_frequency_dict = {}
        for index in range(len(word_frequency_pd)):
            word_frequency_dict[word_frequency_pd.loc[index, '词语']] = word_frequency_pd.loc[index, '频率（%）']

        familiar_words_ratio_list = []
        for index in range(self.sentences_num):
            minus = 0  # 记录有几个简单句子没有可统计的词

            sentence_c = self.complex_sentences[index]
            tokenize_result_c = self.tok([sentence_c])[0]

            sentence_word_num_c = 0
            sentence_frequency_sum_c = 0
            for word in tokenize_result_c:
                if word in word_frequency_list:
                    sentence_word_num_c += 1
                    sentence_frequency_sum_c += word_frequency_dict[word]
            if sentence_frequency_sum_c == 0:
                continue
            complex_word_mean = sentence_frequency_sum_c / sentence_word_num_c

            familiar_words_simple_group_list = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                tokenize_result_s = self.tok([sentence_s])[0]

                sentence_frequency_sum_s = 0
                sentence_word_num_s = 0
                for word in tokenize_result_s:
                    if word in word_frequency_list:
                        sentence_word_num_s += 1
                        sentence_frequency_sum_s += word_frequency_dict[word]
                if sentence_word_num_s == 0:
                    minus += 1
                    continue
                else:
                    familiar_words_simple_group_list.append(sentence_frequency_sum_s / sentence_word_num_s)
            familiar_words_simple_mean = sum(familiar_words_simple_group_list) / (len(familiar_words_simple_group_list) - minus)

            familiar_words_ratio_list.append(familiar_words_simple_mean / complex_word_mean)

        mean_value = sum(familiar_words_ratio_list) / len(familiar_words_ratio_list)
        print(self.name, '词汇熟悉度', mean_value)
        # plt.hist(familiar_words_ratio_list)
        # plt.show()

    def entropy_pos(self):
        """
        计算词性熵的比值
        """

        pos_parser = hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)

        entropy_pos_ratio_list = []
        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]
            tokenize_result_c = self.tok([sentence_c])[0]

            # while '，' in tokenize_result_c:
            #     tokenize_result_c.remove('，')
            # while '。' in tokenize_result_c:
            #     tokenize_result_c.remove('，')
            # while '、' in tokenize_result_c:
            #     tokenize_result_c.remove('，')

            pos_tagging_result = pos_parser(tokenize_result_c)
            pos_tagging_set = set(pos_tagging_result)
            pos_tagging_dict = {}  # 记录不同pos的频率
            entropy_pos_c = 0  # 这一个句子的pos熵
            for pos in pos_tagging_set:
                pos_tagging_dict[pos] = pos_tagging_result.count(pos) / len(pos_tagging_result)
                entropy_pos_c += -(pos_tagging_dict[pos]) * math.log(pos_tagging_dict[pos], 2)

            entropy_pos_group_s = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                tokenize_result = self.tok([sentence_s])[0]

                # while '，' in tokenize_result:
                #     tokenize_result.remove('，')
                # while '。' in tokenize_result:
                #     tokenize_result.remove('，')
                # while '、' in tokenize_result:
                #     tokenize_result.remove('，')

                pos_tagging_result = pos_parser(tokenize_result)
                pos_tagging_set = set(pos_tagging_result)
                pos_tagging_dict = {}  # 记录不同pos的频率
                entropy_pos_s = 0  # 这一个句子的pos熵
                for pos in pos_tagging_set:
                    pos_tagging_dict[pos] = pos_tagging_result.count(pos) / len(pos_tagging_result)
                    entropy_pos_s += -(pos_tagging_dict[pos]) * math.log(pos_tagging_dict[pos], 2)
                entropy_pos_group_s.append(entropy_pos_s)
            entropy_pos_mean_s = sum(entropy_pos_group_s) / len(entropy_pos_group_s)

            entropy_pos_ratio_list.append(entropy_pos_mean_s / entropy_pos_c)

        mean_value = sum(entropy_pos_ratio_list) / len(entropy_pos_ratio_list)
        print(self.name, 'POS熵', mean_value)
        # plt.hist(entropy_pos_ratio_list)
        # plt.show()

    def entropy(self):
        """
        分别基于字和词统计熵
        """

        entropy_char_ratio_list = []
        entropy_word_ratio_list = []

        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]

            # 基于字的统计-复杂句子
            char_freq_dict = {}  # 记录形符的频率
            char_set = set(sentence_c)  # 形符
            char_num = len(sentence_c)  # 句子字数
            for char in char_set:
                char_freq_dict[char] = sentence_c.count(char) / char_num

            if '，' in char_freq_dict.keys():
                del char_freq_dict['，']
            if '、' in char_freq_dict.keys():
                del char_freq_dict['、']
            if '。' in char_freq_dict.keys():
                del char_freq_dict['。']

            # 以字为单位计算信息熵
            entropy_char_c = 0
            for char in char_freq_dict.keys():
                entropy_char_c += -(char_freq_dict[char]) * math.log(char_freq_dict[char], 2)

            # 基于词的统计-复杂句子
            tokenize_result = self.tok([sentence_c])[0]

            while '，' in tokenize_result:
                tokenize_result.remove('，')
            while '。' in tokenize_result:
                tokenize_result.remove('。')
            while '、' in tokenize_result:
                tokenize_result.remove('、')

            word_freq_dict = {}
            word_set = set(tokenize_result)  # 形符
            word_num = len(tokenize_result)
            for word in word_set:
                word_freq_dict[word] = sentence_c.count(word) / word_num
            # 以词为单位计算信息熵
            entropy_word_c = 0
            for word in word_set:
                entropy_word_c += -(word_freq_dict[word]) * math.log(word_freq_dict[word], 2)

            entropy_char_group_s = []
            entropy_word_group_s = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]

                # 基于字的统计-简单句子
                char_freq_dict = {}  # 记录形符的频率
                char_set = set(sentence_s)  # 形符
                char_num = len(sentence_s)  # 句子字数
                for char in char_set:
                    char_freq_dict[char] = sentence_s.count(char) / char_num

                if '，' in char_freq_dict.keys():
                    del char_freq_dict['，']
                if '、' in char_freq_dict.keys():
                    del char_freq_dict['、']
                if '。' in char_freq_dict.keys():
                    del char_freq_dict['。']

                entropy_char_s = 0
                for char in char_freq_dict.keys():
                    entropy_char_s += -(char_freq_dict[char]) * math.log(char_freq_dict[char], 2)
                entropy_char_group_s.append(entropy_char_s)

                # 基于词的统计-简单句子
                tokenize_result = self.tok([sentence_s])[0]

                while '，' in tokenize_result:
                    tokenize_result.remove('，')
                while '。' in tokenize_result:
                    tokenize_result.remove('。')
                while '、' in tokenize_result:
                    tokenize_result.remove('、')

                word_freq_dict = {}
                word_set = set(tokenize_result)  # 形符
                word_num = len(tokenize_result)
                for word in word_set:
                    word_freq_dict[word] = sentence_s.count(word) / word_num
                # 以词为单位计算信息熵
                entropy_word_s = 0
                for word in word_set:
                    entropy_word_s += -(word_freq_dict[word]) * math.log(word_freq_dict[word], 2)
                entropy_word_group_s.append(entropy_word_s)
            entropy_char_group_s_mean = sum(entropy_char_group_s) / len(entropy_char_group_s)
            entropy_word_group_s_mean = sum(entropy_word_group_s) / len(entropy_word_group_s)

            if entropy_char_c != 0:
                entropy_char_ratio_list.append(entropy_char_group_s_mean / entropy_char_c)
            if entropy_word_c != 0:
                entropy_word_ratio_list.append(entropy_word_group_s_mean / entropy_word_c)

        mean_value_char = sum(entropy_char_ratio_list) / len(entropy_char_ratio_list)
        mean_value_word = sum(entropy_word_ratio_list) / len(entropy_word_ratio_list)
        print(self.name, '字熵', mean_value_char)
        print(self.name, '词熵', mean_value_word)
        # plt.hist(entropy_char_ratio_list)
        # plt.show()
        # plt.hist(entropy_word_ratio_list)
        # plt.show()

    def dot_cal(self):
        """
        计算标点符号 、，。 个数比值，同时计算以标点符号分割的短句的长度比值
        """

        count_dot_ratio = []
        length_short_ratio = []

        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]

            count_dot_complex = sentence_c.count('、') + sentence_c.count('。') + sentence_c.count('，')
            if count_dot_complex == 0:
                count_dot_complex = 1

            short_complex_sentences = re.split('，|。|、', sentence_c)
            if '' in short_complex_sentences:
                short_complex_sentences.remove('')

            length = 0
            num = 0
            for short_complex_sentence in short_complex_sentences:
                length += len(short_complex_sentence)
                num += 1
            complex_length_ave = length / num

            count_dot_simple_group = []
            length_simple_group = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                count_dot_simple = sentence_s.count('、') + sentence_s.count('。') + sentence_s.count('，')
                if count_dot_simple == 0:
                    count_dot_simple = 1
                count_dot_simple_group.append(count_dot_simple)

                short_simple_sentences = re.split('，|。|、', sentence_s)
                if '' in short_simple_sentences:
                    short_simple_sentences.remove('')

                length = 0
                num = 0
                for short_simple_sentence in short_simple_sentences:
                    length += len(short_simple_sentence)
                    num += 1
                simple_length_ave = length / num

                length_simple_group.append(simple_length_ave)
            count_dot_simple_group_mean = sum(count_dot_simple_group) / len(count_dot_simple_group)
            length_simple_group_mean = sum(length_simple_group) / len(length_simple_group)

            count_dot_ratio.append(count_dot_simple_group_mean / count_dot_complex)
            length_short_ratio.append(length_simple_group_mean / complex_length_ave)

        count_dot_ratio_mean = sum(count_dot_ratio) / len(count_dot_ratio)
        length_short_ratio_mean = sum(length_short_ratio) / len(length_short_ratio)

        print(self.name, '标点个数', count_dot_ratio_mean)
        print(self.name, '短句长度', length_short_ratio_mean)

        # plt.hist(count_dot_ratio)
        # plt.show()
        # plt.hist(length_short_ratio)
        # plt.show()

    def professional_words(self):
        """
        统计OOV词比例
        """

        # 读取词频表
        word_frequency_pd = pd.read_excel("word_frequency_nation.xlsx")
        word_frequency_list = word_frequency_pd['词语'].tolist()

        oov_ratio_list = []
        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]
            tokenized_result_c = self.tok([sentence_c])[0]
            tokenized_result_c = set(tokenized_result_c)

            while '，' in tokenized_result_c:
                tokenized_result_c.remove('，')
            while '。' in tokenized_result_c:
                tokenized_result_c.remove('。')
            while '、' in tokenized_result_c:
                tokenized_result_c.remove('、')

            oov_complex = 0
            for word in tokenized_result_c:
                if word not in word_frequency_list:
                    oov_complex += 1

            oov_simple_group = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                tokenized_result_s = self.tok([sentence_s])[0]
                tokenized_result_s = set(tokenized_result_s)

                while '，' in tokenized_result_s:
                    tokenized_result_s.remove('，')
                while '。' in tokenized_result_s:
                    tokenized_result_s.remove('。')
                while '、' in tokenized_result_s:
                    tokenized_result_s.remove('、')

                oov_simple = 0
                for word in tokenized_result_s:
                    if word not in word_frequency_list:
                        oov_simple += 1
                oov_simple_group.append(oov_simple)

            oov_simple_result = sum(oov_simple_group) / len(oov_simple_group)
            if oov_complex != 0:
                oov_ratio_list.append(oov_simple_result / oov_complex)

        print(self.name, 'oov比值', sum(oov_ratio_list) / len(oov_ratio_list))




    def analyze(self):
        # self.length_cal()
        # self.hsk_mean_level()
        # self.tree_depth_cal()
        self.word_statics()
        # self.familiar_words()
        # self.entropy_pos()
        # self.entropy()
        # self.dot_cal()
        # self.professional_words()


if __name__ == '__main__':
    name = 'My_Dataset'
    save_path = ''
    excel = "dataset.xlsx"
    excel = pd.read_excel(excel)
    complex_sentences = excel['复杂句子']
    simple_sentences = excel['简化结果']
    analyzer = Analyzer(name, save_path, complex_sentences, [simple_sentences])
    analyzer.analyze()

    # name = 'CSS'
    # save_path = ''
    # CSS_dataset = '../CSS/CSS_pre_precessed.xlsx'
    # CSS_dataset = pd.read_excel(CSS_dataset)
    # complex_sentences = CSS_dataset['complex']
    # simple_sentences = [CSS_dataset['simple_1'], CSS_dataset['simple_2']]
    # analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    # analyzer.analyze()

    # name = 'CSS_Wiki'
    # save_path = ''
    # CSS_Wiki_dataset = '../CSS_WIKI/CSSWiki_preprocess.xlsx'
    # CSS_Wiki_dataset = pd.read_excel(CSS_Wiki_dataset)
    # complex_sentences = CSS_Wiki_dataset['复杂句子']
    # simple_sentences = [CSS_Wiki_dataset['简单句子1'], CSS_Wiki_dataset['简单句子2'], CSS_Wiki_dataset['简单句子3']]
    # analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    # analyzer.analyze()
    #
    # name = 'MCTS'
    # save_path = ''
    # mcts_dataset = '../MCTS/MCTS_preprocessed.xlsx'
    # mcts_dataset = pd.read_excel(mcts_dataset)
    # complex_sentences = mcts_dataset['complex']
    # simple_sentences = [mcts_dataset['simple_1'], mcts_dataset['simple_2'], mcts_dataset['simple_3'],
    #                     mcts_dataset['simple_4'], mcts_dataset['simple_5']]
    # analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    # analyzer.analyze()

    name = 'ifly'
    save_path = ''
    ifly_dataset = '用大模型简化复杂句子/ifly.xlsx'
    ifly_dataset = pd.read_excel(ifly_dataset)
    complex_sentences = ifly_dataset['complex']
    simple_sentences = [ifly_dataset['simple']]
    analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    analyzer.analyze()

    name = 'ali_zero'
    save_path = ''
    ali_zero_dataset = '用大模型简化复杂句子/ali_zero_shot.xlsx'
    ali_zero_dataset = pd.read_excel(ali_zero_dataset)
    complex_sentences = ali_zero_dataset['complex']
    simple_sentences = [ali_zero_dataset['simple']]
    analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    analyzer.analyze()

"""
词汇添加比例看上去有点怪，看看要不要调整
标点符号数量要不要改成平均每个字有多少标点符号
"""