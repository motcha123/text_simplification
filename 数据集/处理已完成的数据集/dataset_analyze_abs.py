"""
将绝对指标记录在一个excel中
"""

import sys
import pandas as pd
from matplotlib import pyplot as plt
from difficulty_dict_construct import diff_dict_cons
import spacy
import hanlp
import math
import re
import numpy as np
from tqdm import tqdm


class Analyzer:

    def __init__(self, name, save_path, complex_sentences, simple_sentences_list):
        self.name = name  # 给数据集起个名字，将用于存储文件夹的命名
        print('正在对', self.name, '进行统计')
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
        sentence_c_length = []
        sentence_s_length = []
        for index in range(self.sentences_num):
            sentence_c = self.complex_sentences[index]
            sentence_c_length.append(len(sentence_c))
            sentence_s_length_group = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                sentence_s_length_group.append(len(sentence_s))
            sentence_s_length.append(sum(sentence_s_length_group) / len(sentence_s_length_group))
        self.save_df['c_length'] = sentence_c_length
        self.save_df['s_length'] = sentence_s_length
        self.save_df['length_ratio'] = self.save_df['s_length'] / self.save_df['c_length']
        print('长度计算完成')

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

        hsk_ratio_list = []
        sentence_c_hsk_list = []
        sentence_s_hsk_list = []

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
                self.save_df.loc[index, 'c_hsk']= sentence_c_hsk_level
                self.save_df.loc[index, 's_hsk']= sentence_s_mean_hsk
                self.save_df.loc[index, 'hsk_ratio']= sentence_s_mean_hsk / sentence_c_hsk_level

        print('hsk计算完成')

    def tree_depth_cal(self):
        """
        用于计算依赖树深度比值
        """

        nlp = spacy.load("zh_core_web_trf")

        c_depth_list = []
        s_depth_list = []

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

            c_depth_list.append(max_depth_c)
            s_depth_list.append(mean_simple_depth)
        self.save_df['c_depth'] = c_depth_list
        self.save_df['s_depth'] = s_depth_list
        self.save_df['depth_ratio'] = self.save_df['s_depth'] / self.save_df['c_depth']
        print('依赖树计算完成')


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

        self.save_df['add'] = add_list
        self.save_df['delete'] = delete_list
        self.save_df['reorder'] = reorder_list
        print('词汇操作统计完成')

    def familiar_words(self):
        """
        统计词汇熟悉度，用词频代替，只统计在词频表中的词汇
        """

        c_familiar_list = []
        s_familiar_list = []

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
            self.save_df.loc[index, 'c_familiar'] = complex_word_mean
            self.save_df.loc[index, 's_familiar'] = familiar_words_simple_mean
            self.save_df.loc[index, 'familiar_ratio'] = familiar_words_simple_mean / complex_word_mean


        print('熟悉度统计完成')


    def entropy_pos(self):
        """
        计算词性熵的比值
        """

        pos_parser = hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)

        c_pos_list = []
        s_pos_list = []
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

            c_pos_list.append(entropy_pos_c)
            s_pos_list.append(entropy_pos_mean_s)

        self.save_df['pos_c'] = c_pos_list
        self.save_df['pos_s'] = s_pos_list
        self.save_df['pos_ratio'] = self.save_df['pos_s'] / self.save_df['pos_c']
        print('pos计算完成')

    def entropy(self):
        """
        分别基于字和词统计熵
        """

        c_char_list = []
        c_word_list = []
        s_char_list = []
        s_word_list = []

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
                c_char_list.append(entropy_char_c)
                s_char_list.append(entropy_char_group_s_mean)
            if entropy_word_c != 0:
                c_word_list.append(entropy_word_c)
                s_word_list.append(entropy_word_group_s_mean)
        self.save_df['c_char_entropy'] = c_char_list
        self.save_df['s_char_entropy'] = s_char_list
        self.save_df['char_entropy_ratio'] = self.save_df['s_char_entropy'] / self.save_df['c_char_entropy']

        self.save_df['c_word_entropy'] = c_char_list
        self.save_df['s_word_entropy'] = s_char_list
        self.save_df['word_entropy_ratio'] = self.save_df['s_word_entropy'] / self.save_df['c_word_entropy']

        print('字词熵计算完成')

    def dot_cal(self):
        """
        计算一个句子中，标点符号(，。、！？)数量 / 非标点的字符长度
        如果一个句子结尾不是 。 ！ ？ 则自动补充一个 。
        """

        c_dot_count_per_word_list = []
        s_dot_count_per_word_list = []

        for index in tqdm(range(self.sentences_num)):
            sentence_c = self.complex_sentences[index]
            if not (sentence_c.endswith('。') or sentence_c.endswith('!') or sentence_c.endswith('?')):
                sentence_c = sentence_c + '。'

            count_dot_complex = sentence_c.count('、') + sentence_c.count('。') + sentence_c.count('，') + sentence_c.count('：')

            short_complex_sentences = re.split('，|。|、|：', sentence_c)
            while '' in short_complex_sentences:
                short_complex_sentences.remove('')

            length_c = 0
            for short_complex_sentence in short_complex_sentences:
                length_c += len(short_complex_sentence)

            dot_per_sentence_c = count_dot_complex / length_c

            dot_per_sentence_s_group = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]

                if not (sentence_s.endswith('。') or sentence_s.endswith('!') or sentence_s.endswith('?')):
                    sentence_s = sentence_s + '。'

                count_dot_simple = sentence_s.count('、') + sentence_s.count('。') + sentence_s.count('，') + sentence_c.count('：')

                short_complex_sentences = re.split('，|。|、|：', sentence_s)
                while '' in short_complex_sentences:
                    short_complex_sentences.remove('')

                length_s = 0
                for short_complex_sentence in short_complex_sentences:
                    length_s += len(short_complex_sentence)

                dot_per_sentence_s_group.append(count_dot_simple / length_s)

            dot_per_sentence_s = sum(dot_per_sentence_s_group) / len(dot_per_sentence_s_group)

            c_dot_count_per_word_list.append(dot_per_sentence_c)
            s_dot_count_per_word_list.append(dot_per_sentence_s)

        self.save_df['dot_per_sentence_c'] = c_dot_count_per_word_list
        self.save_df['dot_per_sentence_s'] = s_dot_count_per_word_list
        print('标点和短句统计完成')

    def professional_words(self):
        """
        统计OOV种类词比例
        """

        # 读取词频表
        word_frequency_pd = pd.read_excel("word_frequency_nation.xlsx")
        word_frequency_list = word_frequency_pd['词语'].tolist()

        for index in tqdm(range(self.sentences_num)):
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
            oov_simple_length = []
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
                oov_simple_length.append(len(tokenized_result_s))

            oov_simple_result = sum(oov_simple_group) / len(oov_simple_group)
            simple_sentences_length_ave = sum(oov_simple_length) / len(oov_simple_length)
            self.save_df.loc[index, 'c_oov_type'] = oov_complex  # 复杂句子中有多少不同的oov词汇
            self.save_df.loc[index, 's_oov_type'] = oov_simple_result  # 简单句子中有多少不同的oov词汇
            self.save_df.loc[index, 'c_oov_type_percentage'] = oov_complex / len(tokenized_result_c)  # 复杂句子中有多少不同的oov词汇百分比
            self.save_df.loc[index, 's_oov_type'] = oov_simple_result / simple_sentences_length_ave  # 简单句子中有多少不同的oov词汇百分比


        print(self.name, 'oov种类统计完成')

    def professional_words_num(self):
        """
        统计OOV数量
        """

        # 读取词频表
        word_frequency_pd = pd.read_excel("word_frequency_nation.xlsx")
        word_frequency_list = word_frequency_pd['词语'].tolist()

        for index in tqdm(range(self.sentences_num)):
            sentence_c = self.complex_sentences[index]
            tokenized_result_c = self.tok([sentence_c])[0]

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
            simple_sentence_length = []
            for sentences_s in self.simple_sentences_list:
                sentence_s = sentences_s[index]
                tokenized_result_s = self.tok([sentence_s])[0]

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
                simple_sentence_length.append(len(tokenized_result_s))

            oov_simple_result = sum(oov_simple_group) / len(oov_simple_group)
            simple_length_ave = sum(simple_sentence_length) / len(simple_sentence_length)

            self.save_df.loc[index, 'c_oov_num'] = oov_complex  # 复杂句子中的oov词汇数量
            self.save_df.loc[index, 's_oov_num'] = oov_simple_result  # 简单句子中的oov词汇数量
            self.save_df.loc[index, 'c_oov_num_percentage'] = oov_complex / len(tokenized_result_c)  # 复杂句子中的oov词汇数量比例
            self.save_df.loc[index, 's_oov_num_percentage'] = oov_simple_result / simple_length_ave  # 简单句子中的oov词汇数量比例


        print(self.name, 'oov数量统计完成')


    def analyze(self):
        # self.length_cal()
        # self.hsk_mean_level()
        # self.tree_depth_cal()
        # self.word_statics()
        # self.familiar_words()
        # self.entropy_pos()
        # self.entropy()
        self.dot_cal()
        # self.professional_words()
        # self.professional_words_num()

        self.save_df.to_excel(self.save_path + self.name + '_dot' + '.xlsx', index=False)


if __name__ == '__main__':

    name = 'My_Dataset'
    save_path = 'temp/'
    excel = "dataset.xlsx"
    excel = pd.read_excel(excel)
    complex_sentences = excel['复杂句子']
    simple_sentences = excel['简化结果']
    analyzer = Analyzer(name, save_path, complex_sentences, [simple_sentences])
    analyzer.analyze()

    sys.exit()

    name = 'CSS'
    save_path = 'temp/'
    CSS_dataset = '../CSS/CSS_pre_precessed.xlsx'
    CSS_dataset = pd.read_excel(CSS_dataset)
    complex_sentences = CSS_dataset['complex']
    simple_sentences = [CSS_dataset['simple_1'], CSS_dataset['simple_2']]
    analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    analyzer.analyze()

    name = 'CSS_Wiki'
    save_path = 'temp/'
    CSS_Wiki_dataset = '../CSS_WIKI/CSSWiki_preprocess.xlsx'
    CSS_Wiki_dataset = pd.read_excel(CSS_Wiki_dataset)
    complex_sentences = CSS_Wiki_dataset['复杂句子']
    simple_sentences = [CSS_Wiki_dataset['简单句子1'], CSS_Wiki_dataset['简单句子2'], CSS_Wiki_dataset['简单句子3']]
    analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    analyzer.analyze()

    name = 'MCTS'
    save_path = 'temp/'
    mcts_dataset = '../MCTS/MCTS_preprocessed.xlsx'
    mcts_dataset = pd.read_excel(mcts_dataset)
    complex_sentences = mcts_dataset['complex']
    simple_sentences = [mcts_dataset['simple_1'], mcts_dataset['simple_2'], mcts_dataset['simple_3'],
                        mcts_dataset['simple_4'], mcts_dataset['simple_5']]
    analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    analyzer.analyze()

    name = 'ifly_zero'
    save_path = 'temp/'
    ifly_dataset = '用大模型简化复杂句子/ifly.xlsx'
    ifly_dataset = pd.read_excel(ifly_dataset)
    complex_sentences = ifly_dataset['complex']
    simple_sentences = [ifly_dataset['simple']]
    analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    analyzer.analyze()

    name = 'ali_zero'
    save_path = 'temp/'
    ali_zero_dataset = '用大模型简化复杂句子/ali_zero_shot.xlsx'
    ali_zero_dataset = pd.read_excel(ali_zero_dataset)
    complex_sentences = ali_zero_dataset['complex']
    simple_sentences = [ali_zero_dataset['simple']]
    analyzer = Analyzer(name, save_path, complex_sentences, simple_sentences)
    analyzer.analyze()

