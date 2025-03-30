import sys

import pandas as pd
import re
import random


css_wiki_raw = pd.read_excel('CSSWiki.xls')

css_wiki_complex_sentences = css_wiki_raw['Sentence']
css_wiki_simple_sentences_1 = css_wiki_raw['simp1_rewrite']
css_wiki_simple_sentences_2 = css_wiki_raw['simp2_rewrite']
css_wiki_simple_sentences_3 = css_wiki_raw['Simplification']

# rand_index = random.randint(0, 1600)
# print(rand_index)
# print(css_wiki_complex_sentences[rand_index])
# # rand_index = 205
# pattern = '\|\w*]'
# str = re.sub(pattern, '', css_wiki_simple_sentences_1[rand_index]).replace('[', '').replace(']', '')
# print(str)
# str = re.sub(pattern, '', css_wiki_simple_sentences_2[rand_index]).replace('[', '').replace(']', '')
# print(str)
# str = re.sub(pattern, '', css_wiki_simple_sentences_3[rand_index]).replace('[', '').replace(']', '')
# print(str)

pattern = '\|\w*]'

processed_df = pd.DataFrame(columns=['复杂句子', '简单句子1', '简单句子2', '简单句子3'])

for index in range(len(css_wiki_raw)):
    sentence_complex = css_wiki_complex_sentences[index]
    sentence_simple_1 = re.sub(pattern, '', css_wiki_simple_sentences_1[index]).replace('[', '').replace(']', '')
    if pd.isna(css_wiki_simple_sentences_2[index]):
        sentence_simple_2 = re.sub(pattern, '', css_wiki_raw['simp2_highlight'][index]).replace('[', '').replace(']', '')
    else:
        sentence_simple_2 = re.sub(pattern, '', css_wiki_simple_sentences_2[index]).replace('[', '').replace(']', '')
    sentence_simple_3 = re.sub(pattern, '', css_wiki_simple_sentences_3[index]).replace('[', '').replace(']', '')
    processed_df.loc[index, '复杂句子'] = sentence_complex
    processed_df.loc[index, '简单句子1'] = sentence_simple_1
    processed_df.loc[index, '简单句子2'] = sentence_simple_2
    processed_df.loc[index, '简单句子3'] = sentence_simple_3

print(processed_df.head())

processed_df.to_excel('CSSWiki_preprocess.xlsx', index=False)
