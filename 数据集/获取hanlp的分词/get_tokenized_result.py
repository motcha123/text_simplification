from hanlp_restful import HanLPClient
import time, pickle

# with open('CHSS_c_washed.txt', model='r', encoding='utf-8') as f:
#     complex_sentence_file = f.readlines()
#
# with open('CHSS_s_washed.txt', model='r', encoding='utf-8') as f:
#     simple_sentence_file = f.readlines()
#
# HanLP = HanLPClient('https://www.hanlp.com/api', auth='Nzc2N0BiYnMuaGFubHAuY29tOlNCQ3haaEEwc09WRDJyc2c=', language='zh')
#
# # result = HanLP.tokenize(text=sentence_list, coarse=True)
#
# result_list = []
#
#
# index = 0
# for sentence in complex_sentence_file:
#     result = HanLP.tokenize(sentence, coarse=True)
#     result_list.append(result[0])
#     with open('tokenized_list.pkl', model='wb') as file:
#         pickle.dump(result_list, file)
#     index += 1
#     print(index, result[0])
#     time.sleep(20)
#
# with open('tokenized_list.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)
#
# print(loaded_data)
# print(loaded_data[0])
# print(loaded_data[1])

# 给 train_dataset_1 重新建立分词表
with open('CHSS_c_washed.txt', mode='r', encoding='utf-8') as f:
    complex_sentence_file = f.readlines()

HanLP = HanLPClient('https://www.hanlp.com/api', auth='Nzc2N0BiYnMuaGFubHAuY29tOlNCQ3haaEEwc09WRDJyc2c=', language='zh')

result = HanLP.tokenize(text=sentence_list, coarse=True)

result_list = []


index = 0
for sentence in complex_sentence_file:
    result = HanLP.tokenize(sentence, coarse=True)
    result_list.append(result[0])
    with open('tokenized_list.pkl', mode='wb') as file:
        pickle.dump(result_list, file)
    index += 1
    print(index, result[0])
    time.sleep(20)

with open('tokenized_list.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)
print(loaded_data[0])
print(loaded_data[1])