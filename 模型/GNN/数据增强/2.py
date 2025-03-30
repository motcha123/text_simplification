from nlpcda import Randomword, Similarword

# # smw = Randomword(create_num=3, change_rate=0.3)  # 实体替换
# smw = Similarword(create_num=3, change_rate=0.3)  # 近义词替换
# rs1 = smw.replace('这是一个实体替换')

path = '../baselines/cross_validation/random_sample_complex_sentences.txt'
with open(path, mode='r', encoding='utf-8') as f:
    sentences = f.readlines()

target_path = '../baselines/cross_validation/random_sample_simple_sentences.txt'
with open(target_path, mode='r', encoding='utf-8') as f:
    target_sentences = f.readlines()

save_path = '../baselines/cross_validation/enhanced_data/with_bug/3倍/enhanced_complex_sentence.txt'
target_save_path = '../baselines/cross_validation/enhanced_data/with_bug/3倍/enhanced_simple_sentence.txt'

f_save = open(save_path, mode='w', encoding='utf-8')
f_target_save = open(target_save_path, mode='w', encoding='utf-8')

for i in range(len(sentences)):
    sentence = sentences[i]
    target_sentence = target_sentences[i]
    smw = Similarword(create_num=3, change_rate=0.3)  # 近义词替换
    rs1 = smw.replace(sentence)

    for s in rs1:
        f_save.write(s+'\n')
        f_target_save.write(target_sentence)
        f_save.flush()
        f_target_save.flush()
