#  用于构建一个字典，key是词汇，value是对应的hsk等级
def diff_dict_cons():
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


if __name__ == '__main__':
    for i in diff_dict_cons().items():
        print(i)
