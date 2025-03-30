import os
import pandas


# 本文件用于把拆分后的所有句子写入一个txt中
if __name__ == '__main__':
    with open('all_sentences.txt', 'w', encoding='utf-8') as o:
        o.truncate()
    save_file = open('all_sentences.txt', 'a', encoding='utf-8')
    for pdf_name in os.listdir(r'E:\毕业论文\知网血友病文献\txt'):
        pdf_file = r'E:\毕业论文\知网血友病文献\txt' + '\\' + pdf_name
        f = open(pdf_file, 'r', encoding='utf-8')
        for sentence in f.readlines():
            save_file.write(sentence.rstrip()+'$'+pdf_name+'\n')
        f.close()
