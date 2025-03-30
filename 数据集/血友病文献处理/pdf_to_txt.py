# 目前这个py文件的功能是将所有的pdf转化为txt，且一行一个句子

import sys

import fitz
from bs4 import BeautifulSoup
import re
import os
import random


sentences_write = 0


def pdf_process(pdf_path, buffer_path, txt_path, pdf_name):
    """
    用于单个 pdf 文件处理，将pdf中截取获得的句子写入txt文件
    在每个pdf中随机挑选至多30个句子
    :param pdf_path: pdf 文件路径
    :param buffer_path: 用于过渡的文件路径，作用是存储当前 pdf 中的文字
    :param txt_path: 写入的 txt 文件路径
    :return: None
    """
    with open(buffer_path, 'w', encoding='utf-8') as buffer_file:
        buffer_file.truncate()

    global sentences_write
    i = 0
    # pdf处理
    doc = fitz.open(pdf_path)
    html_content = ''
    for page in doc:
        html_content += page.get_text('html')
    html_content += '</body></html>'
    with open('test.html', mode='w', encoding='utf-8') as h:
        h.write(html_content)

    html_file = open('test.html', 'r', encoding='utf-8')
    htmlhandle = html_file.read()
    soup = BeautifulSoup(htmlhandle, 'html.parser')
    for div in soup.find_all('div'):
        for p in div:
            text = str()
            for span in p:
                p_info = '<span .*?>(.*?)</span>'
                res = re.findall(p_info, str(span))
                if len(res) == 0:
                    pass
                else:
                    text += res[0]
            with open(buffer_path, 'a', encoding='utf-8') as text_file:
                text_file.write(text)
                text_file.write('\n')
    text_file.close()

    # 句子切分
    origin_text = open(buffer_path, 'r', encoding='utf-8').read()
    sentences = re.split('。|？|！', origin_text.replace('\n', '').strip())

    # 向txt中写入句子
    txt_path = r'E:\毕业论文\知网血友病文献\txt'
    with open(txt_path + '\\' + pdf_name.split('.')[0]+'.txt', 'w',  encoding='utf-8') as h:
        pass
    save_file = open(txt_path + '\\' + pdf_name.split('.')[0]+'.txt', 'a',  encoding='utf-8')
    print(pdf_name)
    for sentence in sentences:
        save_file.write(sentence+'\n')
        save_file.flush()
    save_file.close()


if __name__ == '__main__':
    # 把所有pdf文献中的句子进行拆分
    with open('origin.txt', 'w', encoding='utf-8') as o:
        o.truncate()

    file_folder_path = 'E:\毕业论文\知网血友病文献\期刊论文'
    buffer_path = 'test.txt'
    for pdf_name in os.listdir('E:\毕业论文\知网血友病文献\期刊论文'):
        pdf_path = file_folder_path + '\\' + pdf_name
        pdf_process(pdf_path, buffer_path, 'origin.txt', pdf_name)

    # i = 0
    # s = open('random_selected_lines.txt', 'a', encoding='utf-8')
    # with open('origin.txt', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         if random.randint(0, 10) == 8:
    #             s.write(line)
    #             i += 1
    #             print(i)
    #             if i == 10000:
    #                 sys.exit()
