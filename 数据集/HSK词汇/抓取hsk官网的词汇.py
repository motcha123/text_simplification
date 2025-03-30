import sys

import requests
import time



header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
          'Referer': 'https://www.chinesetest.cn/'}

url = 'https://api.hskmock.com/mock/word/searchWords'

save_path = 'hsk2.0/六级.txt'
level = 6
pages = 250

i = 129
f = open(save_path, 'a')
while i < pages+1:
    # 网页中的 Content-Type:application/json 所以传参用 json参数
    params = {'level_ids': [level], 'initial': "", 'keyword': "", 'page_num': i, 'page_size': 10}
    response = requests.post(url=url, headers=header, json=params, timeout=(5, 10))
    data = response.json()['data']['list']
    for d in data:
        f.write(d['word']+'\n')
    time.sleep(5)
    print(i)
    i += 1

