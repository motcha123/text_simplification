import json
import pandas as pd

parsed_df = pd.DataFrame(columns=['complex', 'simple_1', 'simple_2'])

with open('test.json') as f:
    data = f.readline()
    data_parse = json.loads(data)
    index = 0
    for item in data_parse:
        # print(item[0]['source'])
        # print(item[0]['target'])
        # print(item[1]['target'])
        parsed_df.loc[index, 'complex'] = item[0]['source']
        parsed_df.loc[index, 'simple_1'] = item[0]['target'][0]
        parsed_df.loc[index, 'simple_2'] = item[0]['target'][0]
        index += 1

parsed_df.to_excel('CSS_pre_precessed.xlsx', index=False)
