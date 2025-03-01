#!/usr/bin/env python
# coding: utf-8

# In[216]:


import re
import json
import numpy as np
import pandas as pd


def extract_answer(data, column, new_column):
    def find_answer(row):
        
        split1= row[column].rsplit('####', 1)
        split2 = row[column].rsplit('\\boxed{', 1)
        split3 = row[column].rsplit('**Final Answer:**')
        split4 = row[column].rsplit('**Answer:**')
        split5 = row[column].rsplit('**')
        split6 = row[column].rsplit('\nboxed{')
        split7 = row[column].rsplit('\n\n')
        #print(len(split1))
        #print(split2)
        
        if len(split2)>1:
            #print(split2)
            if len(split2[1]) > len(split2[0]):
                return split2[1], split2[0].split('}')[0]
            else:
                return split2[0],split2[1].split('}')[0]
        if len(split6)>1:
            #print(split6)
            if len(split6[1]) > len(split6[0]):
                return split6[1], split6[0].split('}')[0]
            else:
                return split6[0], split6[1].split('}')[0]
        if len(split1)>1: 
            #print(split1)
            if len(split1[0]) > len(split1[1]):
                #print("im here")
                return split1[0], split1[1]
            else:
                return split1[1], split1[0]
        
        if len(split3)>1:
            #print(split3)
            if len(split3[1]) > len(split3[0]):
                return split3[1], split3[0]
            else:
                return split3[0], split3[1]
        if len(split4)>1:
            #print(split4)
            if len(split4[1]) > len(split4[0]):
                return split4[1], split4[0]
            else:
                return split4[0], split4[1]
        if len(split5)>1:
            #print(split5)
            if len(split5[1]) > len(split5[0]):
                return split5[1], split5[0].split('**')[0]
            else:
                return split5[0], split5[1].split('**')[0]
        if len(split7)>1:
            #print(split7)
            if len(split7[1]) > len(split7[0]):
                return split7[1], split7[0]
            else:
                return split7[0], split7[1]
          
        return np.nan, np.nan

    cal_col = 'calculation_'+column
    data[[cal_col, new_column]] = data.apply(find_answer,axis=1, result_type="expand")
    data[new_column] = data[new_column].str.extract('(-?\d+)')
    return data    


def main(input_file):

    df = pd.read_csv(input_file)
    answer_column = 'answer'
    
    df[['calculation','numeric_answer']] = df[answer_column].str.rsplit('####', n=1, expand=True)#[-1]

    for t in df.columns.values:
        if t.endswith('_prompt'):
            
            df = extract_answer(df, t, f'numeric_{t}')


    for t in df.columns.values:
        if t.startswith('numeric_'):
            df[t] = df[t].str.replace('$', '')
            df[t] = df[t].str.replace(',', '')
            df[t] = df[t].str.replace('}', '')
            
            df[t] = df[t].str.split('\\').str[0]
            df[t] = df[t].str.strip()
    out_dict = {}
    for t in df.columns.values:
        if t.startswith('numeric_') and t!= 'numeric_answer':
            df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
            print(t.replace('numeric_', ''))
            print("Matched answers for ", df_.shape[0], "out of ", df.shape[0])
            acc = df_.shape[0]/df.shape[0]
            print("Accuracy on complex GSM8K question: ", round(acc,4)*100)
            out_dict.update({f"{t}":round(acc,4)*100})
    with open("/".join(input_file.split('/')[:-1] + [f"results.json"]), "w") as f:
        f.write(json.dumps(out_dict))
    f.close()


def load_args():
    parser = argparse.ArgumentParser(description="Parsing output for GSM8K")
    parser.add_argument("--input_filepath", required=True,
                        help="input csv filepath with the columns, ['answer', '*_prompt', ]")
    
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = load_args()
    main(
        args['input_filepath'] 
    )

