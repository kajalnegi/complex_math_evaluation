#!/usr/bin/env python
# coding: utf-8

import re
import json
import pandas as pd

def clean_column(data, col):
    data[col] = data[col].str.replace('$', '')
    data[col] = data[col].str.replace(',', '')
    data[col] = data[col].str.replace('}', '')
    data[col] = data[col].str.strip()
    data[col] = data[col].str.extract('(-?\d+)')
    return data

def main(input_file):
    df = pd.read_csv(input_file)
    answer_column = 'answer'
    
    df[['calculation','numeric_answer']] = df[answer_column].str.rsplit('####', n=1, expand=True)#[-1]
    for t in df.columns.values:
        if t.endswith('_prompt'):
            df[[f'calculation_{t}', f'numeric_{t}']] = df[t].str.rsplit('####', n=1, expand=True)#[-1]

    for t in df.columns.values:
        if t.startswith('numeric_'):
            clean_column(df, t)
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

