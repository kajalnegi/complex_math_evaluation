#!/usr/bin/env python
# coding: utf-8

import re
import argparse
import json
import numpy as np
import pandas as pd

def clean_column(data, col):
    data[col] = data[col].str.replace('$', '')
    data[col] = data[col].str.replace(',', '')
    data[col] = data[col].str.replace('}', '')       
    data[col] = data[col].str.split('\\').str[0]       
    data[col] = data[col].str.strip()
    data[col] = data[col].str.extract('(-?\d+)')

    return data

def dump_results(df, header, output_dir):
    results = pd.DataFrame(columns=['Original', 'BODMAS', 'Inverse', 'Trigonometry'], index=['Generic', 'Notice', '5-CoT', 'Complex 5-CoT'])

    for t in df.columns.values:
        if t.startswith('numeric_') and t!= 'numeric_answer':
            df_ = None
            if t.startswith('numeric_bodmas') and t.endswith('_math_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Generic', 'BODMAS'] = round(acc,3)*100
            elif t.startswith('numeric_bodmas') and t.endswith('_math_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Notice', 'BODMAS'] = round(acc,3)*100
            elif t.startswith('numeric_bodmas') and t.endswith('_complex_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Complex 5-CoT', 'BODMAS'] = round(acc,3)*100
            elif t.startswith('numeric_bodmas') and t.endswith('_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['5-CoT', 'BODMAS'] = round(acc,3)*100
            
        #if t.startswith('numeric_') and t!= 'numeric_answer':
            elif t.startswith('numeric_elog') and t.endswith('_math_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Generic', 'Inverse'] = round(acc,3)*100
            elif t.startswith('numeric_elog') and t.endswith('_math_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Notice', 'Inverse'] = round(acc,3)*100
            elif t.startswith('numeric_elog') and t.endswith('_complex_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Complex 5-CoT', 'Inverse'] = round(acc,3)*100
            elif t.startswith('numeric_elog') and t.endswith('_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['5-CoT', 'Inverse'] = round(acc,3)*100
            
        #if t.startswith('numeric_') and t!= 'numeric_answer':
            elif t.startswith('numeric_trigno') and t.endswith('_math_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Generic', 'Trigonometry'] = round(acc,3)*100
            elif t.startswith('numeric_trigno') and t.endswith('_math_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Notice', 'Trigonometry'] = round(acc,3)*100
            elif t.startswith('numeric_trigno') and t.endswith('_complex_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Complex 5-CoT', 'Trigonometry'] = round(acc,3)*100
            elif t.startswith('numeric_trigno') and t.endswith('_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['5-CoT', 'Trigonometry'] = round(acc,3)*100
            
        #if t.startswith('numeric_') and t!= 'numeric_answer':
            elif t.startswith('numeric_'+header) and t.endswith('_math_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Generic', 'Original'] = round(acc,3)*100
            elif t.startswith('numeric_'+header) and t.endswith('_math_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Notice', 'Original'] = round(acc,3)*100
            elif t.startswith('numeric_'+header) and t.endswith('_complex_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['Complex 5-CoT', 'Original'] = round(acc,3)*100
            elif t.startswith('numeric_'+header) and t.endswith('_example_trick_prompt'):
                df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
                acc = df_.shape[0]/df.shape[0]
                results.loc['5-CoT', 'Original'] = round(acc,3)*100
        
    results.to_csv(output_dir+'results.csv')

def parse_answer(df, answer_column):
    
    try:
        df[['calculation','numeric_answer']] = df[answer_column].str.rsplit('####', n=1, expand=True)
    except ValueError:
        pass
    df['numeric_answer'] = df[answer_column].str.extract('(-?\d+)')
    df['numeric_answer'] = df['numeric_answer'].str.strip()
    return df

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

def main(input_file, output_dir, header, answer_column):
    if type(input_file) == str:
        df = pd.read_csv(input_file)
        if answer_column not in df:
            raise Exception(f"{input_file} does not have {answer} column")
    else:
        df = input_file
        #print(df.head())
    df = parse_answer(df, answer_column)
    for t in df.columns.values:
        if t.endswith('_prompt'):
            #try:
            #    df[[f'calculation_{t}', f'numeric_{t}']] = df[t].str.rsplit('####', n=1, expand=True)#[-1]
            #except ValueError:
                df = extract_answer(df, t, f'numeric_{t}')
    for t in df.columns.values:
        if t.startswith('numeric_'):
            clean_column(df, t)
    for t in df.columns.values:
        if t.startswith('numeric_') and t!= 'numeric_answer':
            df_ = df[['numeric_answer', t]][df['numeric_answer']==df[t]]
            print(t.replace('numeric_', ''))
            print("Matched answers for ", df_.shape[0], "out of ", df.shape[0])
            acc = df_.shape[0]/df.shape[0]
            print("Accuracy on complex GSM8K question: ", round(acc,4)*100)
    dump_results(df, header, output_dir)
    


def load_args():
    parser = argparse.ArgumentParser(description="Parsing output for dataset")
    parser.add_argument("--input_filepath", required=True,
                        help="input csv filepath with the columns, ['answer', '*_prompt', ]")
    parser.add_argument("--answer", required=True,
                        help="name of the field for correct answer of question", default="answer")
    parser.add_argument("--header", required=True,
                        help="header of the field for input question", default="question")
    parser.add_argument("--output_directory", required=True,
                        help="output directory for local output dump")
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = load_args()
    main(
        input_file=args['input_filepath'],
        output_dir=args['output_directory'],
        header=args['header'],
        answer_column=args['answer']
        
    )

