#!/usr/bin/env python
# coding: utf-8

import re
import argparse
import pandas as pd


def extract_answer(data, column, new_column):
    def find_answer(row):
        split1= row.split('####')
        split2 = row.split('\\boxed{')
        split3 = row.split('**Final Answer:**')
        split4 = row.split('**Answer:**')
        split5 = row.split('**')
        split6 = row.split('\nboxed{')
        
        if len(split2)>1:
            if len(split2[1]) > len(split2[0]):
                return split2[0].split('}')[0]
            else:
                return split2[1].split('}')[0]
        if len(split6)>1:
            if len(split6[1]) > len(split6[0]):
                return split6[0].split('}')[0]
            else:
                return split6[1].split('}')[0]
        if len(split1)>1:
            if len(split1[0]) > len(split1[1]):
                return split1[1]
            else:
                return split1[0]
        
        if len(split3)>1:
            if len(split3[1]) > len(split3[0]):
                return split3[0]
            else:
                return split3[1]
        if len(split4)>1:
            if len(split4[1]) > len(split4[0]):
                return split4[0]
            else:
                return split4[1]
        if len(split5)>1:
            if len(split5[1]) > len(split5[0]):
                return split5[0]
            else:
                return split5[1]

    data[new_column] = data[column].apply(find_answer)
    
    return data    


def clean_column(data, col):
    data[col] = data[col].str.replace('$', '')
    data[col] = data[col].str.replace(',', '')
    data[col] = data[col].str.replace('}', '')
    data[col] = data[col].str.replace('\\', '')
    data[col] = data[col].str.strip()
    return data
    
def main(input_file): 
    answer_column = 'answer'
    model_response_column = 'question_response'
    model_response_complex = 'new_question_response'
    df = pd.read_csv(input_file)
    df = extract_answer(df, model_response_column, 'numeric_answer_question_response')
    df = extract_answer(df, model_response_complex, 'numeric_answer_new_question_response')

    df[['calculation','numeric_answer']] = df[answer_column].str.rsplit('####', n=1, expand=True)

    df = clean_column(df, 'numeric_answer_question_response')
    df = clean_column(df, 'numeric_answer')
    df = clean_column(df, 'numeric_answer_new_question_response')

    print("No response extracted from column, ", 'numeric_answer_question_response')
    print(df[df['numeric_answer_question_response'].isnull()]['question_response'])
    print("No response extracted from column, ", 'numeric_answer_new_question_response')
    print(df[df['numeric_answer_new_question_response'].isnull()]['new_question_response'])

    df_ = df[['numeric_answer', 'numeric_answer_question_response']][df['numeric_answer']==df['numeric_answer_question_response']]

    print("Matched answers for ", df_.shape[0], "out of ", df.shape[0])
    acc = df_.shape[0]/df.shape[0]

    print("Accuracy on GSM8K question: ", round(acc, 4)*100)

    df_ = df[['numeric_answer', 'numeric_answer_new_question_response']][df['numeric_answer']==df['numeric_answer_new_question_response']]
    print("Matched answers for ", df_.shape[0], "out of ", df.shape[0])
    acc = df_.shape[0]/df.shape[0]
    print("Accuracy on complex GSM8K question: ", round(acc, 4)*100)


def load_args():
    parser = argparse.ArgumentParser(description="Optimized implementation of SaGe method")
    parser.add_argument("--input_filepath", required=True,
                        help="input csv filepath with the columns, ['answer', 'question_response', 'new_question_response']")
    
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = load_args()
    main(
        args['input_filepath'] 
    )