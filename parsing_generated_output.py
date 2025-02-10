#!/usr/bin/env python
# coding: utf-8


import re
import argparse
import pandas as pd


def clean_column(data, col):
    data[col] = data[col].str.replace('$', '')
    data[col] = data[col].str.replace(',', '')
    data[col] = data[col].str.replace('}', '')
    data[col] = data[col].str.strip()
    return data

def main(input_file): 
    answer_column = 'answer'
    model_response_column = 'question_response'
    model_response_complex = 'new_question_response'
    df = pd.read_csv(input_file)
    df[['calculation','numeric_answer']] = df[answer_column].str.rsplit('####', n=1, expand=True)

    df[['calculation_question_response', 'numeric_answer_question_response']] = df[model_response_column].str.rsplit('####', n=1, expand=True)

    df[['calculation_new_question_response', 'numeric_answer_new_question_response']] = df[model_response_complex].str.rsplit('####', n=1, expand=True)


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

    print("Accuracy on GSM8K question: ", round(acc,4)*100)

    df_ = df[['numeric_answer', 'numeric_answer_new_question_response']][df['numeric_answer']==df['numeric_answer_new_question_response']]
    print("Matched answers for ", df_.shape[0], "out of ", df.shape[0])
    acc = df_.shape[0]/df.shape[0]
    print("Accuracy on complex GSM8K question: ", round(acc,4)*100)

def load_args():
    parser = argparse.ArgumentParser(description="Parsing output for GSM8K")
    parser.add_argument("--input_filepath", required=True,
                        help="input csv filepath with the columns, ['answer', 'question_response', 'new_question_response']")
    
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = load_args()
    main(
        args['input_filepath'] 
    )
