#!/usr/bin/env python
# coding: utf-8
import os
import re
import json
import random
import math
import argparse
import numpy as np
import pandas as pd
trig_exp = {1:["cos^2($) + sin^2($)", "sec^2($)-tan^2($)",  "cosec^2($) - cot^2($)"]}
bodmas_exp =  {1:[ "$/$", "0.5*2"]}
elog_exp = {'x':["LOG(EXPONENTIAL($))", "EXPONENTIAL(LOG($))"]}

def pattern_digit():
    """
    regex patter to find digit in the string
    """
    r = r'\s\d+\s'
    r = r'\s\d+(,\d+)?\s'
    patt = re.compile(r)
    return patt

def pattern_dollar_digit():
    """
    regex patter to find digit with dollar in the string
    """
    r = r'\s\$\d+\s'
    r = r'\s\$\d+(\s|,|.)\s'
    r = r'\s\$\d+(,\d+)?(\s|,|.)?\s'
    patt = re.compile(r)
    return patt

def generate_math_expression(d):
    sign = ["+", "-", "*", "/"]
    d1 = random.randint(0, d)
    d2 = random.randint(0, d)
    sgn = random.choice(sign)
    exp = '(' + str(d1) + sgn + str(d2) + ')'
    return exp

def evaluate(exp):
    return eval(exp)

def generate_two_num_math_expression(d1, d2):
    sign = ["/$*", "*$/","+$-", "-$+"]
    sgn = random.choice(sign)
    
    d = evaluate(str(d1) + sgn.split('$')[1] + str(d2))
    exp = '(' + str(d) + sgn.split('$')[0] + str(d2) + ')'
    check = d1 == evaluate(exp)
    if check:
        exp = '(' + str(d) + sgn.split('$')[0] + '(key))'

        return exp


def generate_replacement(num, exp1, math_exp_key, math_func):
    if type(math_exp_key)==int:  
        
        two_num_exp = generate_two_num_math_expression(num, math_exp_key)
        
        math_func = math_func.replace('$', str(random.randint(1, 100)),100)#for constant functions choose any random integer
        two_num_exp = two_num_exp.replace('key', math_func)
        
        return two_num_exp
    if math_exp_key=='x':
        
        return math_func.replace('$', str(num),100)
    
def make_question_string(q, num, replace_exp, extra_str, end):
    return q[:num[0]] +' '+ extra_str+replace_exp +' '+ end +' ' +q[num[1]:] 

def generate_three_questions(q, int_num, num, extra_str, end):
    exp1 = generate_math_expression(int_num)
    
    bodmas_exp_key = random.choice(list(bodmas_exp.keys()))
    math_func = random.choice(bodmas_exp[bodmas_exp_key])
    
    replace_exp = generate_replacement(int_num, exp1, bodmas_exp_key, math_func)
    print(replace_exp)
    bodmas_question = make_question_string(q, num, replace_exp, extra_str, end)
        
    elog_exp_key = random.choice(list(elog_exp.keys()))
    math_func = random.choice(elog_exp[elog_exp_key])
    replace_exp = generate_replacement(int_num, exp1, elog_exp_key, math_func)
    elog_question = make_question_string(q, num, replace_exp, extra_str, end)
    print(replace_exp) 
    trig_exp_key = random.choice(list(trig_exp.keys()))
    math_func = random.choice(trig_exp[trig_exp_key])
    replace_exp = generate_replacement(int_num, exp1, trig_exp_key, math_func)
    trigno_question = make_question_string(q, num, replace_exp, extra_str, end)
    print(replace_exp) 
    return bodmas_question, elog_question, trigno_question

def remove_punction(s):
    s = s.strip()
    if s[-1]=='.':
        s = s.replace('.', '')
        s = s.replace(',', '')
        return s, ' . '
    if s[-1]==',':
        s = s.replace('.', '')
        s = s.replace(',', '')
        return s, ' , '
    s = s.replace(',', '')
    return s, ''

def create_alternate_question(data, header, n=1):
    q = data[header]
    #print(q)
    try:
        patt = pattern_digit()
        ind = [(m.start(0), m.end(0)) for m in re.finditer(patt, q)]
        if ind:
            nums = random.sample(ind, n)
            for num in nums:
                
                int_num, end = remove_punction(q[num[0]:num[1]])
                int_num = int(int_num)
                bodmas_question, elog_question, trigno_question = generate_three_questions(q, int_num, num, '', end)
                
                return bodmas_question, elog_question, trigno_question
        else:
            patt = pattern_dollar_digit()
            ind = [(m.start(0), m.end(0)) for m in re.finditer(patt, q)]
            if ind:
                nums = random.sample(ind, n)
                for num in nums:
                    int_num = q[num[0]+2:num[1]]
                    print(int_num)
                    int_num, end = remove_punction(int_num)
                    #print(int_num, end)
                    int_num = int(int_num)
                    bodmas_question, elog_question, trigno_question = generate_three_questions(q, int_num, num, '$', end)
                    return bodmas_question, elog_question, trigno_question
        return np.nan, np.nan, np.nan
    except ValueError:
        #print()
        return np.nan, np.nan, np.nan
    
def main(input_file, output_filepath, header):
    isExist = os.path.exists(input_file)
    if not isExist:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_file)
    isExist = os.path.exists(output_filepath)
    if not isExist:    
        raise NotADirectoryError(
            errno.ENOTDIR, os.strerror(errno.ENOTDIR), output_filepath)
    filename = input_file.split('/')[-1]
    try:
        filename = filename.split('.')[0]
    except:
        pass
    output_filepath = os.path.join(output_filepath, filename+ ".csv")  
    try: 
        with open(input_file, 'r') as f:
            df = pd.DataFrame([json.loads(l) for l in f.readlines()])
        print("Generating new questions \n")
        df[['bodmas_question', 'elog_question', 'trigno_question']] = df.apply(create_alternate_question,axis=1, args=(header,), result_type="expand")
        print("Dumping the output file \n")
        df.to_csv(output_filepath, index=False)
    except Exception as e:
        print(e)
        
def load_args():
    parser = argparse.ArgumentParser(description="Generating new questions from GSM8K")
    parser.add_argument("--input_filepath", required=True,
                        help="input filepath with the question")
    parser.add_argument("--output_directory", required=True,
                        help="output directory for local output dump")
    parser.add_argument("--header", required=True,
                        help="header of the field for input question", default="question")
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = load_args()
    main(
        args['input_filepath'],
        args['output_directory'],
        args['header']
    )
