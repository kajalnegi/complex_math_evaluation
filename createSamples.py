#!/usr/bin/env python
# coding: utf-8

import os
import re
import errno
import random
import math
import json
import pandas as pd
import argparse


# using $ as a proxy for replacing with any value
math_exp = {1:["cos^2($) + sin^2($)", "sec^2($)-tan^2($)", "$/$", "0.5*2", "cosec^2($) - cot^2($)"], 
            'x':["LOG(EXPONENTIAL($))", "EXPONENTIAL(LOG($))"]
           }

def pattern_digit():
    """
    regex patter to find digit in the string
    """
    r = r'\s\d+\s'
    patt = re.compile(r)
    return patt

def generate_math_expression(d):
    """
    generates multiple kinds of math expressions
    """
    sign = ["+", "-", "*", "/"]
    d1 = random.randint(1, d+1)
    d2 = random.randint(1, d+1)
    sgn = random.choice(sign)
    exp = '(' + str(d1) + sgn + str(d2) + ')'
    return exp


def evaluate(exp):
    """
    evaluated value of the expressions
    """
    return eval(exp)


def generate_two_num_math_expression(d1, d2):
    """
    creates mathematical expression with two input numbers
    using $ as a separation for opposite operations
    """
    sign = ["/$*", "*$/","+$-", "-$+"]
    sgn = random.choice(sign)
    d = evaluate(str(d1) + sgn.split('$')[1] + str(d2))
    exp = '(' + str(d) + sgn.split('$')[0] + str(d2) + ')'
    #check if the original number can be retrieved from the expression
    check = d1 == evaluate(exp)
    if check:
        exp = '(' + str(d) + sgn.split('$')[0] + '(key))'
        return exp


def generate_replacement(num, exp1, math_exp_key, math_func):
    if type(math_exp_key)==int:  
        two_num_exp = generate_two_num_math_expression(num, math_exp_key)
        math_func = math_func.replace('$', str(random.randint(1, 100)),100)
        two_num_exp = two_num_exp.replace('key', math_func)
        return two_num_exp
    if math_exp_key=='x':
        return math_func.replace('$', str(num),100)
    
def create_alternate_question(q, n=1):
    """
    creates alternate questions with the same answer
    n is the number of the integers to change in the question
    return: a new question (str)
    """
    patt = pattern_digit()
    ind = [(m.start(0), m.end(0)) for m in re.finditer(patt, q)]
    if ind:
        nums = random.sample(ind, n)
        for num in nums:
            int_num = int(q[num[0]:num[1]])
            exp1 = generate_math_expression(int_num)
            math_exp_key = random.choice(list(math_exp.keys()))
            math_func = random.choice(math_exp[math_exp_key])
            replace_exp = generate_replacement(int_num, exp1, math_exp_key, math_func)
            new_question = q[:num[0]] +' '+ replace_exp +' ' +q[num[1]:]
            return new_question

def main(input_file, output_filepath):
    isExist = os.path.exists(input_file)
    if not isExist:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_file)
    isExist = os.path.exists(output_filepath)
    if not isExist:    
        raise NotADirectoryError(
            errno.ENOTDIR, os.strerror(errno.ENOTDIR), output_filepath)
    output_filepath = os.path.join(output_filepath, "output.csv")  
    try: 
        with open(input_file, 'r') as f:
            df = pd.DataFrame([json.loads(l) for l in f.readlines()])
        print("Generating new questions \n")
        df['new_question'] = df['question'].apply(create_alternate_question)
        print("Dumping the output file \n")
        df.to_csv(output_filepath, index=False)
    except Exception as e:
        print(e)
    
def load_args():
    parser = argparse.ArgumentParser(description="Optimized implementation of SaGe method")
    parser.add_argument("--input_filepath", required=True,
                        help="input filepath with the question")
    parser.add_argument("--output_directory", required=True,
                        help="output directory for local output dump")

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = load_args()
    main(
        args['input_filepath'],
        args['output_directory'],
    )
