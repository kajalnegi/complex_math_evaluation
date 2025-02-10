#!/usr/bin/env python
# coding: utf-8

import os
import errno
import transformers
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#model_id = "meta-llama/Llama-2-7b-chat-hf"
#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model_id = "mistralai/Mistral-Nemo-Instruct-2407"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
generate_text = pipeline('text-generation', model=model_id, tokenizer=tokenizer, device_map="auto")


def create_math_prompt(problem_text: str) -> str:
    prompt = f"""
    Solve the following  mathematics problem:
    {problem_text}
    Provide your solution in the following format:
    1. A step-by-step brief numeric calculations on how to arrive at the solution (No programming code)
    2. Place the final numeric answer without any unit or sentence after #### 
    
    Remember, this is a high school level problem, so advanced mathematical concepts should not be used.
    Always follow the format.

    """
    return [{"role": "user", "content": str(prompt)}]

def create_math_trick_prompt(problem_text: str) -> str:

    prompt = f"""
    Solve the following  mathematics problem:
    {problem_text}
    Provide your solution in the following format:
    1. A step-by-step brief numeric calculations on how to arrive at the solution (No programming code)
    2. Place the final numeric answer without any unit or sentence after #### 
    
    Remember, this question contains mathematical identities and trivialities; therefore, resolve them before calculating the answer.
    Also, this is a high school level problem, so advanced mathematical concepts should not be used.
    Always follow the format.
    """
    return [{"role": "user", "content": str(prompt)}]

def create_example_prompt(problem_text: str) -> str:
    prompt = f"""
    Here are some mathematical questions along with their answers.
    Question 1: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    Answer 1: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
            Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
            #### 72
    Question 2: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    Answer 2: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
            Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
            #### 10
    Question 3: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    Answer 3: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
            Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
            This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
            #### 5
    Question 4: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Answer 4: Maila read 12 x 2 = <<12*2=24>>24 pages today.
            So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.
            There are 120 - 36 = <<120-36=84>>84 pages left to be read.
            Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.
            #### 42
    Question 5: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
    Answer 5: He writes each friend 3*2=<<3*2=6>>6 pages a week
            So he writes 6*2=<<6*2=12>>12 pages every week
            That means he writes 12*52=<<12*52=624>>624 pages a year
            #### 624
    Solve the following  mathematics problem:
    {problem_text}
    Provide your solution in the following format:
    1. A step-by-step brief numeric calculations on how to arrive at the solution (No programming code)
    2. Place the final numeric answer without any unit or sentence after #### 
    
    Remember, this is a high school level problem, so advanced mathematical concepts should not be used.
    Always follow the format.

    """
    return [{"role": "user", "content": str(prompt)}]

def create_example_trick_prompt(problem_text: str) -> str:

    prompt = f"""
    Here are some mathematical questions along with their answers.
    Question 1: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    Answer 1: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
            Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
            #### 72
    Question 2: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    Answer 2: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
            Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
            #### 10
    Question 3: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    Answer 3: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
            Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
            This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
            #### 5
    Question 4: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Answer 4: Maila read 12 x 2 = <<12*2=24>>24 pages today.
            So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.
            There are 120 - 36 = <<120-36=84>>84 pages left to be read.
            Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.
            #### 42
    Question 5: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
    Answer 5: He writes each friend 3*2=<<3*2=6>>6 pages a week
            So he writes 6*2=<<6*2=12>>12 pages every week
            That means he writes 12*52=<<12*52=624>>624 pages a year
            #### 624
    Solve the following  mathematics problem:
    {problem_text}
    Provide your solution in the following format:
    1. A step-by-step brief numeric calculations on how to arrive at the solution (No programming code)
    2. Place the final numeric answer without any unit or sentence after #### 
    
    Remember, this question contains mathematical identities and trivialities; therefore, resolve them before calculating the answer.
    Also, this is a high school level problem, so advanced mathematical concepts should not be used.
    Always follow the format.
    """
    return [{"role": "user", "content": str(prompt)}]

def query_model(problem_text: str):
    try:
        prompt = create_math_prompt(problem_text)
        
        response = generate_text(prompt, 
                                    max_length=2000, 
                                    num_return_sequences=1,
                                    truncation=True
                                    )
        
        for t in response[0]['generated_text']:
            if t['role'] == 'assistant':
                return t['content'].strip()
        return response[0]['generated_text']
    except Exception as inst:
        print(type(inst))    # the exception type

        print(inst.args)     # arguments stored in .args

        print(inst)
        return pd.NA

def new_query_model(problem_text: str):
    try:
        prompt = create_example_trick_prompt(problem_text)
        response = generate_text(prompt, 
                                    max_length=2000, 
                                    num_return_sequences=1,
                                    truncation=True
                                    )
        for t in response[0]['generated_text']:
            if t['role'] == 'assistant':
                return t['content'].strip()
        return response[0]['generated_text']
    except Exception as inst:
        print(type(inst))    # the exception type

        print(inst.args)     # arguments stored in .args

        print(inst)
        return pd.NA
        
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
        df = pd.read_csv(input_file)
        
        df['new_question'] = df['new_question'].fillna(df['question'])
        #my_list = [0,2,4]#8, 19, 70, 53, 1317, 1312, 1316]
        #df = df[df.index.isin(my_list)]
        df['question_response'] = df['question'].apply(query_model)

        df.to_csv(output_filepath, index=False)

        df['new_question_response'] = df['new_question'].apply(query_model)

        df.to_csv(output_filepath, index=False)
    except Exception as e:
        print(e)    
        

def load_args():
    parser = argparse.ArgumentParser(description="Generating answers for GSM8K")
    parser.add_argument("--input_filepath", required=True,
                        help="input filepath with the question in csv format")
    parser.add_argument("--output_directory", required=True,
                        help="output directory for local output dump")

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = load_args()
    main(
        args['input_filepath'],
        args['output_directory'],
    )