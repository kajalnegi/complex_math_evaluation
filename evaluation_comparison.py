#!/usr/bin/env python
# coding: utf-8

import os
import errno
import argparse
import transformers
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from vllm import LLM, SamplingParams

class vllmPipeline:
    def __init__(self, model_id):
        self.llm = LLM(model_id)
        self.sampling_params = SamplingParams(max_tokens=2000)
    def __call__(self, prompt):
        return self.llm.chat(messages=prompt, sampling_params=self.sampling_params)

def return_model(model_id, model_type="hf"):
    #model_id = "meta-llama/Llama-2-7b-chat-hf"
    #model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    #model_id = "mistralai/Mistral-Nemo-Instruct-2407"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    try:
        if model_type == "hf":
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            generate_text = pipeline('text-generation', model=model_id, tokenizer=tokenizer, device_map="cuda:4")
        elif model_type == "vllm":
            generate_text = vllmPipeline(model_id)

    except Exception as inst:
        print(type(inst))    # the exception type

        print(inst.args)     # arguments stored in .args

        raise Exception("Check model_id")
    return generate_text

def create_math_prompt(problem_texts: list) -> list:
    prompts = [
    f"""
    Solve the following  mathematics problem:
    {problem_text}
    Provide your solution in the following format:
    1. A step-by-step brief numeric calculations on how to arrive at the solution (No programming code)
    2. Place the final numeric answer without any unit or sentence after ####

    Remember, this is a high school level problem, so advanced mathematical concepts should not be used.
    Always follow the format.

    """ for problem_text in problem_texts]
    return [[{"role": "user", "content": str(prompt)}] for prompt in prompts]

def create_math_trick_prompt(problem_texts: str) -> str:

    prompts = [f"""
    Solve the following  mathematics problem:
    {problem_text}
    Provide your solution in the following format:
    1. A step-by-step brief numeric calculations on how to arrive at the solution (No programming code)
    2. Place the final numeric answer without any unit or sentence after ####

    Remember, this question contains mathematical identities and trivialities; therefore, resolve them before calculating the answer.
    Also, this is a high school level problem, so advanced mathematical concepts should not be used.
    Always follow the format.
    """ for problem_text in problem_texts]
    return [{"role": "user", "content": str(prompt)} for prompt in prompts]

def create_example_prompt(problem_texts: str) -> str:
    prompts = [f"""
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

    """ for problem_text in problem_texts]
    return [{"role": "user", "content": str(prompt)} for prompt in prompts]

def create_example_trick_prompt(problem_texts: str) -> str:

    prompts = [f"""
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
    """ for problem_text in problem_texts]
    return [[{"role": "user", "content": str(prompt)}] for prompt in prompts]

def query_model(problem_text: str, generate_text, model_type="hf"):
    try:
        prompts = create_math_prompt(problem_text)

        if model_type == 'hf':
            responses = generate_text(prompts,
                max_length=2000,
                num_return_sequences=1,
                truncation=True)

        elif model_type == 'vllm':
            responses = generate_text(prompts)

        out = []
        for response in responses:
            if model_type == "hf":
                for t in response[0]['generated_text']:
                    if t['role'] == 'assistant':
                        out.append(t['content'].strip())
            elif model_type == "vllm":
                out.append(response.outputs[0].text.strip())

        print("Done One")
        return out
    except Exception as inst:
        print(type(inst))    # the exception type

        print(inst.args)     # arguments stored in .args

        print(inst)
        return pd.NA

def new_query_model(problem_text: str, generate_text, model_type):
    try:
        prompts = create_example_trick_prompt(problem_text)
        if model_type == "hf":
            responses = generate_text(prompts,
                max_length=2000,
                num_return_sequences=1,
                truncation=True)

        elif model_type == "vllm":
            responses = generate_text(prompts)

        out = []
        for response in responses:
            if model_type == "hf":
                for t in response[0]['generated_text']:
                    if t['role'] == 'assistant':
                        out.append(t['content'].strip())
            elif model_type == "vllm":
                out.append(response.outputs[0].text.strip())
        return out
    except Exception as inst:
        print(type(inst))    # the exception type

        print(inst.args)     # arguments stored in .args

        print(inst)
        return pd.NA

# batched iter on list:
def batched_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main(input_file, output_filepath, model_id, model_type, batch_size=32):
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
        generate_text = return_model(model_id, model_type=model_type)
        df['new_question'] = df['new_question'].fillna(df['question'])
        if model_type == "hf":
            question_response = [
                query_model(x, generate_text, model_type)
                for x in tqdm(batched_iter(df['question'].to_list(), batch_size), total=df.shape[0] // batch_size)]
            question_response = [sample for batch in question_response for sample in batch]
        elif model_type == "vllm":
            question_response = query_model(df['question'].to_list(), generate_text, model_type)
        df['question_response'] = question_response
        df.to_csv(output_filepath, index=False)
        if model_type == "hf":
            new_question_response = [
                new_query_model(x, generate_text, model_type)
                for x in tqdm(batched_iter(df['new_question'].to_list(), batch_size), total=df.shape[0] // batch_size)]
            new_question_response = [sample for batch in new_question_response for sample in batch]
        elif model_type == "vllm":
            new_question_response = new_query_model(df['new_question'].to_list(), generate_text, model_type)
        df['new_question_response'] = new_question_response
        df.to_csv(output_filepath, index=False)
    except Exception as e:
        print(e)


def load_args():
    parser = argparse.ArgumentParser(description="Generating answers for GSM8K")
    parser.add_argument("--input_filepath", required=True,
                        help="input filepath with the question in csv format with columns ['question', 'new_question']")
    parser.add_argument("--output_directory", required=True,
                        help="output directory for local output dump")
    parser.add_argument("--model_id", required=True,
                        help="model_id")
    parser.add_argument("--model_type", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--batch_size", default=32, type=int)

    return vars(parser.parse_args())


if __name__ == '__main__':
    from tqdm import tqdm
    args = load_args()
    main(
        input_file=args['input_filepath'],
        output_filepath=args['output_directory'],
        model_id=args['model_id'],
        model_type=args["model_type"],
        batch_size=args["batch_size"]
    )
