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
    return [[{"role": "user", "content": str(prompt)}] for prompt in prompts]

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
    return [[{"role": "user", "content": str(prompt)}] for prompt in prompts]

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

def create_complex_example_trick_prompt(problem_texts: str) -> str:

    prompts = [f"""
    Here are some mathematical questions along with their answers.
    	Question 1: Natalia sold clips to LOG(EXPONENTIAL(48)) of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

    Answer 1:
    * Calculate LOG(EXPONENTIAL(48)):  Since the exponential function and the logarithm function are inverses of each other, LOG(EXPONENTIAL(48)) = 48.
    * Calculate the number of clips sold in May: 48 / 2 = 24
    * Calculate the total number of clips sold: 48 + 24 = 72

	#### 72
    Question 2: Weng earns $12 an hour for babysitting. Yesterday, she just did (50/(cos^2(2) + sin^2(2))) minutes of babysitting. How much did she earn?

    Answer 2: "
		* cos^2(2) + sin^2(2) = 1 (Trigonometric Identity)
		* 50 / (cos^2(2) + sin^2(2)) = 50 / 1 = 50 minutes
		* Weng earns $12 per hour, and there are 60 minutes in an hour.
		*  Earnings = (50 minutes / 60 minutes/hour) * $12/hour

	 #### 10"

    Question 3: Randy has (60.0*(0.5*2)) mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?
    Answer 3: "
    * Calculate the number of mango trees: 60.0 * (0.5 * 2) = 60.0
    * Calculate half the number of mango trees: 60.0 / 2 = 30.0
    * Calculate the number of coconut trees: 30.0 - 5 = 25.0
    * Calculate the total number of trees: 60.0 + 25.0 = 85.0

	#### 85"

    Question 4: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed (846+(7/7)) people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?

    Answer 4: "  Calculate the total number of people eaten: 846 + (7 / 7) = 846 + 1 = 847 people
	2.  Let 'x' be the number of people on the first ship.
	3.  The second ship had 2*x people.
	4.  The third ship had 2*(2*x) = 4*x people.
	5.  The total number of people eaten is x + 2x + 4x = 847
	6.  Combine like terms: 7x = 847
	7.  Divide both sides by 7: x = 121

	#### 121"

    Question 5: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read LOG(EXPONENTIAL(120)) pages?
    Answer 5:   **Find Joy's reading rate:**
    * Joy reads 8 pages / 20 minutes = 0.4 pages per minute.

    **Calculate the total number of pages:**
    *  LOG(EXPONENTIAL(120)) = 120 (The logarithm and exponential functions cancel each other out)

    **Calculate the total time in minutes:**
    * 120 pages / 0.4 pages/minute = 300 minutes

    **Convert minutes to hours:**
    * 300 minutes / 60 minutes/hour = 5 hours

	#### 5

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

def query_model(problem_text: str, generate_text, model_type, prompt_func):
    try:
        prompts = prompt_func(problem_text)

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
        prompt_funcs = [
            create_math_prompt, create_math_trick_prompt, create_example_prompt, create_example_trick_prompt, create_complex_example_trick_prompt
        ]
        prompt_funcs = {func.__name__.replace('create_', ""): func for func in prompt_funcs}
        for col_name, prompt_func in prompt_funcs.items():
            if model_type == "hf":
                question_response = [
                    query_model(x, generate_text, model_type, prompt_func)
                    for x in tqdm(batched_iter(df['question'].to_list(), batch_size), total=df.shape[0] // batch_size)]
                question_response = [sample for batch in question_response for sample in batch]
            elif model_type == "vllm":
                question_response = query_model(df['question'].to_list(), generate_text, model_type, prompt_func)
            df[col_name] = question_response
            # df.to_csv(output_filepath, index=False)
            # if model_type == "hf":
            #     new_question_response = [
            #             query_model(x, generate_text, model_type)
            #             for x in tqdm(batched_iter(df['new_question'].to_list(), batch_size), total=df.shape[0] // batch_size)]
            #     new_question_response = [sample for batch in new_question_response for sample in batch]
            # elif model_type == "vllm":
            #     new_question_response = new_query_model(df['new_question'].to_list(), generate_text, model_type)
            # df['new_question_response'] = new_question_response
            # df.to_csv(output_filepath, index=False)
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
