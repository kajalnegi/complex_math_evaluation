#!/usr/bin/env python
# coding: utf-8
import argparse
import create_samples
import evaluation_comparison_vllm
import parsing_generated_output
import parsing_generated_output_deepseek


def main(input_filepath, output_directory, header, model_id,
    model_type, batch_size, tp_size, seed, level, answer_column):
    
    try:
        df_input, input_file = create_samples.main(input_filepath, output_directory, header, level)
        print("\n-------Data creation complete-------\n")
        if answer_column not in df_input:
            raise Exception(f"{input_filepath} does not have {answer} column")
        
    except Exception as e:
        print(type(e))    # the exception type
        print(e.args)
        raise Exception("Error creating complexified question for inputfile")
    try:
        df_generated, generated_file_path = evaluation_comparison_vllm.main(df_input, output_directory, header, model_id,
        model_type, batch_size, tp_size, seed)
        print("\n-------Model inference complete-------\n")
    except Exception as e:
        print(type(e))    # the exception type
        print(e.args)
        raise Exception("Error in model inference")
    try:
        if 'deepseek-ai' in model_id:
            parsing_generated_output_deepseek.main(df_generated, output_directory, header, answer_column)
        else:
            parsing_generated_output.main(df_generated, output_directory, header, answer_column)
        print("\n-------Dumping results-------\n")
    except Exception as e:
        print(type(e))    # the exception type
        print(e.args)
        raise Exception("Error in parsing results")

def load_args():
    parser = argparse.ArgumentParser(description="Generating new questions from GSM8K")
    parser.add_argument("--input_filepath", required=True,
                        help="input filepath with the question")
    parser.add_argument("--output_directory", required=True,
                        help="output directory for local output dump")
    parser.add_argument("--header", required=False,
                        help="header of the field for input question", default="question")
    parser.add_argument("--model_id", required=True,
                        help="model_id")
    parser.add_argument("--model_type", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--tp_size', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument("--answer", required=False,
                        help="name of the field for correct answer of question", default="answer")
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = load_args()
    main(
        input_filepath=args['input_filepath'],
        output_directory=args['output_directory'],
        header=args['header'],
        model_id=args['model_id'],
        model_type=args["model_type"],
        batch_size=args["batch_size"],
        tp_size=args["tp_size"],
        seed=args["seed"],
        level=args["level"],
        answer_column=args['answer']
    )
