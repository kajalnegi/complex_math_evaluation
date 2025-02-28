# complex_math_evaluation


### To create new questions from with the same result
```bash
python createSamples.py --input_filepath <input_filepath> --output_directory <DIR>
```

input_filepath is in the format of https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data


### To generate model output from the question created from createSamples.py

```bash
python evaluation_comparison.py --input_filepath <input_filepath> --output_directory <DIR>
```

input_filepath is .csv file with two columns "question", "new_question"

## To run targer models you should now be able to use a command similar to the following


An example running llama 3.3 70b using vllm with 4 gpus and a tensor parallel size of 4.

```bash
export CUDA_VISIBLE_DEVICES="0,1,3,5"
python evaluation_comparison.py --input_filepath datasets/output.csv \
    --output_directory outputs/llama_3.3_results \
    --model_id meta-llama/Llama-3.3-70B-Instruct \
    --model_type vllm \
    --tp_size 4
```

## Seeding generations

There is now a `--seed` argument that will seed the generation that use sampling.


### To calculate accuracy from model output from evaluation_comparison.py

```bash
python parsing_generated_output.py --input_filepath <input_filepath>
```

For extracting answer from "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" use parsing_generated_output-deepseek.py

