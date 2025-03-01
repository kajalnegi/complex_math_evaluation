# complex_math_evaluation


### To create new questions from with the same result
python create_samples.py --input_filepath <input_filepath> --output_directory <DIR>

input_filepath is in the format of https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data


### To generate model output from the question created from create_samples.py

using vLLM (for faster inference)
<br>
CUDA_VISIBLE_DEVICES="<>" python evaluation_comparison_vllm.py --input_filepath <input_filepath> --output_directory <DIR> --model_id <model_id> --model_type "vllm" --tp_size 2 --batch_size 16
<br>
using hugging face
<br>
CUDA_VISIBLE_DEVICES="<>" python evaluation_comparison_vllm.py --input_filepath <input_filepath> --output_directory <DIR> --model_id <model_id> --model_type "hf" --tp_size 2 --batch_size 16

input_filepath is .csv file with two columns "question", "new_question"

<br>

### To calculate accuracy from model output from evaluation_comparison_vllm.py


python parsing_generated_output.py --input_filepath <input_filepath>


<br> For extracting answer from "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" use parsing_generated_output-deepseek.py