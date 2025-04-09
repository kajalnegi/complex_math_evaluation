# GSM-Identity
This code repository is for GSM-Identity.
<br>
For assesing a LLM model in a fresh math complex dataset, use below command for any input_filepath is in the format of 
<br>
https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data

<br>
*CUDA_VISIBLE_DEVICES="<>" python math_llms.py --input_filepath <input_filepath> --output_directory <DIR> --model_id <model_id> --model_type "vllm" --tp_size <tp_size> --batch_size <batch_size> --answer <field_for_answer> --header <header_for_question>*
<br>
 *tp_size* [default=1] is an integer for variable *tensor_parallel_size* in  *vllm.LLM*
 <br>

To run the process in breaks use below commands.

### To create new questions from with the same result
*python create_samples.py --input_filepath <input_filepath> --output_directory <DIR> --header <header_for_question> --level <level>*

*level* [default=1] changes the difficulty level of newly generated question. *level 2* is more difficult than *level 1*

Input_filepath is in the format of https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data

Output files create columns *'bodmas_question', 'elog_question', 'trigno_question'*

<br>
### To generate model output from the question created from create_samples.py

Using vLLM (for faster model inference)
<br>
*CUDA_VISIBLE_DEVICES="<>" python evaluation_comparison.py --input_filepath <input_filepath> --output_directory <DIR> --model_id <model_id> --model_type "vllm" --tp_size 2 --batch_size 16 --header <header_for_question>*
<br>

Using hugging face
<br>
*CUDA_VISIBLE_DEVICES="<>" python evaluation_comparison.py --input_filepath <input_filepath> --output_directory <DIR> --model_id <model_id> --model_type "hf" --tp_size 2 --batch_size 16*

input_filepath is .csv file with  columns *"question", and 'bodmas_question', 'elog_question', 'trigno_question'*

<br>

### To calculate accuracy from model output from evaluation_comparison_vllm.py


*python parsing_generated_output.py --input_filepath <input_filepath>*
