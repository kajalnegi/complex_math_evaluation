# complex_math_evaluation


### To create new questions from with the same result
python createSamples.py --input_filepath <input_filepath> --output_directory <DIR>

input_filepath is in the format of https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data


### To generate model output from the question created from createSamples.py


python evaluation_comparison.py --input_filepath <input_filepath> --output_directory <DIR>

input_filepath is .csv file with two columns "question", "new_question"

