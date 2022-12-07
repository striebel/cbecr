## Career-Based Explainable Course Recommendation

This repository hosts code and data associated with the paper
```bib
@inproceedings{striebel2023career,
  author    = {Striebel, Jacob and Myers, Rebecca and Liu, Xiaozhong},
  year      = 2023,
  title     = {Career-Based Explainable Course Recommendation},
  booktitle = {Proceedings of i{C}onference 2023}
}
```

### Reproduce the Results

The results of the system evaluation, which are reported in the paper,
are given in the file
```
eval/out/eval_table.csv
```
The hand-scored outputs of our system and the two baselines,
from which the above results are tabulated,
are given in the file
```
eval/in/blind_eval_filled.csv
```
To rerun the tabulation script and/or our full system and the baselines, first
create a Python virtual environment and install the required packages by executing
the following Linux shell commands from this repository's root directory:[^1]
```sh
python -m venv myvenv
```
```sh
source myvenv/bin/activate
```
```sh
pip install -r requirements.txt
```
To retabulate the evaluation results (`eval/out/eval_table.csv`) from the
hand-scored system outputs (`eval/in/blind_eval_filled.csv`), execute
```sh
python src/tabulate_evaluation_forms.py
```

To rerun our explainable course recommendation system and the two
baselines using the job queries given in the paper, execute
```sh
python src/generate_evaluation_forms.py
```
This will generate the following two files
```
eval/out/full_eval.csv
```
```
eval/out/blind_eval.csv
```
To perform your own evaluation, copy the first file to
```
eval/in/full_eval.csv
```
and copy the second file to
```
eval/in/blind_eval_filled.csv
```
Open `blind_eval_filled.csv` with a spreadsheet program, and enter a score for
each course recommendation and each explanation with respect to their job query.
Use a four-point scale with 0 worst and 3 best.
After entering the scores, save the file.
Then run
```sh
python src/tabulate_evaluation_forms.py
```
The file
```
eval/out/eval_table.csv
```
will be generated which contains the results of the new evaluation.

[^1]: We tested the workflow described in this document on Fedora 33 (Linux)
with Python 3.9.9.
