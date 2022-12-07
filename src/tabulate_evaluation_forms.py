# <repo_root>/src/tabulate_evaluation_forms.py
#
# Author: Jacob Striebel
#
# This script tabulates the results of the system evaluation.
#
# The following input files are required:
#   <repo_root>/eval/in/blind_eval_filled.csv
#   <repo_root>/eval/in/full_eval.csv
#
# The following output file is generated:
#   <repo_root>/eval/out/eval_table.csv

import os
import pandas as pd
import math
import copy

src_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(src_path, os.pardir)
eval_path = os.path.join(root_path, 'eval')
eval_in_path = os.path.join(eval_path, 'in')
eval_out_path = os.path.join(eval_path, 'out')
blind_eval_filled_path = os.path.join(eval_in_path, 'blind_eval_filled.csv')
full_eval_path = os.path.join(eval_in_path, 'full_eval.csv')

assert os.path.exists(full_eval_path)
assert os.path.exists(blind_eval_filled_path)

full_df  = pd.read_csv(full_eval_path)
blind_df = pd.read_csv(blind_eval_filled_path)

full_df.rename(columns={list(full_df)[0]: 'first_index'}, inplace=True)
full_df.set_index('first_index', inplace=True)
blind_df.set_index('second_index', inplace=True)

full_df['course_is_relevant']     = [-1 for i in full_df.index]
full_df['course_dcg']             = [-1 for i in full_df.index]
full_df['explanation_is_quality'] = [-1 for i in full_df.index]
full_df['explanation_dcg']        = [-1 for i in full_df.index]

job_query__score_df = dict()

approaches=['X', 'Y', 'Z']

score_df_init_values={
  'approach'                       : [approach for approach in approaches],
  'course_precision_at_five'       : [0. for approach in approaches],
  'course_precision_at_ten'        : [0. for approach in approaches],
  'course_precision_at_twenty'     : [0. for approach in approaches],
  'course_ncg_at_five'             : [0. for approach in approaches],
  'course_ncg_at_ten'              : [0. for approach in approaches],
  'course_ncg_at_twenty'           : [0. for approach in approaches],
  'course_ndcg_at_five'            : [0. for approach in approaches],
  'course_ndcg_at_ten'             : [0. for approach in approaches],
  'course_ndcg_at_twenty'          : [0. for approach in approaches],
  'explanation_precision_at_five'  : [0. for approach in approaches],
  'explanation_precision_at_ten'   : [0. for approach in approaches],
  'explanation_precision_at_twenty': [0. for approach in approaches],
  'explanation_ncg_at_five'        : [0. for approach in approaches],
  'explanation_ncg_at_ten'         : [0. for approach in approaches],
  'explanation_ncg_at_twenty'      : [0. for approach in approaches],
  'explanation_ndcg_at_five'       : [0. for approach in approaches],
  'explanation_ndcg_at_ten'        : [0. for approach in approaches],
  'explanation_ndcg_at_twenty'     : [0. for approach in approaches]
}

idcg_at_five   = sum([3. / math.log(i+1, 2) for i in range(1,  6)])
idcg_at_ten    = sum([3. / math.log(i+1, 2) for i in range(1, 11)])
idcg_at_twenty = sum([3. / math.log(i+1, 2) for i in range(1, 21)])

for second_index in blind_df.index:
  first_indices = full_df.loc[full_df.second_index == second_index].index
  assert 1==len(first_indices)
  first_index = first_indices[0]
  course_relevance = int(blind_df.at[second_index, 'course_relevance'])
  explanation_quality = int(blind_df.at[second_index, 'explanation_quality'])
  assert course_relevance in [0, 1, 2, 3]
  assert explanation_quality in [0, 1, 2, 3]
  full_df.at[first_index, 'course_relevance'] = course_relevance
  full_df.at[first_index, 'explanation_quality'] = explanation_quality
  course_is_relevant = 1 if course_relevance >= 2 else 0
  explanation_is_quality = 1 if explanation_quality >= 2 else 0
  full_df.at[first_index, 'course_is_relevant'] = course_is_relevant
  full_df.at[first_index, 'explanation_is_quality'] = explanation_is_quality
  rank = int(full_df.at[first_index, 'rank'])
  assert 1 <= rank and rank <= 20
  course_dcg = float(course_relevance) / math.log(rank+1, 2)
  explanation_dcg = float(explanation_quality) / math.log(rank+1, 2)
  full_df.at[first_index, 'course_dcg'] = course_dcg
  full_df.at[first_index, 'explanation_dcg'] = explanation_dcg
  
  job_query = full_df.at[first_index, 'job_query']
  if job_query not in job_query__score_df:
    score_df = pd.DataFrame(data=copy.deepcopy(score_df_init_values))
    score_df.set_index('approach', inplace=True, verify_integrity=True)
    job_query__score_df[job_query] = score_df
  
  score_df = job_query__score_df[job_query]
  approach = full_df.at[first_index, 'approach']
  
  score_df.at[approach, 'course_precision_at_five'       ] += course_is_relevant     * 0.2     if rank <= 5  else 0.
  score_df.at[approach, 'course_precision_at_ten'        ] += course_is_relevant     * 0.1     if rank <= 10 else 0.
  score_df.at[approach, 'course_precision_at_twenty'     ] += course_is_relevant     * 0.05    if rank <= 20 else 0.
  score_df.at[approach, 'course_ncg_at_five'             ] += course_relevance    / (3 * 5 )   if rank <= 5  else 0.
  score_df.at[approach, 'course_ncg_at_ten'              ] += course_relevance    / (3 * 10)   if rank <= 10 else 0.
  score_df.at[approach, 'course_ncg_at_twenty'           ] += course_relevance    / (3 * 20)   if rank <= 20 else 0.
  score_df.at[approach, 'course_ndcg_at_five'            ] += course_dcg     / idcg_at_five    if rank <= 5  else 0.
  score_df.at[approach, 'course_ndcg_at_ten'             ] += course_dcg     / idcg_at_ten     if rank <= 10 else 0.
  score_df.at[approach, 'course_ndcg_at_twenty'          ] += course_dcg     / idcg_at_twenty  if rank <= 20 else 0.
  score_df.at[approach, 'explanation_precision_at_five'  ] += explanation_is_quality * 0.2     if rank <= 5  else 0.
  score_df.at[approach, 'explanation_precision_at_ten'   ] += explanation_is_quality * 0.1     if rank <= 10 else 0.
  score_df.at[approach, 'explanation_precision_at_twenty'] += explanation_is_quality * 0.05    if rank <= 20 else 0.
  score_df.at[approach, 'explanation_ncg_at_five'        ] += explanation_quality / (3 * 5 )   if rank <= 5  else 0.
  score_df.at[approach, 'explanation_ncg_at_ten'         ] += explanation_quality / (3 * 10)   if rank <= 10 else 0.
  score_df.at[approach, 'explanation_ncg_at_twenty'      ] += explanation_quality / (3 * 20)   if rank <= 20 else 0.
  score_df.at[approach, 'explanation_ndcg_at_five'       ] += explanation_dcg / idcg_at_five   if rank <= 5  else 0.
  score_df.at[approach, 'explanation_ndcg_at_ten'        ] += explanation_dcg / idcg_at_ten    if rank <= 10 else 0.
  score_df.at[approach, 'explanation_ndcg_at_twenty'     ] += explanation_dcg / idcg_at_twenty if rank <= 20 else 0.

master_score_df = pd.DataFrame(data=copy.deepcopy(score_df_init_values))
master_score_df.set_index('approach', inplace=True, verify_integrity=True)

job_count = 0
for job_query, score_df in job_query__score_df.items():
  master_score_df += score_df
  job_count += 1

master_score_df /= job_count
master_score_df.index = ['BM25', 'Cos Sim', 'KG']
master_score_df.drop(columns=['course_ncg_at_five',      'course_ncg_at_ten',      'course_ncg_at_twenty',
                         'explanation_ncg_at_five', 'explanation_ncg_at_ten', 'explanation_ncg_at_twenty'],
                     inplace=True)
master_score_df.to_csv(os.path.join(eval_out_path, 'eval_table.csv'))
