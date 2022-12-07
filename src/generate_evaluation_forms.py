# <repo_root>/src/generate_evaluation_forms.py
#
# Author: Jacob Striebel
#
# This script generates the forms needed to perform the system evaluation.
#
# The following input files are required:
#   <repo_root>/data/skills_graph.csv
#   <repo_root>/data/courses_data.csv
#   <repo_root>/data/jobs_data.csv
#   [The cosine_similarity function defined in
#    <repo_root>/src/cosine_similarity.py
#    which is invoked by this script has further data dependencies which
#    are listed in the header of its source file]
#
# The following output files are generated:
#   <repo_root>/eval/out/full_eval.csv
#   <repo_root>/eval/out/blind_eval.csv

import os, copy
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from shutil import rmtree
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, OrGroup
from knowledge_graph import KnowledgeGraph
from cosine_similarity import cosine_similarity
import random

job_queries = [
  'Test Engineer',
  'Summer 2021 Robotics Intern',
  'Electronic Technician',
  'Associate Field Service Representative - Portland',
  'Tester II - Semiconductor Product Validation',
  'Systems Engineer',
  'Bead Blast Technician',
  'Field Engineer',
  'Hardware Design Engineer',
  'Research Scientist'
]

src_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(src_path, os.pardir)
data_path = os.path.join(root_path, 'data')
skills_path = os.path.join(data_path, 'skills_graph.csv')
courses_path = os.path.join(data_path, 'courses_data.csv')
courses_dir_path = os.path.join(data_path, 'courses')
jobs_path = os.path.join(data_path, 'jobs_data.csv')
jobs_dir_path = os.path.join(data_path, 'jobs')
eval_path = os.path.join(root_path, 'eval')
eval_out_path = os.path.join(eval_path, 'out')
full_eval_path = os.path.join(eval_out_path, 'full_eval.csv')
blind_eval_path = os.path.join(eval_out_path, 'blind_eval.csv')

def read_skills():
  global skills_path
  deepest_to_keep = 3
  
  skills_df = pd.read_csv(skills_path)
  print('Total skills of any depth:', len(skills_df))
  skills_df.drop(skills_df[skills_df.depth>deepest_to_keep].index, inplace=True)
  print('Total skills after pruning:', len(skills_df))
  skills_df.rename(columns={'skill_name':'page_title'}, inplace=True)
  skills_df['skill_name'] = [str(title).lower().replace('_', ' ') for title in skills_df['page_title']]
  
  # Remove duplicates based on enwiki_page_id
  duplicate_indices=[]
  enwiki_page_ids_encountered=set()
  for skill in skills_df.iterrows():
    if skill[1].enwiki_page_id not in enwiki_page_ids_encountered:
      enwiki_page_ids_encountered.add(skill[1].enwiki_page_id)
    elif skill[1].enwiki_page_id in enwiki_page_ids_encountered:
      duplicate_indices.append(skill[0])
    else:
      # Bad state
      assert False
  skills_df.drop(duplicate_indices, inplace=True)
  print('Total skills after removing duplicates', len(skills_df))
  
  # Identify the pages that are categories
  indices_to_remove=[]
  categories = set()
  categories_ids = set()
  for skill in skills_df.iterrows():
    if skill[1].page_namespace == 14:
      categories.add(skill[1].skill_name)
      categories_ids.add(skill[1].enwiki_page_id)
  
  # Remap articles whose parent isn't a category into a category
  remap_count=0
  for skill in skills_df.iterrows():
    if (skill[1].enwiki_page_id_of_parent != 0 and
        skill[1].enwiki_page_id_of_parent not in categories_ids):
      suitable_parent_found = False
      for candidate_parent in skills_df.iterrows():
        if (candidate_parent[0] != skill[0] and
            candidate_parent[1].page_namespace == 14 and
            candidate_parent[1].skill_name ==
              skills_df.at[skills_df[skills_df.enwiki_page_id==skill[1].enwiki_page_id_of_parent].index[0], 'skill_name']):
          skills_df.at[skill[0], 'enwiki_page_id_of_parent'] = candidate_parent[0] 
          suitable_parent_found=True
          remap_count+=1
      if not suitable_parent_found:
        print('A suitable parent could not be found for', skill)
  print('Total skills remapped to a new parent', remap_count)

  # Remove articles that duplicate a category
  for skill in skills_df.iterrows():
    if skill[1].page_namespace != 14 and skill[1].skill_name in categories:
      indices_to_remove.append(skill[0])
  skills_df.drop(indices_to_remove, inplace=True)
  print('Total skills after removing articles that have the same name as a category', len(skills_df))
  
  skills_df['skill_name_long'] = [skill_name.replace('_', ' ') for skill_name in skills_df['skill_name']]
  
  return skills_df
  
def read_courses():
  global courses_path
  courses_df = pd.read_csv(courses_path)
  print('There are originally '+str(len(courses_df))+' courses in our dataset')
  print('We are now checking for duplicate course titles')
  courses_to_drop = []
  courses_in_dset = set()
  for course in courses_df.iterrows():
    title = course[1].title
    if title not in courses_in_dset:
      courses_in_dset.add(title)
    else:
      courses_to_drop.append(course[0])
  courses_df.drop(courses_to_drop, inplace=True)
  print('There are '+str(len(courses_df))+' courses after removing duplicates')
  if len(courses_to_drop) > 0:
    print('We are updating the courses csv file so the duplicate courses are removed')
    courses_df.to_csv(courses_path)
    print('Done updating courses csv')
  else:
    print('No duplicate courses found')
  courses_df['Title'] = courses_df['title']
  courses_df['combined_content'] = courses_df['cleaned_content']
  return courses_df

class PreprocessCourses():
  def __init__(self, courses):
    nltk.download('stopwords')
    nltk.download('punkt')
    self._stop_words = set(stopwords.words('english'))
    courses['count_token'] = [None for i in courses.index]
  
  def find_most_frequent_words_in_course_syallabi(self, courses):
    for i in courses.index:
      token_count = dict()
      tokenized_syllabus = word_tokenize(courses.at[i, 'cleaned_content'])
      for token in tokenized_syllabus:
        if token not in self._stop_words and token not in [',', 'num']:
          if token not in token_count:
            token_count[token] = 1
          else:
            token_count[token] += 1
      count_token = []
      for token, count in token_count.items():
        count_token.append((count, token))
      count_token.sort(reverse=True)
      courses.at[i, 'count_token'] = count_token

def read_jobs():
  global jobs_path
  jobs_df = pd.read_csv(jobs_path)
  print('There are '+str(len(jobs_df))+' jobs in our original dataset')
  jobs_to_drop = []
  jobs_in_dset = set()
  for job in jobs_df.iterrows():
    if job[1].jobKey not in jobs_in_dset:
      jobs_in_dset.add(job[1].jobKey)
    else:
      jobs_to_drop.append(job[0])
  jobs_df.drop(jobs_to_drop, inplace=True)
  print('There are '+str(len(jobs_df))+' jobs after removing duplicates')
  if len(jobs_to_drop) > 0:
    print('We are updating the jobs csv file so the duplicate jobs (uniqueness check against job key) are removed')
    jobs_df.to_csv(jobs_path)
    print('Done updating jobs csv')
  else:
    print('No duplicate jobs found')
  print('Checking for leading or trailing whitespace on job titles')
  updated_titles = 0
  for i in jobs_df.index:
    original_title = jobs_df.at[i, 'jobTitle']
    stripped_title = original_title.strip()
    if original_title!=stripped_title:
      jobs_df.at[i, 'jobTitle'] = stripped_title
      updated_titles+=1
  if updated_titles>0:
    print('We are updating the jobs csv file to remove leading and/or trailing whitespace on job titles')
    print(str(updated_titles) + ' titles will be updated')
    jobs_df.to_csv(jobs_path)
    print('Done updating jobs csv')
  else:
    print('No title had leading or trailing whitespace')    
  jobs_df['fullDescription'] = jobs_df['job_content']
  jobs_df['combined_content'] = jobs_df['job_content']
  return jobs_df

class WhooshSearchJobs():
  def __init__(self, jobs_df):
    global jobs_dir_path
    # First, create Schema and Index object (initial configuration).
    schema = Schema(title=TEXT(stored=True),
                    jobKey=TEXT(stored=True),
                    body=TEXT(stored=True,
                    analyzer=StemmingAnalyzer()))
    indx_dir = os.path.join(jobs_dir_path, 'index')
    try:
      rmtree(jobs_dir_path)
    except:
      pass
    try:
      os.rmdir(jobs_dir_path)
    except:
      pass
    try:
      os.makedirs(indx_dir)
    except:
      pass
    ix = create_in(indx_dir, schema)
    self._ix = open_dir(indx_dir)
    # Next, create the IndexWriter object (make the job postings searchable).
    writer = ix.writer()
    for idx in jobs_df.index:
      jobTitle = jobs_df.at[idx, 'jobTitle'].replace('\n', '')
      jobKey   = jobs_df.at[idx, 'jobKey']
      desc     = jobs_df.at[idx, 'combined_content']
      writer.add_document(title=jobTitle, jobKey=jobKey, body=desc)
    writer.commit()
  
  def search(self, job_query, jobs_df):
    parser = QueryParser('body', schema=self._ix.schema, group=OrGroup)
    myquery = parser.parse(job_query)
    with self._ix.searcher() as searcher:
      results = searcher.search(myquery, limit=100)
      if len(results)==0:
        jobs_df['score']= 1.  ## Each job will have the same probability.
      else:
        jobs_df['score']= 0.  ## Probability is determined by the user-provided search string.
      #print('These are the search matches.')
      for i, hit in enumerate(results):
        score= results.score(i)
        #print('score: '+str(score)+'; title: '+hit['title'])
        job_index = jobs_df[jobs_df.jobKey==hit['jobKey']].index
        if len(job_index)!=1:
          print('Data warning: Job key', hit['jobKey'], 'appears multiple times (', len(job_index), ') in the jobs dataset.')
        job_index = job_index[0]
        jobs_df.at[job_index, 'score'] = score
      scores_sum = sum(jobs_df['score'])
      #print('The sum of scores is: '+str(scores_sum))
      for job_index in jobs_df.index:
        prob= jobs_df.at[job_index, 'score'] / scores_sum
        jobs_df.at[job_index, 'activation']= prob

class WhooshSearchCourses():
  def __init__(self, courses_df):
    global courses_dir_path
    # First, create Schema and Index object (initial configuration).
    schema = Schema(title=TEXT(stored=True),
                    course_id=NUMERIC(int, 64, signed=True, stored=True),
                    body=TEXT(stored=True,
                    analyzer=StemmingAnalyzer()))
    indx_dir = os.path.join(courses_dir_path, 'index')
    try:
      rmtree(courses_dir_path)
    except:
      pass
    try:
      os.rmdir(courses_dir_path)
    except:
      pass
    try:
      os.makedirs(indx_dir)
    except:
      pass
    ix = create_in(indx_dir, schema)
    self._ix = open_dir(indx_dir)
    # Next, create the IndexWriter object (make the job postings searchable).
    writer = ix.writer()
    for idx in courses_df.index:
      crs_title = courses_df.at[idx, 'title'].replace('\n', '')
      crs_id    = courses_df.at[idx, 'course_id']
      desc      = courses_df.at[idx, 'cleaned_content']
      writer.add_document(title=crs_title, course_id=crs_id, body=desc)
    writer.commit()
  
  def search(self, job_query, courses_df):
    # As a baseline approach we search the courses dataset with a job query using a text-based search
    parser = QueryParser('body', schema=self._ix.schema, group=OrGroup)
    myquery = parser.parse(job_query)
    with self._ix.searcher() as searcher:
      results = searcher.search(myquery, limit=20)
      if len(results)==0:
        courses_df['score']= 1.  # Each job will have the same probability.
      else:
        courses_df['score']= 0.  # Probability is determined by the user-provided search string.
      #print('These are the search matches.')
      for i, hit in enumerate(results):
        score= results.score(i)
        #print('score: '+str(score)+'; title: '+hit['title'])
        course_index = courses_df[courses_df.course_id==hit['course_id']].index
        if len(course_index)!=1:
          print('Data warning: course_id', hit['course_id'], 'appears multiple times (', len(course_index), ') in the jobs dataset.')
        course_index = course_index[0]
        courses_df.at[course_index, 'score'] = score
      scores_sum = sum(courses_df['score'])
      #print('The sum of scores is: '+str(scores_sum))
      courses_df.sort_values('score', inplace=True)

decision=input('Running this script will clobber any previously '+
                 'generated evaluation files in:\n'+
                 '    '+eval_out_path+'\n'+
                 'Would you like to continue?\n'+
                 'Enter [Y]es to run, [n]o to abort > ')
while True:
  if decision in ['Y', 'Yes']:
    break
  elif decision in ['n', 'no']:
    print('Script aborted')
    quit()
  else:
    decision = input('Please enter [Y]es or [n]o > ')

try:
  rmtree(eval_out_path)
except:
  pass
try:
  os.rmdir(eval_out_path)
except:
  pass
try:
  os.makedirs(eval_out_path)
except:
  pass

print('Reading and preprocessing skills data')
skills = read_skills()
print('Done reading and preprocessing skills')
print('Reading and preprocessing courses data')
courses = read_courses()
course_preprocessor = PreprocessCourses(courses)
course_preprocessor.find_most_frequent_words_in_course_syallabi(courses)
print('Done reading and preprocessing courses data')
print('Reading and preprocessing jobs data')
jobs = read_jobs()
print('Done reading and preprocessing jobs data')
print('Setting up indexing of the job postings')
job_searcher = WhooshSearchJobs(jobs)
print('Done setting up indexing of the job postings')
print('Executing: build KG, the random walk using course skills, and the cosine similarity baseline')
job_title__skill_result, job_title__cosim_result = cosine_similarity(job_queries, approach='course_skills')
print('Done executing: build KG, the random walk using course skills, and the cosine similarity baseline')
print('Rebuilding the knowledge graph')
kg = KnowledgeGraph(skills, courses, jobs)
print('Done rebuilding the knowledge graph')
print('Assigning edge weights within the graph')
kg.update_kg_edges_with_tfidf()
print('Done assigning edge weights')
print('Setting up indexing of the course syllabi')
course_searcher = WhooshSearchCourses(courses)
print('Done setting up indexing of the course syllabi')
print('Executing queries')
column_labels = {
  'random_coefficient':[],
  'approach':[],
  'rank':[],
  'second_index':[],
  'job_query':[],
  'course_relevance':[],
  'course_name':[],
  'explanation_quality':[],
  'explanation_skill_terms':[],
  'explanation':[],
  'full_explanation':[]
}
results = pd.DataFrame(data=copy.deepcopy(column_labels))
next_i = 0
for job_query_index, job_query_string in enumerate(job_queries):
  print('Query', str(job_query_index+1), '/', str(len(job_queries)))
  new_results = pd.DataFrame(data=copy.deepcopy(column_labels))
  print('    Executing BM25 baseline')
  course_searcher.search(job_query_string, courses)
  for i in range(1,21):
    course_index = courses.index[-i]
    course_score = courses.at[course_index, 'score']
    course_title = courses.at[course_index, 'title']
    
    #print('score: '+str(course_score)+'; title: '+course_title)
    
    count_token = courses.at[course_index, 'count_token']
    skill_terms = '      '.join([token for count, token in count_token[:4]])
    
    expl = "In '"+course_title+"' you will learn '"+count_token[0][1]+"' and '"+count_token[1][1]+"' which are important for the job '"+job_query_string+"'."
    
    new_results.loc[next_i]=[-1., 'X', i, -1, job_query_string, '', courses.at[course_index, 'title'], '', skill_terms, expl, '']
    next_i+=1
  
  print('    Done executing BM25 baseline')
  print('    Executing KG approach')
  job_searcher.search(job_query_string, jobs)
  kg.execute_job_query(job_query_string)
  for i in range(1, 21):
    skills, expl_full, expl_abrv = kg.explain(job_query_string, -i)
    course_index = courses.index[-i]
    
    new_results.loc[next_i]=[-1., 'Z', i, -1, job_query_string, '', courses.at[course_index, 'title'], '', '      '.join(skills), expl_abrv, expl_full]
    next_i+=1
  
  print('    Done executing KG approach')
  print('    Processing results for cosine similarity approach')
  
  cosim_result = job_title__cosim_result[job_query_string]
  cosim_result['new_index'] = range(next_i, next_i+20)
  next_i+=20
  cosim_result.set_index('new_index', inplace=True, verify_integrity=True)
  new_results = pd.concat([new_results, cosim_result])
  
  print('    Cosine similarity done')
  
  # Randomize the order of the recommended courses for the current job query
  course_title__random_number = dict()
  for i in new_results.index:
    course_title = new_results.at[i, 'course_name']
    if course_title not in course_title__random_number:
      course_title__random_number[course_title] = random.random()
    new_results.at[i, 'random_coefficient'] = course_title__random_number[course_title]
  new_results.sort_values('random_coefficient', inplace=True)
  
  results = pd.concat([results, new_results])

print('All queries complete')

for second_index, first_index in enumerate(results.index):
  results.at[first_index, 'second_index'] = second_index
results.to_csv(full_eval_path)
results.drop(columns=['random_coefficient', 'approach', 'rank', 'explanation_skill_terms', 'full_explanation'], inplace=True)
results.set_index('second_index', inplace=True, verify_integrity=True)
results.to_csv(blind_eval_path)

print('Evaluation form generation complete')
