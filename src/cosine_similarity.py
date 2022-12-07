# <repo_root>/src/cosine_similarity.py
#
# Authors: Rebecca Myers, Jacob Striebel
#
# This source file provides the function that is used to run the cosine
#   similarity baseline.
#
# This source file has the following data dependencies:
#   <repo_root>/data/jobs_data.csv
#   <repo_root>/data/courses_data.csv
#   <repo_root>/data/job_skills.csv
#   <repo_root>/data/course_skills.csv
#   <repo_root>/data/course_keywords.csv

import pandas as pd
import numpy as np
import os
from collections import namedtuple, defaultdict, Counter
import copy

def cosine_similarity(job_titles, approach):
    
    assert approach in ['course_skills', 'course_keywords']
    
    src_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(src_path, os.pardir)
    data_path = os.path.join(root_path, 'data')
    jobs_path = os.path.join(data_path, 'jobs_data.csv')
    courses_path = os.path.join(data_path, 'courses_data.csv')
    job_skills_path = os.path.join(data_path, 'job_skills.csv')
    course_skills_path = os.path.join(data_path, 'course_skills.csv')
    course_keywords_path = os.path.join(data_path, 'course_keywords.csv')
    
    micro_jobs_data = pd.read_csv(jobs_path)
    courses_data    = pd.read_csv(courses_path)
    
    courses_data = courses_data.dropna()
    
    micro_jobs_df = pd.DataFrame(micro_jobs_data)
    
    courses_df = pd.DataFrame(courses_data)
    
    job_skills = pd.read_csv(job_skills_path)
    
    job_skills_df = pd.DataFrame(job_skills).drop_duplicates().dropna()
    
    course_skills = pd.read_csv(course_skills_path)
    
    crs_skills = np.array(course_skills)
    crs_skills_df = pd.DataFrame(crs_skills).drop_duplicates().dropna()
    
    course_keywords = pd.read_csv(course_keywords_path)
    
    crs_keywords = np.array(course_keywords)
    crs_keywords_df = pd.DataFrame(crs_keywords).drop_duplicates().dropna()
    
    set_skills = set(crs_skills_df[0].tolist())
    crs_skills = np.array(list(set_skills))
    
    # Search course skills in course content
    def skill_search(data):
        temp = [(skill in data) for skill in crs_skills]
        return crs_skills[temp]
    
    temp = courses_df['cleaned_content'].apply(skill_search)
    
    # Matching course skills to courses
    crs_sk_match_to_crs_content_df = pd.DataFrame()
    crs_sk_match_to_crs_content_df['course_id'] = courses_df['course_id']
    crs_sk_match_to_crs_content_df['title'] = courses_df['title']
    crs_sk_match_to_crs_content_df['skills'] = temp
    crs_sk_match_to_crs_content_df['skill_count'] = temp.apply(len)
    
    # Course keywords to course content
    set_keywords = set(crs_keywords_df[0].tolist())
    crs_keywords = np.array(list(set_keywords))
    
    # Search course keywords in course content
    def keyword_search(data):
        temp1 = [(keyword in data) for keyword in crs_keywords]
        return crs_keywords[temp1]
    
    temp1 = courses_df['cleaned_content'].apply(keyword_search)
    
    # Matching course keywords to courses
    crs_kw_match_to_crs_content_df = pd.DataFrame()
    crs_kw_match_to_crs_content_df['course_id'] = courses_df['course_id']
    crs_kw_match_to_crs_content_df['title'] = courses_df['title']
    crs_kw_match_to_crs_content_df['keywords'] = temp1
    crs_kw_match_to_crs_content_df['keywords_count'] = temp1.apply(len)
    
    # Job skill matching to course content
    job_set_skills = set(job_skills_df['0'].tolist())
    job_skills = np.array(list(job_set_skills))
    
    # Search job skills in course content
    def skill_search(data):
        temp2 = [(skill in data) for skill in job_skills]
        return job_skills[temp2].tolist()
    
    temp2 = courses_df['cleaned_content'].apply(skill_search)
    
    # Matching job skills to courses
    job_sk_match_to_crs_content_df = pd.DataFrame()
    job_sk_match_to_crs_content_df['course_id'] = courses_df['course_id']
    job_sk_match_to_crs_content_df['title'] = courses_df['title']
    job_sk_match_to_crs_content_df['skills'] = temp2
    job_sk_match_to_crs_content_df['skill_count'] = temp2.apply(len)
    
    job_sk_leftovers = set(job_skills) - set(job_sk_match_to_crs_content_df['skills'].sum())
    
    job_sk_in_courses = set(job_sk_match_to_crs_content_df['skills'].sum())
    
    # Search job skills in course content (temp) or job post (temp2)
    def skill_search(data):
        temp = [(skill in data) for skill in job_skills]
        return job_skills[temp]
    
    temp = courses_df['cleaned_content'].apply(skill_search)
    temp2 = micro_jobs_df['job_content'].apply(skill_search)
    
    # Matching job skills to courses
    job_sk_match_to_crs_content_df = pd.DataFrame()
    job_sk_match_to_crs_content_df['course_id'] = courses_df['course_id']
    job_sk_match_to_crs_content_df['title'] = courses_df['title']
    job_sk_match_to_crs_content_df['skills'] = temp
    job_sk_match_to_crs_content_df['skill_count'] = temp.apply(len)
    
    # Matching job skills to jobs
    job_sk_match_to_job_content_df = pd.DataFrame()
    job_sk_match_to_job_content_df['jobKey'] = micro_jobs_df['jobKey']
    job_sk_match_to_job_content_df['title'] = micro_jobs_df['jobTitle']
    job_sk_match_to_job_content_df['skills'] = temp2
    job_sk_match_to_job_content_df['skill_count'] = temp2.apply(len)
    
    # Record is a tuple created to assist readability
    # Name can be job, crs, or skill 
    # Type can be job skill, crs skill, job or crs
    Record = namedtuple('Record',['name', 'type'])
    
    # Example
    tmp = Record('Field Test Engineer', 'job')
    
    mapper1 = defaultdict(list)
    
    # Maps course to course skills
    for i, row in crs_sk_match_to_crs_content_df.iterrows():
        key = Record(row['title'], 'course')
        values = [Record(skill, 'crs skill') for skill in row['skills']]
        mapper1[key] += values
        # Maps course skills to course
        for value in values:
            mapper1[value].append(key)
            
    # Maps course to job skills
    for i, row in job_sk_match_to_crs_content_df.iterrows():
        key = Record(row['title'], 'course')
        values = [Record(skill, 'job skill') for skill in row['skills']]
        mapper1[key] += values
        # Maps job skills to course
        for value in values:
            mapper1[value].append(key)
    
    # Maps job to job skills
    for i, row in job_sk_match_to_job_content_df.iterrows():
        key = Record(row['title'], 'job')
        values = [Record(skill, 'job skill') for skill in row['skills']]
        mapper1[key] += values
        # Maps job skills to job
        for value in values:
            mapper1[value].append(key)
    
    mapper2 = defaultdict(list)
    
    # Maps course to course keywords
    for i, row in crs_kw_match_to_crs_content_df.iterrows():
        key = Record(row['title'], 'course')
        values = [Record(keyword, 'crs keyword') for keyword in row['keywords']]
        mapper2[key] += values
        # Maps course skills to course
        for value in values:
            mapper2[value].append(key)
            
    # Maps course to job skills
    for i, row in job_sk_match_to_crs_content_df.iterrows():
        key = Record(row['title'], 'course')
        values = [Record(skill, 'job skill') for skill in row['skills']]
        mapper2[key] += values
        # Maps job skills to course
        for value in values:
            mapper2[value].append(key)
    
    # Maps job to job skills
    for i, row in job_sk_match_to_job_content_df.iterrows():
        key = Record(row['title'], 'job')
        values = [Record(skill, 'job skill') for skill in row['skills']]
        mapper2[key] += values
        # Maps job skills to job
        for value in values:
            mapper2[value].append(key)
    
    def randomstep2(record, mapper):
        connections = mapper[record]
        return connections[np.random.choice(len(connections))]
    
    def randomwalk2(orig_record, threshold, mapper, cutoff=10000):
        record = orig_record
        records = []
        while len(records)<cutoff: # It takes 100000 random walks
            # For normer
            if threshold is None:
                record = randomstep2(record, mapper)
                records.append(record)
            # For nomral run   
            else:
                while np.random.random()<threshold: # The smaller this number the smaller the step i.e., closer neighbors
                    record = randomstep2(record, mapper)
                    records.append(record)
                record = orig_record
        return records
    
    # Generate normer
    def build_normer(mapper):
        results_normer = Counter(randomwalk2(Record('Chemical Engineer', 'job'), None, mapper, 10000000))
        raw_count = sorted([(v,k.name) for (k,v) in results_normer.items() if k.type=='course'])
        normer = dict([(k,v) for (v,k) in raw_count])
        return normer
    
    def recommend(job_id, threshold, mapper, normer):
        job_title = job_sk_match_to_job_content_df['title'][job_id]
        results2 = Counter(randomwalk2(Record(job_title, 'job'), threshold, mapper))
        results2 = dict([(k,v/normer[k.name]*100) for (k, v) in results2.items() if k.type=='course'])
        RW_matches = sorted([(v,k.name) for (k,v) in results2.items() if k.type=='course']) # Return all results, not just top ten [-10:]
        job_sk_match_to_crs_content_df['simularity'] = crs_job_sk_vec @ job_job_sk_vec[job_id]
        job_sk_match_to_crs_content_df[job_sk_match_to_crs_content_df['title'].isin([el[1] for el in RW_matches])]
        summary = job_sk_match_to_crs_content_df[job_sk_match_to_crs_content_df['title'].isin([el[1] for el in RW_matches])]
        summary['count'] = summary['title'].map(dict([(k,v) for (v,k) in RW_matches]))
        return summary.sort_values('count').iloc[::-1]
    
    # Encode job skills into a vector space for cosine similarity evaluation
    def encode_job_skills(row):
        return np.array([(skill in row)for skill in job_sk_in_courses]).astype(int)
    
    job_job_skill_vec = np.vstack(job_sk_match_to_job_content_df['skills'].apply(encode_job_skills).values)
    crs_job_skill_vec = np.vstack(job_sk_match_to_crs_content_df['skills'].apply(encode_job_skills).values)
    
    crs_job_sk_vec = crs_job_skill_vec/np.linalg.norm(crs_job_skill_vec, axis=1).reshape(-1,1)
    job_job_sk_vec = job_job_skill_vec/np.linalg.norm(job_job_skill_vec, axis=1).reshape(-1,1)
    
    mapper = None
    
    if approach=='course_skills':
      mapper = mapper1
    elif approach=='course_keywords':
      mapper = mapper2
    else:
      # Bad state
      assert False
    
    assert mapper!=None
    
    normer = build_normer(mapper)
    
    assert normer!=None
    
    def get_job_id(job_title):
      indices_with_job_title = micro_jobs_df.iloc[[micro_jobs_df.at[i, 'jobTitle']==job_title for i in micro_jobs_df.index]].index
      # Below used to be an assert equal to one, but there are many jobs in the
      #   dataset that have the same name but different job key; so, we take
      #   the first job with the sought `job_title'
      assert len(indices_with_job_title)>=1 
      return indices_with_job_title[0]
    
    def get_top_two_skills(skills_list):
      list_to_sort = [(len(skill), skill) for skill in skills_list]
      list_to_sort.sort(reverse=True)
      return list_to_sort[0][1], list_to_sort[1][1]
    
    job_title__graph_result = dict()
    job_title__cosim_result = dict()
    
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
    
    for job_title in job_titles:
      
      recommendation = recommend(get_job_id(job_title), 0.4, mapper, normer)
      
      top_twenty_for_graph = recommendation.iloc[[recommendation.at[i, 'skill_count']>=2 for i in recommendation.index]][:20]
      
      assert len(top_twenty_for_graph.index)==20
      
      graph_result = pd.DataFrame(data=copy.deepcopy(column_labels))
      
      for zero_based_index, recommendation_df_index in enumerate(top_twenty_for_graph.index):
        
        skill0, skill1 = get_top_two_skills(top_twenty_for_graph.at[recommendation_df_index, 'skills'])
        
        random_coefficient      = -1.
        this_approach           = 'C' if approach=='course_skills' else 'E'
        rank                    = zero_based_index+1
        second_index            = -1
        job_query               = job_title
        course_relevance        = ''
        course_name             = top_twenty_for_graph.at[recommendation_df_index, 'title']
        explanation_quality     = ''
        explanation_skill_terms = skill0+'      '+skill1
        explanation             = "In '"+course_name.strip()+"' you will learn '"+skill0+"' and '"+skill1+"' which are important for the job '"+job_title.strip()+"'."
        full_explanation        = ''
        
        graph_result.loc[zero_based_index] = [random_coefficient, this_approach, rank, second_index, job_query, course_relevance, course_name, explanation_quality, explanation_skill_terms, explanation, full_explanation]
      
      # If we're currently doing the course skills run, we also obtain the cosine similarity predictions here
      cosim_result = None
      if approach=='course_skills':
        
        recommendation = recommendation.sort_values('simularity', ascending=False, inplace=False)
        top_twenty_for_cosim = recommendation.iloc[[recommendation.at[i, 'skill_count']>=2 for i in recommendation.index]][:20]
        
        assert len(top_twenty_for_cosim.index)==20
        
        cosim_result = pd.DataFrame(data=copy.deepcopy(column_labels))
        
        for zero_based_index, recommendation_df_index in enumerate(top_twenty_for_cosim.index):
          
          skill0, skill1 = get_top_two_skills(top_twenty_for_cosim.at[recommendation_df_index, 'skills'])
          
          random_coefficient      = -1.
          this_approach           = 'Y'
          rank                    = zero_based_index+1
          second_index            = -1
          job_query               = job_title
          course_relevance        = ''
          course_name             = top_twenty_for_cosim.at[recommendation_df_index, 'title']
          explanation_quality     = ''
          explanation_skill_terms = skill0+'      '+skill1
          explanation             = "In '"+course_name.strip()+"' you will learn '"+skill0+"' and '"+skill1+"' which are important for the job '"+job_title.strip()+"'."
          full_explanation        = ''
          
          cosim_result.loc[zero_based_index] = [random_coefficient, this_approach, rank, second_index, job_query, course_relevance, course_name, explanation_quality, explanation_skill_terms, explanation, full_explanation]
      
      job_title__graph_result[job_title] = graph_result
      job_title__cosim_result[job_title] = cosim_result
    
    return job_title__graph_result, job_title__cosim_result
