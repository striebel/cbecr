# <repo_root>/src/knowledge_graph.py
#
# Author: Jacob Striebel
#
# This source file defines the KnowledgeGraph class which is used to execute
#   our proposed algorithm for career-based explainable course recommendation

import networkx as nx
import numpy as np
import math

SKILL_NAME_COLUMN = 'skill_name'
COURSE_DESCRIPTION_COLUMN = 'combined_content'
COURSE_TITLE_COLUMN = 'Title'
JOB_DESCRIPTION_COLUMN = 'combined_content'
JOB_TITLE_COLUMN = 'jobTitle'

class KnowledgeGraph(nx.DiGraph):
  """
  Class representing a heterogeneous knowledge graph whose nodes are skills,
  courses, and jobs.

  Parameters
  ----------
  skills : pandas.DataFrame
    A DataFrame whose rows are the skills that will be searched for in the
    course descriptions and in the job descriptions; each skill will become a
    node in the graph with an edge between itself and each individual course
    and each individual job in whose description it appears.
  courses : pandas.DataFrame
    A DataFrame whose each row is a course.
  jobs : pandas.DataFrame
    A DataFrame whose each row is a job posting.
  """
  def __init__(self, skills, courses, jobs):
    super().__init__() 
    skills['id_in_graph']  = range(0,                        len(skills))
    courses['id_in_graph'] = range(len(skills),              len(skills)+len(courses))
    jobs['id_in_graph']    = range(len(skills)+len(courses), len(skills)+len(courses)+len(jobs))
    skills.set_index('id_in_graph', inplace=True, verify_integrity=True)
    courses.set_index('id_in_graph', inplace=True, verify_integrity=True)
    jobs.set_index('id_in_graph', inplace=True, verify_integrity=True)
    self.add_nodes_from(skills.index)
    self.add_nodes_from(courses.index)
    self.add_nodes_from(jobs.index)
    self.add_edges_from([edge for edge_pair in [(
      (getattr(course, 'Index'), getattr(skill, 'Index'), {'label':'teaches', 'weight':.0}),
      (getattr(skill, 'Index'), getattr(course, 'Index'), {'label':'is taught by', 'weight':9999})) # Will be set by "update_kg_edges_with_tfidf" method
      for course in courses.itertuples() for skill in skills.itertuples()
      if getattr(skill, SKILL_NAME_COLUMN) in getattr(course, COURSE_DESCRIPTION_COLUMN)]
      for edge in edge_pair
    ])
    self.add_edges_from([edge for edge_pair in [(
      (getattr(job, 'Index'), getattr(skill, 'Index'), {'label':'requires', 'weight':9999}), # Will be set by "update_kg_edges_with_tfidf" method
      (getattr(skill, 'Index'), getattr(job, 'Index'), {'label':'is required by', 'weight':.0}))
      for job in jobs.itertuples() for skill in skills.itertuples()
      if getattr(skill, SKILL_NAME_COLUMN) in getattr(job, JOB_DESCRIPTION_COLUMN)]
      for edge in edge_pair
    ])
    
    for skill in skills.itertuples():
      if skill.enwiki_page_id_of_parent!=0:
        self.add_edges_from([(getattr(skill, 'Index'),
          skills[skills.enwiki_page_id==skill.enwiki_page_id_of_parent].index[0],
          {'label':'is child of', 'weight':0.5})]
        )
        self.add_edges_from([(skills[skills.enwiki_page_id==skill.enwiki_page_id_of_parent].index[0],
          getattr(skill, 'Index'),
          {'label':'is parent of', 'weight':0.5})]
        )
    
    self._skills = skills
    self._courses = courses
    self._jobs = jobs
  
  def node_to_string(self, id_in_graph):
    def get_node_name(id_in_graph):
      if id_in_graph in self._skills.index:
        name= (str(self._skills.at[id_in_graph, 'page_title']) + ', enwiki_page_id: ' +
               str(self._skills.at[id_in_graph, 'enwiki_page_id']))
        return name
      elif id_in_graph in self._courses.index:
        return self._courses.at[id_in_graph, COURSE_TITLE_COLUMN]
      elif id_in_graph in self._jobs.index:
        name= (self._jobs.at[id_in_graph, JOB_TITLE_COLUMN] + ', jobKey:' +
               self._jobs.at[id_in_graph, 'jobKey'])
        return name
      else:
        assert False # Bad/unexpected state
    adj = self[id_in_graph]
    keys = [x for x in adj]
    desc = ''
    for key in keys:
      desc += adj[key]['label'] + ' (edge weight: ' + str(adj[key]['weight']) + ') ' + get_node_name(key) + '\n'
    return desc    
  
  def inspect_skill(self, enwiki_page_id):
    # Convert enwiki page id to node id in the graph
    enwiki_page_id = int(enwiki_page_id)
    enwiki_page_ids = [int(x) for x in self._skills.enwiki_page_id]
    id_in_graph=-1
    if enwiki_page_id not in enwiki_page_ids:
      return 'Bad page id.\n'
    else:
      id_in_graph = enwiki_page_ids.index(enwiki_page_id)
      return 'Skill name: '+self._skills.at[id_in_graph, 'skill_name']+'\n'+self.node_to_string(id_in_graph)
  
  def inspect_course(self, course_title):
    indices = self._courses[self._courses[COURSE_TITLE_COLUMN]==course_title].index
    if len(indices)<=0:
      return 'No such course title.\n'
    elif len(indices)>1:
      return 'Data error: More than one course has this title.\n'
    elif len(indices)==1:
      return self.node_to_string(indices[0])
    else:
      assert False # Bad/unexpected state
  
  def inspect_job(self, indeed_job_key):
    indices = self._jobs[self._jobs['jobKey']==indeed_job_key].index
    if len(indices)<=0:
      return 'No such job key.\n'
    elif len(indices)>1:
      return 'Data error: More than one job has this key.\n'
    elif len(indices)==1:
      id_in_graph = indices[0]
      return 'Job title: '+str(self._jobs.at[id_in_graph, 'jobTitle'])+'\n'+self.node_to_string(id_in_graph)
    else:
      assert False # Bad/unexpected state
  
  def update_kg_edges_with_tfidf(self):
    # First we calculate the number of course descriptions each skill appeared in
    # and the number of job postings each skill appeared in.
    self._skills['course_idf'] = [len([id_y_in_graph for id_y_in_graph in self[id_x_in_graph]
      if id_y_in_graph in self._courses.index]) for id_x_in_graph in self._skills.index
    ]
    self._skills['job_idf'] = [len([id_y_in_graph for id_y_in_graph in self[id_x_in_graph]
      if id_y_in_graph in self._jobs.index]) for id_x_in_graph in self._skills.index
    ]
    # Next we calculate the idf for each skill in the courses corpus and in the
    # jobs corpus.
    n_courses = len(self._courses.index)
    n_jobs = len(self._jobs.index)
    self._skills['course_idf'] = [math.log10(float(n_courses) / (1+n_courses_where_skill_appears))
      for n_courses_where_skill_appears in self._skills['course_idf']
    ]
    self._skills['job_idf'] = [math.log10(float(n_jobs) / (1+n_jobs_where_skill_appears))
      for n_jobs_where_skill_appears in self._skills['job_idf']
    ]
    # Any inverse document frequency that is zero or negative we set to a small positive number.
    for i in range(len(self._skills.index)):
      if self._skills.at[i, 'course_idf'] <= 0:
        self._skills.at[i, 'course_idf'] = .00001
      if self._skills.at[i, 'job_idf'] <= 0:
        self._skills.at[i, 'job_idf'] = 0.00001
    # Finally, we calculate each tf-idf and update every graph edge with tf-idf
    # as its weight.
    for skill_id_in_graph in self._skills.index:
      for adjacent_id_in_graph in self[skill_id_in_graph]:
       if adjacent_id_in_graph not in self._skills.index:
        term_freq = 0
        inv_doc_freq = 0
        if adjacent_id_in_graph in self._courses.index:
          term_freq = self._courses.at[adjacent_id_in_graph, COURSE_DESCRIPTION_COLUMN].count(
            self._skills.at[skill_id_in_graph, SKILL_NAME_COLUMN]
          )
          assert term_freq > 0 # Otherwise these two nodes shouldn't be adjacent.
          inv_doc_freq = self._skills.at[skill_id_in_graph, 'course_idf']
        elif adjacent_id_in_graph in self._jobs.index:
          term_freq = self._jobs.at[adjacent_id_in_graph, JOB_DESCRIPTION_COLUMN].count(
            self._skills.at[skill_id_in_graph, SKILL_NAME_COLUMN]
          )
          assert term_freq > 0 # Otherwise these two nodes shouldn't be adjacent.
          inv_doc_freq = self._skills.loc[skill_id_in_graph]['job_idf']
        else:
          assert False # Bad/unexpected state
        if inv_doc_freq <= 0:
          print('inv_doc_freq was <= 0, in particular:', inv_doc_freq)
          print('Term freq:', term_freq)
          print('Skill id:', skill_id_in_graph)
          print('Skill name:', "'%s'" % self._skills.at[skill_id_in_graph, SKILL_NAME_COLUMN])
          if adjacent_id_in_graph in self._courses.index:
            print('Course id:', adjacent_id_in_graph)
            print('Course name:', self._courses.at[adjacent_id_in_graph, COURSE_TITLE_COLUMN])
           #print('Course description:', self._courses.at[adjacent_id_in_graph, COURSE_DESCRIPTION_COLUMN].split())
          else:
            print('Job id:', adjacent_id_in_graph)
            print('Job name:', self._jobs.at[adjacent_id_in_graph, JOB_TITLE_COLUMN])
          #assert False
          inv_doc_freq = .00001
        tf_idf = float(term_freq) * inv_doc_freq
        if adjacent_id_in_graph in self._courses.index:
          self.edges[skill_id_in_graph, adjacent_id_in_graph]['weight'] = tf_idf #1./tf_idf
        elif adjacent_id_in_graph in self._jobs.index:
          self.edges[adjacent_id_in_graph, skill_id_in_graph]['weight'] = tf_idf #1./tf_idf
        else:
          assert False # Bad/unexpected state
  
  def execute_job_query(self, job_query_string):
    self._skills['accumulated_activation']=.0
    for i in range(5):
      self._skills['starting_activation'] = self._skills['accumulated_activation']
      self._skills['accumulated_activation'] = .0
      self._skills['activation_contributions'] = [[] for x in range(len(self._skills.index))]
      self._courses['activation'] = .0
      self._courses['activation_contributions'] = [[] for x in range(len(self._courses.index))]
      for job_id_in_graph in self._jobs.index:
        for adj_id_in_graph in self[job_id_in_graph]:
          assert adj_id_in_graph in self._skills.index
          contribution = self.edges[job_id_in_graph, adj_id_in_graph]['weight'] * self._jobs.at[job_id_in_graph, 'activation']
          self._skills.at[adj_id_in_graph, 'accumulated_activation'] += contribution
          self._skills.at[adj_id_in_graph, 'activation_contributions'].append((contribution, 'job', job_id_in_graph, '  job '+self._jobs.at[job_id_in_graph, 'jobKey']+' contributed '+str(contribution)))
      for skill_id_in_graph in self._skills.index:
        for adj_id_in_graph in self[skill_id_in_graph]:
          contribution= self.edges[skill_id_in_graph, adj_id_in_graph]['weight'] * self._skills.at[skill_id_in_graph, 'starting_activation']
          contribution_tuple= (contribution, 'skill', skill_id_in_graph, '  skill '+str(self._skills.at[skill_id_in_graph, 'enwiki_page_id'])+' contributed '+str(contribution))
          if adj_id_in_graph in self._skills.index:
            self._skills.at[adj_id_in_graph, 'accumulated_activation'] += contribution
            self._skills.at[adj_id_in_graph, 'activation_contributions'].append(contribution_tuple)
          elif adj_id_in_graph in self._courses.index:
            self._courses.at[adj_id_in_graph, 'activation'] += contribution
            self._courses.at[adj_id_in_graph, 'activation_contributions'].append(contribution_tuple)
          else:
            assert adj_id_in_graph in self._jobs.index
    self._courses.sort_values('activation', inplace=True)
  
  # course_offset is -1 for the first recommendation, -2 for the second recommendation, etc.
  def explain(self, job_query_string, course_offset):
    # Make recommendation and give explanation.
    
    def obtain_contributions(id_in_graph):
      contribution_tuples = []
      if id_in_graph in self._courses.index:
        contribution_tuples = self._courses.at[id_in_graph, 'activation_contributions']
      elif id_in_graph in self._skills.index:
        contribution_tuples = self._skills.at[id_in_graph, 'activation_contributions']      
      else:
        assert False # Bad/unexpected state
      assert len(contribution_tuples)>=1
      contribution_tuples.sort(reverse=True)
      for i, t in zip(reversed(range(len(contribution_tuples))), reversed(contribution_tuples)):
        if t[0]==.0 or t[1]=='job':
          del contribution_tuples[i]
      return contribution_tuples
    
    def get_candidate_explanation_pairs(direct_contribution_tuples):
      candidate_explanation_pairs = []
      
      CONTRIBUTED_ACTIVATION = 0
      NODE_TYPE    = 1
      ID_IN_GRAPH  = 2
      
      for ct0 in direct_contribution_tuples:
        
        assert ct0[NODE_TYPE]=='skill'
        second_level_contribution_tuples = obtain_contributions(ct0[ID_IN_GRAPH])
        
        for i, ct1 in enumerate(second_level_contribution_tuples):
          assert ct1[NODE_TYPE]=='skill'
          if ct1[ID_IN_GRAPH] == self._skills.at[ct0[ID_IN_GRAPH], 'enwiki_page_id_of_parent']:
            candidate_explanation_pairs.append((
              ct0[CONTRIBUTED_ACTIVATION],
              ct0[ID_IN_GRAPH],
              self._skills.at[ct0[ID_IN_GRAPH], 'skill_name_long'],
              self._skills.at[ct0[ID_IN_GRAPH], 'depth'],
              ct1[ID_IN_GRAPH],
              self._skills.at[ct1[ID_IN_GRAPH], 'skill_name_long'],
              self._skills.at[ct1[ID_IN_GRAPH], 'depth']
            ))
            del second_level_contribution_tuples[i]
            break
        
        if len(second_level_contribution_tuples)>0:
          ct1 = second_level_contribution_tuples[0]
          candidate_explanation_pairs.append((
            ct0[CONTRIBUTED_ACTIVATION],
            ct1[ID_IN_GRAPH],
            self._skills.at[ct1[ID_IN_GRAPH], 'skill_name_long'],
            self._skills.at[ct1[ID_IN_GRAPH], 'depth'],
            ct0[ID_IN_GRAPH],
            self._skills.at[ct0[ID_IN_GRAPH], 'skill_name_long'],
            self._skills.at[ct0[ID_IN_GRAPH], 'depth']
          ))
      return candidate_explanation_pairs
    
    course_id_in_graph = self._courses.index[course_offset]
    contribution_tuples = obtain_contributions(course_id_in_graph)
    candidate_explanation_pairs = get_candidate_explanation_pairs(contribution_tuples)
    
    ACTIVATION_CONTRIBUTION = 0
    CHILD_ID                = 1
    CHILD_NAME              = 2
    CHILD_DEPTH             = 3
    PARENT_ID               = 4
    PARENT_NAME             = 5
    PARENT_DEPTH            = 6
    
    selected_pair_of_explanation_pairs = None
    
    for depth in [2,1,0]:
      candidate_explanation_pair_pairs = []
      for cep0, cep1 in zip(candidate_explanation_pairs[:-1], candidate_explanation_pairs[1:]):
        if (cep0[PARENT_DEPTH] >= depth and cep1[PARENT_ID] >= depth and 
            cep0[CHILD_ID] != cep1[CHILD_ID] and cep0[CHILD_ID] != cep1[PARENT_ID] and cep0[PARENT_ID] != cep1[CHILD_ID] and cep0[PARENT_ID] != cep1[PARENT_ID]):
          candidate_explanation_pair_pairs.append((cep0, cep1))
      
      # Find the deepest (most specific) pair of candidate explanation pairs (there may be ties which we will collect and break in a moment).
      deepest = -1
      deepest_pairs_of_pairs = []
      for cepp in candidate_explanation_pair_pairs:
        cep0 = cepp[0]
        cep1 = cepp[1]
        depth = cep0[CHILD_DEPTH] + cep1[CHILD_DEPTH]
        if depth == deepest:
          deepest_pairs_of_pairs.append(cepp)
        elif depth > deepest:
          deepest_pairs_of_pairs = [cepp]
      
      # Break the tie, if we have one
      largest_activation = -1
      index_of_largest = -1
      for i, cepp in enumerate(deepest_pairs_of_pairs):
        cep0 = cepp[0]
        cep1 = cepp[1]
        activation = cep0[ACTIVATION_CONTRIBUTION] + cep1[ACTIVATION_CONTRIBUTION]
        if activation > largest_activation:
          largest_activation = activation
          index_of_largest = i
      if index_of_largest != -1:
        selected_pair_of_explanation_pairs = deepest_pairs_of_pairs[index_of_largest]
        break
    
    recommendation="You are recommended to take '%s'" % self._courses.at[course_id_in_graph, COURSE_TITLE_COLUMN]
    skills=[]
    
    # If there was no compatible pair of explanation pairs, see if we have a single pair: not a pair of pairs, just one pair.
    if selected_pair_of_explanation_pairs == None:
      
      # If we have no explanation pairs, then we have to give our explanation based on a single skill.
      # The contribution tuples are sorted in decreasing order of activation contributed, so if we take the first one, it will be the most relevant.
      if len(candidate_explanation_pairs) == 0:
        skill_name = self._skills.at[contribution_tuples[0][2], 'skill_name_long'].replace('_',' ')
        recommendation+= "; in this course you will learn '%s' which is important for the job" % skill_name
        skills.append(skill_name)
      
      # We just take the first candidate explanation pair in the list.
      else:
        cep0 = candidate_explanation_pairs[0]
        recommendation+= "; in this course you will learn '%s' which will help you understand '%s', which is important for the job" % (cep0[CHILD_NAME], cep0[PARENT_NAME])
        skills.append(cep0[CHILD_NAME])
        skills.append(cep0[PARENT_NAME])
    
    # We have a pair of explanation pairs: now we give our explanation
    else:
      cep0 = selected_pair_of_explanation_pairs[0]
      cep1 = selected_pair_of_explanation_pairs[1]
      recommendation += "; in this course you will learn '%s' and '%s' which will help you understand '%s' and '%s'; they are important for the job" % (
        cep0[CHILD_NAME], cep1[CHILD_NAME], cep0[PARENT_NAME], cep1[PARENT_NAME])
      skills.append(cep0[CHILD_NAME])
      skills.append(cep1[CHILD_NAME])
      skills.append(cep0[PARENT_NAME])
      skills.append(cep1[PARENT_NAME])
    
    recommendation+= " '%s'." % job_query_string
    
    #print('\n\n'+recommendation)
    
    # Create an abbreviated explanation in a uniform format to compare with the explanations produced by other methods
    if len(skills) <= 1:
      assert len(skills)==1
      skills.append('N/A')
    abbreviated_recommendation= "In '"+self._courses.at[course_id_in_graph, COURSE_TITLE_COLUMN]+"' you will learn '"+skills[0]+"' and '"+skills[1]+"' which are important for the job '"+job_query_string+"'."
    
    return skills, recommendation, abbreviated_recommendation
