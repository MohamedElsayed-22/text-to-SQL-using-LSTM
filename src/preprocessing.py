import jsonlines
import re
import pandas as pd

# Loading Json Train Data.
def load_data(path):
  phase = []
  table_id = []
  question = []
  sql = []
  with jsonlines.open(path) as f:
      for sample in f.iter():
        phase.append(sample['phase'])
        table_id.append(sample['table_id'])
        question.append(sample['question'])
        sql.append(sample['sql'])
  return pd.DataFrame({'phase': phase, 'id': table_id, 'question': question, 'sql': sql})

# Loading Json Tables info.
def load_table_info(path):    
  id = []
  header = []
  types = []
  rows = []
  with jsonlines.open(path) as f:
    for table in f.iter():
      id.append((table['id']))
      header.append((table['header']))
      types.append((table['types']))
      rows.append(table['rows'])
  f.close()
  return pd.DataFrame({'id': id, 'header': header, 'types': types, 'rows': rows})


# Merging questions and SQL queries with tables info.
def merge_data(train_data, train_table_info):
  merged_data = train_data.merge(train_table_info, on='id')
  return merged_data


# Converting SQL Queries to strings, more suitable for LSTMs.
def convert_to_sql_strings(merged_data, file_name):
  sql_in_text = []
  aggregate = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
  operator = ['=', '<', '>']
  word_operator = ['eq', 'lt', 'gt']

  for querry_sample in range(len(merged_data)):
    sql = merged_data.sql[querry_sample]
    col = merged_data.header[querry_sample][sql['sel']]
    agg = aggregate[sql['agg']]
    cond_col = []
    cond_opr = []
    cond_value = []
    
    for j in sql['conds']:
      cond_col.append(merged_data.header[querry_sample][j[0]])
      cond_opr.append(word_operator[j[1]])
      cond_value.append(j[2])
    resulted_sql = "select "
    if(agg==''):
      resulted_sql += col+" where "
    else:
      resulted_sql += agg+"("+col+") where "

    for j in range(len(cond_col)):
      if(j!=0):
        resulted_sql += "and "
      resulted_sql += cond_col[j]+' '+cond_opr[j]+' '
      resulted_sql += str(cond_value[j])+' '

    sql_in_text.append(resulted_sql)
  merged_data['resulted_sql'] = sql_in_text

  # Saving the preprocessed data into .csv file.
  merged_data.to_csv(file_name)
  return merged_data


# Concatingating headers and questions
def concatenate_headers_and_questions(merged_data):
  print("merged_data: ", len(merged_data))

  question_header = []
  for i in range(len(merged_data)):
    question = merged_data.question[i]
    header = merged_data.header[i]
    
    conc_header_question = question
    
    for elem in header:
      conc_header_question += ' ' + elem
    question_header.append(conc_header_question)
  print("questions_header", len(question_header))
  return question_header

# Decontracting and lowercasing text
def preprocess_text(text):    
  contractions_dict = {
      "won't": "will not",
      "can't": "can not",
      "n't"  : " not",
      "'re"  : " are",
      "'s"   : " is",
      "'d"   : " would",
      "'ll"  : " will",
      "'t"   : " not",
      "'ve"  : " have",
      "'m"   : " am",
      "’re"  : " are",
      "’s"   : " is",
      "’d"   : " would",
      "’ll"  : " will",
      "’t"   : " not",
      "’ve"  : " have",
      "’m"   : " am"
  }

  text = text.lower()

  for contraction, expansion in contractions_dict.items():
      text = re.sub(re.escape(contraction), expansion, text)
  
  return text


# Formating the inputs for the LSTM model
def format_inputs(question_header, merged_data, file_name):
  preprocess_question_header = []
  preprocess_sql_in_text = []

  for i in question_header:
      preprocess_question_header.append(preprocess_text(i))

  for i in merged_data['resulted_sql'].values:
      preprocess_sql_in_text.append(preprocess_text(i))
  
  print("question_header", len(question_header))
  print("preprocess_question_header", len(preprocess_question_header))
  print("preprocess_sql_in_text", len(preprocess_sql_in_text))


  final_data = pd.DataFrame()
  final_data['question_header'] = preprocess_question_header
  final_data['sql'] = preprocess_sql_in_text

  final_data.to_csv(file_name)


def filter_data_by_sequence_length(input_lengths, output_lengths, input_length_lim, output_length_lim, final_data):
  
  filtered = []
  for sample in range(len(input_lengths)):
    if (input_lengths[sample]<=input_length_lim and output_lengths[sample]<=output_length_lim):
      filtered.append(sample)
    
  return final_data.iloc[filtered]
