# Simple script to create an app to query an existing BigQuery database using natural language
# Run locally using `streamlit run YOUR_APP_NAME.py`

# Updates
# - with background reasoning viewable
# - added caching for performance

from io import StringIO
import sys
import re

# Create the in-memory "file"
temp_out = StringIO()
# Replace default stdout (terminal) with stream
sys.stdout = temp_out
# For removal of the ANSI formatting from stdout when outputting
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

import streamlit as st

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import VertexAI

# API and model set up

# Add project ID
project_id = 'ADD_NAME'
region = 'ADD_REGION'
model = "text-bison@001"

# BigQuery dataset details
dataset_id = 'kaggle_survey_2022' # @param {type:"string"}
table_name1 = 'data_subset' # @param {type:"string"}
# table_name2 = 'another_table' # @param {type:"string"}
# table_name3 = 'and_another_table' # @param {type:"string"}

# table_names = (table_name1,table_name2,table_name3)
table_names = [table_name1]

table_uri = f"bigquery://{project_id}/{dataset_id}"

parameters = {
  "temperature": 0,
  "max_output_tokens": 512,
  "top_p": 0.8,
  "top_k": 40,
}

@st.cache_resource(show_spinner="Initialising SQL Engine")
def sql_engine_init():

    engine = create_engine(f"bigquery://{project_id}/{dataset_id}")

    #create SQLDatabase instance from BQ engine
    db = SQLDatabase(engine=engine,metadata=MetaData(bind=engine),include_tables=[x for x in table_names])

    # initiate VertexAI llm
    llm =  VertexAI(
    model_name=model,
    temperature=parameters["temperature"],
    max_output_tokens=parameters["max_output_tokens"],
    top_p=parameters["top_p"],
    verbose=True,
    )

    #create SQL DB Chain with the initialized LLM and above SQLDB instance
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)

    return db_chain

def send_bq_query(query):

    db_chain = sql_engine_init()

    #Define prompt for BigQuery SQL
    _googlesql_prompt = """
    You are a GoogleSQL expert. Given an input question, first create a syntactically correct GoogleSQL query to run, 
    then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results 
    using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. 
    Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Do not query for columns that do not exist. 
    Also, pay attention to which column is in which table.

    The dataset given is for survey data. The table contains demographic data and coding language preferences.

    If someone asks for aggregation on a STRING data type column, then CAST column as NUMERIC before you do the aggregation.

    Only give a maximum of 2 decimal places in floating point numbers.

    If someone asks for column names in the table, use the following format:
    SELECT column_name
    FROM `{project_id}.{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
    WHERE table_name in ('{table_info}')

    Only use the following tables:
    {table_info}

    Question: {input}"""

    GOOGLESQL_PROMPT = PromptTemplate(
      input_variables=["input", "table_info", "top_k", "project_id", "dataset_id"],
      template=_googlesql_prompt,
    )

    #pass question to the prompt template
    final_prompt = GOOGLESQL_PROMPT.format(input=query, project_id =project_id, dataset_id=dataset_id, 
                                         table_info=table_names[0], top_k=10000)

    #pass final prompt to SQL Chain
    output = db_chain(final_prompt)

    # return output['result'], output['intermediate_steps'][1]
    st.write("\n\nAnswer:")
    st.write(output['result'])
    "---"
    st.write("\n\nSQL code used:")
    st.code(output['intermediate_steps'][1], language="sql")

# Page set up 

st.set_page_config(page_title='Working with BigQuery data', page_icon = ':robot_face:')
st.markdown("""
    <style>
        #working-with-bigquery-data {
        text-align: center;
        padding: 50px 50px;
        }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Note that if this title is changed, need to change the style pointer as well above
st.header('Working with BigQuery data')

# Set up side bar
st.sidebar.markdown("""
    <center>
    <img src='https://storage.googleapis.com/gweb-cloudblog-publish/images/Google_Networking_02.max-2576x1042.jpg'
     width=340>
    </center>
    """, unsafe_allow_html=True)
st.sidebar.markdown("""
    ---

    This demo shows how it is possible to perform simple queries of 
    a raw SQL dataset using natural language. 

    The dataset used is a subset of the 
    [2022 Kaggle ML & Data Science Survey](https://www.kaggle.com/competitions/kaggle-survey-2022/data) 
    which has been imported into BigQuery. 

    ---

    **Sample questions**:
    - Where do the majority of the respondents come from?
    - What are the top ten countries with the most responses? Show in a table.
    - What's the gender breakdown of python users? (return answer as a table)
    - What are the levels of education obtained?
    - How many users use both python and SQL?

    ---
    """)
st.sidebar.markdown('Model used: \t`{}`'.format(model))
st.sidebar.write('Parameters used: ',parameters)

# Create query area 
query = st.text_area("Input", label_visibility='hidden',
    key="query", placeholder="Ask your question here")

# If query is entered, call API for prediction
if query:

    with st.spinner('Querying dataset...'):
        send_bq_query(query)

        with st.expander("Background: prompt and reasoning"):
            st.code(ansi_escape.sub('', temp_out.getvalue()))

