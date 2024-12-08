from operator import index
import streamlit as st
import plotly.express as px
#import yaml
from yaml.loader import SafeLoader

import pandas as pd
#from streamlit_pandas_profiling import st_profile_report
import os
import numpy as np

#from composabl_ddm.composabl_ddm_create import reward_sim_op_ML
#from composabl_ddm.composabl_ddm_create import sim_op_ML

import pandas as pd
import datetime as dt
import random
import json
import seaborn as sns

import matplotlib.pyplot as plt
import pickle
import copy
from statsmodels.iolib.smpickle import load_pickle
from collections import defaultdict

import openai
#from dotenv import load_dotenv
import shutil
import re
import zipfile
import ast
import textwrap


data_quality = False

with st.sidebar: 
    ##st.image("https://cdn.prod.website-files.com/65973bba7be64ecd9a0c2ee8/6597409a88b276d54f037d29_logo-primary%402x.jpg")
    st.title("Suporte Saude")
    choice = st.radio("Menu", ["Relatorio Medico", "Teste", ])
    st.info("Apllicacao para suporte medico.")

if choice == "Teste": 
    st.title("Upload Your Dataset")
    '''file = st.file_uploader("Upload Your Dataset")
    
    if file: 
        data = pd.read_csv(file, index_col=None)

        project = st.text_input('What is the project Name (without white spaces):')
        index_col = st.selectbox('Select your index column: ', ['Index'] + list(data.columns))

        if index_col!= 'Index' :
            data = data.set_index(pd.Index(data[index_col]))
        var_list = st.multiselect('Choose your State Variables', data.columns)
        u_list = st.multiselect('Choose your Action Variables', [s for s in data.columns if (s not in var_list)])

        if len(var_list) != 0 and len(u_list) != 0 :
            for path in [f'config/{project}', f'data/{project}', f'models/{project}', f'agent/{project}']:
                if not os.path.exists(path):
                    os.makedirs(path)

            if st.button("Create DDM Sim"):
                #remove duplicates
                ###variables_list_full = [var_list + u_list + y_list]
                variables_list_full = [var_list + u_list]
                variables_list_full  = list(set(variables_list_full[0]))

                if project == 'fhr':
                    data.columns = [ x + '.vec' for x in data.columns]
                    variables_list_full = [ x + '.vec' for x in variables_list_full]
                    var_list = [ x + '.vec' for x in var_list]
                    u_list = [ x + '.vec' for x in u_list]

                data = data[variables_list_full]

                #Data Cleaning
                #check if all variables are in the dataframe
                if set(variables_list_full).issubset(data.columns):
                    st.text("All columns are in the dataframe")
                else:
                    st.text("Not all columns are in the dataframe")

                # Check for String values
                def isstring(x):
                    return type(x) == str
                
                #pre process
                num_df = data.copy()

                #tab1, tab2, tab3, tab4 = st.tabs(["Constraints and Initial State Config", "Constraint Check","DDM Creation"])
                #with tab1:
                #Constraints
                #define constraints - State
                obs_space_constraints = {}
                action_space_constraints = {}
                action_constraints = {}
                reward_constraints = {}

                st.title('Edit Constraints in the table below:')
                st.write('State Constraints:')
                state_cons_df = st.data_editor(num_df[var_list].describe().T[['min','max']])

                #print(edited_df)
                for c in var_list:
                    obs_space_constraints[c] = {}
                    obs_space_constraints[c]['low'] = state_cons_df.loc[c,'min']
                    obs_space_constraints[c]['high'] = state_cons_df.loc[c,'max']

                st.write('Action Constraints:')
                action_space_cons_df = st.data_editor(num_df[u_list].describe().T[['min','max']])

                for c in u_list:
                    action_space_constraints[c] = {}
                    action_space_constraints[c]['low'] = action_space_cons_df.loc[c,'min']
                    action_space_constraints[c]['high'] = action_space_cons_df.loc[c,'max']

                st.write('Delta Action Constraints:')
                action_cons_df = st.data_editor(num_df[u_list].describe().T[['min','max']] /2)

                for c in u_list:
                    action_constraints[c] = {}
                    action_constraints[c]['low'] = action_cons_df.loc[c,'min']
                    action_constraints[c]['high'] = action_cons_df.loc[c,'max']


                tag = st.selectbox('Choose a variable to analyze:', num_df.columns)
                slider_plot1 = st.slider('Select an Index to start: ', int(num_df.index.min()), int(num_df.index.max()))

                df_plot = num_df.reset_index(drop=True).loc[slider_plot1: , tag] #.index 
                fig, ax = plt.subplots()
                plt.figure(figsize=(20, 5))
                ax.plot(list(df_plot.index), list(df_plot.values), c='blue', alpha=0.9)
                st.pyplot(fig)

                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super(NpEncoder, self).default(obj)

                
                with open(f'config/{project}/obs_space_constraints.json', 'w') as f:
                    json.dump(obs_space_constraints, f, cls=NpEncoder)
                f.close()

                with open(f'config/{project}/action_space_constraints.json', 'w') as f:
                    json.dump(action_space_constraints, f, cls=NpEncoder)
                f.close()

                with open(f'config/{project}/action_constraints.json', 'w') as f:
                    json.dump(action_constraints, f, cls=NpEncoder)
                f.close()

                st.write("Saved constraint files!")

                # Save initial state values
                initial_state = {}

                for v in var_list:
                    initial_state[v] = num_df.reset_index(drop=True).loc[slider_plot1 , v]

                with open(f'config/{project}/initial_state.json', 'w') as f:
                    json.dump(initial_state, f, cls=NpEncoder)
                f.close()
                st.write(initial_state)

                # initial action
                initial_action = {}
                for v in u_list:
                    initial_action[v] = num_df.reset_index(drop=True).loc[slider_plot1 , v]
                st.write(initial_action)

                with open(f'config/{project}/initial_action.json', 'w') as f:
                    json.dump(initial_action, f, cls=NpEncoder)
                f.close()

                train_metadata = {}
                train_metadata['index'] = {}
                train_metadata['index']['low'] = slider_plot1
                train_metadata['index']['high'] = num_df.reset_index(drop=True).index.max() ###

                st.write(train_metadata)

                with open(f'config/{project}/train_metadata.json', 'w') as f:
                    json.dump(train_metadata, f, cls=NpEncoder)
                f.close()

                st.write("Saved Initial Value files!")

                # Identify clean sequences
                clean_df = num_df.copy()
                clean_df  = clean_df.dropna()

                clean_sequences = pd.DataFrame({'index_df': clean_df.index, 'index_a': clean_df.index})
                clean_sequences = clean_sequences.set_index('index_a')
                clean_sequences['diff'] = clean_sequences.diff()
                clean_sequences['raw_index_start'] = clean_df.reset_index(drop=True).index
                clean_sequences = clean_sequences[(clean_sequences['diff'].fillna(1000) > 1) | (clean_sequences['diff'].fillna(1000) < 0)] #60
                clean_sequences['duration'] = (clean_sequences['index_df'].diff() - clean_sequences['diff']).shift(periods=-1)
                #clean_sequences['cycle_start'] =  clean_sequences['index_df']
                clean_sequences['cycle_start'] =  clean_sequences['raw_index_start']
                clean_sequences['cycle_end'] =  clean_sequences['cycle_start'] + clean_sequences['duration'].abs()

                clean_sequences['cycle_end'] = clean_sequences['cycle_end'].fillna(clean_df.index.max())

                clean_sequences['duration'] = clean_sequences['cycle_end'] - clean_sequences['cycle_start']

                st.write("Clean Sequences: ", len(clean_sequences))

                # Show clean sequences and length
                st.dataframe(clean_sequences.sort_values(by=['duration'], ascending=False))

                # creating a dataset with all scenarios
                df_scenarios = pd.DataFrame()
                df_cycle_list = []

                for i,seq in clean_sequences[clean_sequences['duration'] >= 5].iterrows(): #TODO: change to parameter (minimum steps to consider valid)
                    c_start = int(seq['cycle_start'])
                    c_end = int(seq['cycle_end'])

                    df_cycle = num_df.reset_index(drop=True).loc[c_start:c_end,:]

                    ###assert (df_cycle.isna().sum()).sum() == 0

                    #df_scenarios = df_scenarios.append(df_cycle) # deprecated since 1.4.0 pandas
                    df_scenarios = pd.concat([df_scenarios, df_cycle])
                    df_cycle_list.append(df_cycle)

                st.write("Scenarios Lenght: ", len(df_scenarios))
                st.write("Average Cycle Lenght: ", np.mean([ len(x) for x in df_cycle_list]))
                st.write("Cycle list Lenght: ", len(df_cycle_list))

                cycle_df = df_scenarios

                # Save cycle data
                with open(f'data/{project}/train_cycles.pkl', 'wb') as f:
                    pickle.dump(df_cycle_list, f)
                f.close()

                st.write("Cycles DB saved!")

                st.write(len(df_cycle_list) , "total scenarios to work") 

                def check_constraints():
                    if os.path.exists(f'config/{project}/obs_space_constraints.json'):
                        with open(f'config/{project}/obs_space_constraints.json') as f:
                            obs_space_constraints = json.load(f)
                        f.close()

                        with open(f'config/{project}/action_space_constraints.json') as f:
                            action_space_constraints = json.load(f)
                        f.close()

                        st.write('Read files...')
                        # load full and actual constraints
                        obs_space_constraints.update(action_space_constraints)
                        #obs_space_constraints.update(reward_constraints)

                        # remove values outside constraints 
                        def check_constraints(row, col_names):
                            new_row = []
                            for col in col_names:
                                if obs_space_constraints[col]['low'] <= row[col] <= obs_space_constraints[col]['high']:
                                    new_row.append(row[col])
                                else:
                                    new_row.append(np.nan) 
                                    

                            return pd.Series(new_row)

                        new_df = num_df.apply(lambda row: check_constraints(row, num_df.columns), axis=1)
                        new_df.columns = num_df.columns

                        st.write('Finish constraint check')

                        #Generate a list in percentage of values outside constraints
                        st.dataframe((new_df.isna().sum() / len(new_df) *100).sort_values(ascending=False))

                        #clean sequences to create batches - cycles
                        clean_df = new_df.copy()
                        clean_df  = clean_df.dropna()

                for path in [f'models/{project}/xgboost']:
                    if not os.path.exists(path):
                        os.makedirs(path)
                
                # create ML models
                st.write("columns: " + str([var_list + u_list]))
                #test_df = test_df.loc[:,~test_df.columns.duplicated()]
                model_train_metadata_model = sim_op_ML([cycle_df[var_list + u_list] for cycle_df in df_cycle_list], u_list, project, model_type='xgboost')

                with open(f'config/{project}/model_train_metadata.json', 'w') as f:
                    json.dump(model_train_metadata_model, f)
                f.close()

                st.write("Models created !")

                # Prepare Zip File for download
                def zipdir(path, ziph, project):
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            if project in os.path.join(root, file) and 'agent' not in os.path.join(root, file):
                                if '.zip' not in file:
                                    ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))

                # Directory to zip
                dir_to_zip = "../composabl-product/"
                zip_filename = project + '_ddm_sim' + ".zip"

                # Create the zip file
                with zipfile.ZipFile(zip_filename, 'w') as zipf:
                    zipdir(dir_to_zip, zipf, project)

                # Save Zip File
                with open(project + '_ddm_sim' + ".zip", "rb") as fp:
                    btn = st.download_button(
                        label="Download DDM Sim",
                        data=fp,
                        file_name= project + '_zip' + ".zip",
                        mime="application/zip"
                    )'''

        
        
if choice == "Relatorio Medico":
    openai_api_key = st.text_input('OpenAI Key:')
    
    openai.api_key = openai_api_key
    # Helper functions
    def json_gpt(input: str):
        GPT_MODEL = "gpt-3.5-turbo"
        completion = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Output only valid JSON"},
                {"role": "user", "content": input},
            ],
            temperature=0.1,
        )

        text = completion.choices[0].message.content
        parsed = json.loads(text)

        return parsed


    def embeddings(input: list[str]) -> list[list[str]]:
        response = openai.Embedding.create(model="text-embedding-ada-002", input=input)
        return [data.embedding for data in response.data]

    # get group of skills
    def search():
        # User asks a question
        USER_QUESTION = f"I want to build autonomous system from a business case called project. Which skills should I create to achive success on my system ? \
            Here is a brief explanation about my case: "

        QUERIES_INPUT = f"""
        You have access to API that returns state and actions.
        Skill is an autonomous agent that receive observation variables and execute actions into the system. 
        Skills has their own reward function.
        Examaple of skills are ["NavigationSkill", "JumpSkill", "BuySkill"]
        Generate an array of autonomous system skills that are relevant to this question.
        For example, include skills like ['skill_1', 'skill_2', 'skill_3'].
        Be creative. The more queries you include, the more likely you are to find relevant results.

        User question: {USER_QUESTION}

        Format: {{"skills": ["skill_1", "skill_2", "skill_3"]}}
        """

        queries = json_gpt(QUERIES_INPUT)["skills"]

        # Let's include the original question as well for good measure
        queries.append(USER_QUESTION)

        ###print('Query: ', queries)
    
    #search(project='starship')
        
    def ask_function_calling(query):
        messages = []
        messages.append({"role": "system", "content": "Gere um relatorio como um Médico Profissional sobre as possiveis doenças, tratamentos e famarcologia para o paciente descrito pelo usuário."})
        messages.append({"role": "user", "content": query})
        #messages = [{"role": "user", "content": query}]

        response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo-0613",
            model="gpt-4o-mini",
            messages=messages,
            #functions = function_descriptions,
            #function_call="auto",
            temperature=0.2
        )

        #print('Response: ', response)

        '''while response["choices"][0]["finish_reason"] == "function_call":
            function_response = function_call(response)
            messages.append({
                "role": "function",
                "name": response["choices"][0]["message"]["function_call"]["name"],
                "content": json.dumps(function_response)
            })

            print("messages: ", messages) 

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                functions = function_descriptions,
                function_call="auto",
                temperature=0.05
            )   

            #print("response: ", response) 
        else:'''
        
        return response["choices"][0]["message"]["content"]

    st.title("Suporte Médico com AI")
    st.write("Este é um modelo de AI para suporte ao médico.")
    nome = st.text_input('Nome do Paciente:')
    idade = st.slider('Idade do Paciente: ', int(1), int(120))
    sexo = st.selectbox('Qual o sexo do paciente: ', ['feminino','masculino'])
    explain = st.text_area('Explique o caso do paciente:')

    prompt = f"""O paciente de nome {nome}, idade {idade} e sexo {sexo} fez uma visita ao consultório médico e busca obter um relatório sobre um médico profissional. \
                Segue uma breve explicação do prontuário do paciente: """
    prompt += explain

    if st.button("Gerar Relatorio", type='primary'):
        answer = ask_function_calling(prompt)
        st.write(answer)

        # Prepare Zip File for download
        '''def zipdir(path, ziph, project):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if project in os.path.join(root, file):
                        if '.zip' not in file:
                            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))

        # Directory to zip
        dir_to_zip = "../composabl-product/agent/"
        zip_filename = project_name + '_llm_agent' + ".zip"

        # Create the zip file
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            zipdir(dir_to_zip, zipf, project_name)

        # Save Zip File
        with open(project_name + '_llm_agent' + ".zip", "rb") as fp:
            btn = st.download_button(
                label="Download your Agent",
                data=fp,
                file_name= project_name + '_zip' + ".zip",
                mime="application/zip"
            )
            
        fp.close()''' 
