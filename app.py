from operator import index
import streamlit as st
import plotly.express as px
import yaml
from yaml.loader import SafeLoader

import pandas as pd
#from streamlit_pandas_profiling import st_profile_report
import os
import numpy as np

from composabl_ddm.composabl_ddm_create import reward_sim_op_ML
from composabl_ddm.composabl_ddm_create import sim_op_ML

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
    st.image("https://cdn.prod.website-files.com/65973bba7be64ecd9a0c2ee8/6597409a88b276d54f037d29_logo-primary%402x.jpg")
    st.title("Composabl DDM Sim Platform")
    choice = st.radio("Menu", ["Create DDM Sim", "Evaluate DDM Sim", "Composabl Copilot"])
    st.info("This application helps you Build and Run DataDrivenModel Simulations from your data.")

if choice == "Create DDM Sim": 
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    
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
                    )

        
        
if choice == "Evaluate DDM Sim": 
    #TODO: Change to multiselect - read all folder names and check authorization by license ? user ?
    project = st.text_input('What is the project Name (without white spaces):')
    if os.path.exists(f'data/{project}/train_cycles.pkl'):
        simulations = st.number_input('How many parallel simulations you want to run?', 1, 100,key='simulations')
        model_update = False
        mode = st.selectbox('What is the Actions Mode that you want to run?', ['immi', 'external','random'])
        cycle = st.number_input('What is the cycle that you want to run', 1, 100, key='cycle') # TODO: change to max cycle 
        sim_time = st.number_input('What is the look ahead that you want to run (sim time)?', 1, 100, key='sim_time') # TODO: change to max cycle 
        noise = st.number_input('How much noise (%) you want to apply to the system?', 0, 100, key='noise')

        class Env():
            def __init__(self, env_config = {'noise':0}):

                # load constraints
                with open(f'config/{project}/obs_space_constraints.json') as f:
                    self.obs_space_constraints = json.load(f)
                f.close()

                with open(f'config/{project}/action_constraints.json') as f:
                    self.action_constraints = json.load(f)
                f.close()

                with open(f'config/{project}/action_space_constraints.json') as f:
                    self.action_space_constraints = json.load(f)
                f.close()
                
                low_list = [x['low'] for x in self.obs_space_constraints.values()]
                high_list = [x['high'] for x in self.obs_space_constraints.values()]
                
                #self.observation_space = gym.spaces.Box(low=np.array(low_list), high=np.array(high_list))
                self.observation_space = []

                low_act_list = [x['low'] for x in self.action_constraints.values()]
                high_act_list = [x['high'] for x in self.action_constraints.values()]

                #self.action_space = gym.spaces.Box(low=np.array(low_act_list), high=np.array(high_act_list))
                self.action_space = []

                self.reward_full = []
                self.pred_full = []
                self.action_full= []
                
                self.state_models = {}
                #self.reward_models = {}
                
                self.noise = env_config['noise']

                
            def reset(self):
                self.cnt = 0

                # load initial values
                with open(f'config/{project}/initial_action.json') as f:
                    self.old_action = json.load(f)
                f.close()
                
                #with open(f'config/{project}/initial_reward.json') as f:
                #    self.old_reward = json.load(f)
                #f.close()
                
                with open(f'config/{project}/initial_state.json') as f:
                    self.obs = json.load(f)
                f.close()

                self.intial_state = copy.deepcopy(self.obs)
                #self.initial_reward = copy.deepcopy(self.old_reward)
                self.intial_action = copy.deepcopy(self.old_action)
                
                #self.reward_full.append(0)
                #self.pred_full.append(self.old_reward.values())
                self.action_full.append(self.old_action.values())
                
                #for key_r in self.old_reward.keys():
                #    self.reward_models[key_r] = load_pickle("./models/" + key_r + "_reward_model.pkl")
                #with open(f'config/{project}/model_train_reward_metadata.json') as f:
                #        model_train_reward_metadata = json.load(f)
                #f.close()

                with open(f'config/{project}/model_train_metadata.json') as f:
                        model_train_metadata = json.load(f)
                f.close()
                
                model_type = 'xgboost' #TODO: load from train metadata

                if model_type == 'xgboost' :
                    #for key_r in self.old_reward.keys():
                    #    self.reward_models[key_r] = load_pickle(f"./models/{project}/xgboost/{key_r}_reward_model.pkl")

                    for key_s in self.obs.keys():
                        self.state_models[key_s] = load_pickle(f"./models/{project}/xgboost/{key_s}_model.pkl")

                    # load X metadata for obs and reward
                    #self.state_pred_metadata = list(self.state_models[key_s].params.index)
                    self.state_pred_metadata = list(model_train_metadata[key_s]['x_columns'])
                    #self.reward_pred_metadata = list(model_train_reward_metadata[key_r]['x_columns'])

                else:
                    #for key_r in self.old_reward.keys():
                    #    self.reward_models[key_r] = load_pickle(f"./models/{project}/{key_r}_reward_model.pkl")
                    
                    for key_s in self.obs.keys():
                        self.state_models[key_s] = load_pickle(f"./models/{project}/{key_s}_model.pkl")

                    # load X metadata for obs and reward
                    self.state_pred_metadata = list(self.state_models[key_s].params.index)
                    #self.reward_pred_metadata = list(self.reward_models[key_r].params.index)
                

                self.obs = np.array(list(self.obs.values()))
                info = {}
                return self.obs    

            def step(self, action):
                done = False
                discount = 0
                self.df = pd.DataFrame()

                # Update action with delta actions
                def dsum(*dicts):
                    ret = defaultdict(int)
                    for d in dicts:
                        for k, v in d.items():
                            ret[k] += v
                    return dict(ret)
            
                self.delta_action  = copy.deepcopy(self.action_constraints)

                for i,key in enumerate(self.delta_action.keys()):
                    self.delta_action[key] = action[i]

                action = dsum(self.old_action, self.delta_action )

                #clip action values
                for i,key in enumerate(action.keys()):
                    action[key] = np.clip(float(action[key]), float(self.action_space_constraints[key]['low']), float(self.action_space_constraints[key]['high']))
                
                # State Variable prediction     
                # prepare data to predict
                state_t = pd.DataFrame(data=[self.obs], columns=[str(x) + '0' for x in list(self.intial_state.keys()) ]  ) #list
                action_t0 = pd.DataFrame.from_dict(self.old_action, orient='index').T #dict
                action_t0.columns = [ str(x) + '0' for x in action_t0.columns ]
                action_t = pd.DataFrame(data=[[ float(x) for x in list(action.values())]] , columns=list(action.keys())) #list
                X = pd.concat([state_t, action_t0, action_t], axis=1)
                #reorder dataframe with the real order from metadata
                X = X[self.state_pred_metadata]

                pred_list = []
                for var in self.intial_state.keys():
                    res = self.state_models[var]
                    pred = res.predict(X)
                    
                    #prevent extrapolation
                    if not (self.obs_space_constraints[var]['low'] <= float(pred[0]) <= self.obs_space_constraints[var]['high']):
                        pred = np.clip(float(pred[0]),self.obs_space_constraints[var]['low'], self.obs_space_constraints[var]['high']) 
                        discount += 1
                        done = True
                            
                    pred_list.append(pred)

                #self.df = self.df.append( pd.DataFrame(columns=list(self.intial_state.keys()), data=[pred_list]))
                self.df = pd.concat( [self.df, pd.DataFrame(columns=list(self.intial_state.keys()), data=[pred_list]) ] )
                
                #Increase time
                self.cnt += 1
                
                # set new old action before generating prediction for rewards
                if action != None:
                    self.old_action = copy.deepcopy(action)
                    
                
                # update obs with new state values
                self.obs = {}
                for key in list(self.intial_state.keys()):
                    # add noise 
                    val = float(self.df[key].iloc[0])
                    noise = self.noise #0.02 
                    self.obs[key] = val +  random.uniform(- val * noise, val * noise)
                
                # REWARD ENGINEERING
                #   
                reward = 0
                        
                self.reward_full.append(reward)
                #self.pred_full.append(self.old_reward.values())
                self.action_full.append(self.old_action.values())

                # close the simulation
                if self.cnt == 30: #15 min dt - 288min = 3 days
                    done = True
                    
                self.obs = np.array(list(self.obs.values()))
                info = {}     
                return self.obs, reward, done, info
            
            def render(self, mode='auto'):
                df_pred = pd.DataFrame(data=self.pred_full)
                df_pred.plot(figsize=(12,30), subplots=True)  
        
        
        
        #RUN SIMULATION
        #read files
        with open(f'config/{project}/train_metadata.json') as f:
            train_metadata = json.load(f)
        f.close()

        with open(f'config/{project}/obs_space_constraints.json') as f:
            obs_space_constraints = json.load(f)
        f.close()

        #with open(f'config/{project}/reward_constraints.json') as f:
        #    reward_constraints = json.load(f)
        #f.close()

        with open(f'config/{project}/initial_state.json') as f:
            initial_state = json.load(f)
        f.close()

        #with open(f'config/{project}/initial_reward.json') as f:
        #    pred_dict = json.load(f)
        #f.close()

        with open(f'config/{project}/initial_action.json') as f:
            initial_action = json.load(f)
        f.close()

        with open(f'data/{project}/train_cycles.pkl', 'rb') as f:
            df_cycle_list = pickle.load(f)
        f.close()   
        st.write('All files loaded.')

        #simulations = 3
        #model_update = False
        #mode = 'immi'#'external'
        #sim_time = 20

        obs_dfs = pd.DataFrame()
        act_dfs = pd.DataFrame()
        y_dfs = pd.DataFrame()
        #cycle = 10
        act_df = df_cycle_list[cycle-1]
        start = act_df.index.start
        end = act_df.index.stop
        st.write(f"Cycle {cycle} , start: {start}, end: {end}, Simulation length: {sim_time}, Cycle length: {len(act_df)}")
        st.write("Simulation start")
        for j in range(simulations):
            
            # get start and end from scenarios
            #start = int(train_metadata['index']['low'])
            #end = int(train_metadata['index']['high'])
            ##cycles_list_start = [38271,53336,31170,78782]
            ##cycles_list_end = [39770,56532,33462,80799]
            ##start = cycles_list_start[1]	
            ##end = cycles_list_end[1]
            
            # start env
            bench_env = Env(env_config={'noise': noise/100})
            obs = bench_env.reset()

            if mode == 'immi':
                # update sim obs with real obs for actual step
                obs = {}
                for key in list(initial_state.keys()):
                    val = float(act_df[key].loc[start])
                    obs[key] = val

                #old_reward = {}
                # reward update
                #for key in list(pred_dict.keys()):  
                #    old_reward[key] = float(act_df[key].loc[start])    
                
                old_action = {}
                # action update
                for key in list(initial_action.keys()):  
                    old_action[key] = float(act_df[key].loc[start])       
            
                obs = np.array(list(obs.values()))
                bench_env.obs = obs
                #bench_env.old_reward = old_reward
                bench_env.old_action = old_action

            # initialize history dataframes
            state_df = pd.DataFrame(columns=initial_state.keys() , data=[obs])
            actions_df = pd.DataFrame(columns=bench_env.old_action.keys() , data=[bench_env.old_action])
            #y_list_df = pd.DataFrame(columns=bench_env.old_reward.keys() , data=[bench_env.old_reward])
            reward_list = []

            for n in range(start, start + sim_time):
                
                # random actions #TODO : change to parameter
                if  mode == 'random':
                    action = copy.deepcopy(initial_action)
                    for k in action.keys():
                        action[k] = random.uniform(-0.01,  0.01)

                elif mode == 'immi':
                    action = copy.deepcopy(initial_action)
                    for k in action.keys():
                        action[k] = act_df[k][n+1] - act_df[k][n]
                                            
                #external controller - TODO: import external json with actions
                elif mode == 'external':
                    if n == start:
                        external_cnt = 0
                        action = copy.deepcopy(initial_action)
                        for k in action.keys():
                            action[k] = external_dict[k][external_cnt]

                    else:
                        external_cnt += 1
                        for k in action.keys():
                            action[k] = external_dict[k][external_cnt]
                        
                        
                # run simulation step
                obs, reward, done, info = bench_env.step(list(action.values())) #dt = 15min

                if mode == 'immi' and model_update == True:
                    if n % 5 == 0: #every 5 steps
                        # update sim obs with real obs for next step
                        obs = {}
                        for key in list(obs_space_constraints.keys()):
                            val = float(act_df[key].loc[n+1])
                            obs[key] = val
                        
                        #old_reward = {}
                        # reward dict update
                        #for key in list(pred_dict.keys()):  
                        #    old_reward[key] = float(act_df[key].loc[n+1])               
                        
                        obs = np.array(list(obs.values()))
                        bench_env.obs = obs

                        #bench_env.old_reward = old_reward
                
                #TODO: break the simulation if needed (extrapolation or max iterations)
                #if done:
                #    break
                
                # increment history dataframes
                #state_df = state_df.append( pd.DataFrame(columns=initial_state.keys() , data=[obs]))
                state_df = pd.concat([state_df, pd.DataFrame(columns=initial_state.keys() , data=[obs])])
                #actions_df = actions_df.append(pd.DataFrame(columns=bench_env.old_action.keys() , data=[bench_env.old_action]))
                actions_df = pd.concat([actions_df, pd.DataFrame(columns=bench_env.old_action.keys() , data=[bench_env.old_action])])
                #y_list_df = y_list_df.append(pd.DataFrame(data=[bench_env.pred_full[-1]], columns=bench_env.old_reward.keys() ))
                    
                reward_list.append(reward)

            state_df = state_df.reset_index(drop=True)  
            actions_df = actions_df.reset_index(drop=True)
            #y_list_df = y_list_df.reset_index(drop=True)

            obs_dfs = pd.concat([obs_dfs, state_df])
            act_dfs = pd.concat([act_dfs, actions_df])
            #y_dfs = pd.concat([y_dfs, y_list_df])  

        #aggregate
        state_df = obs_dfs.groupby(obs_dfs.index).mean().iloc[:,:]
        actions_df = act_dfs.groupby(act_dfs.index).mean().iloc[:,:]
        #y_list_df = y_dfs.groupby(y_dfs.index).mean().iloc[:,:]

        #validation 
        tab5, tab6, tab7, tab8 = st.tabs(["Reward Validation", "State Validation", "Action Validation", "Trajectories"])
        with tab5:
            # reward
            plot_df = state_df
            vals = []
            cols = []
            
            #fig, ax = plt.subplots()
            #plt.figure(figsize=(20, 5))
            #ax.plot(list(df_plot.index), list(df_plot.values), c='blue', alpha=0.9)
            #ax.title(tag)
            
            for tag in [ x for x in plot_df.columns if ('index' not in x)]:
                fig = plt.figure()
                plt.plot(act_df.loc[start:start+sim_time,:].reset_index(drop=True)[tag])
                plt.plot(plot_df.loc[:sim_time,tag])
                plt.legend(['Train data', 'DDM Sim data'])
                #plt.ylim(reward_constraints[tag]['low'],reward_constraints[tag]['high'])
                plt.title(tag)

                mean_pct = np.mean( ((plot_df.loc[:sim_time,tag] - act_df.loc[start:start+sim_time,:].reset_index()[tag]) / act_df.loc[start:start+sim_time,:].reset_index()[tag]) )*100
                st.write('Mean pct error: ', mean_pct, ' %')
                #print('Max pct error: ', np.max( ((plot_df.loc[:sim_time,tag] - act_df.loc[start:start+sim_time,:].reset_index()[tag]) / act_df.loc[start:start+sim_time,:].reset_index()[tag]) )*100, ' %')
                vals.append(mean_pct)
                cols.append(tag)
                st.pyplot(fig)

            reward_error_df = pd.DataFrame(data=[vals], columns=cols)
            st.subheader("Average Error on Reward: " + str(np.mean(abs(reward_error_df.values[0][:]))) + "%" + ' +- ' + str(np.std(abs(reward_error_df.values[0][:]))))
            st.dataframe(reward_error_df.T)
        
        with tab6:
            # reward
            plot_df = state_df
            vals = []
            cols = []
            
            for tag in [ x for x in plot_df.columns if ('index' not in x)]:
                fig = plt.figure()
                plt.plot(act_df.loc[start:start+sim_time,:].reset_index(drop=True)[tag])
                plt.plot(plot_df.loc[:sim_time,tag])
                plt.legend(['Train data', 'DDM Sim data'])
                #plt.ylim(reward_constraints[tag]['low'],reward_constraints[tag]['high'])
                plt.title(tag)

                mean_pct = np.mean( ((plot_df.loc[:sim_time,tag] - act_df.loc[start:start+sim_time,:].reset_index()[tag]) / act_df.loc[start:start+sim_time,:].reset_index()[tag]) )*100
                st.write('Mean pct error: ', mean_pct, ' %')
                #print('Max pct error: ', np.max( ((plot_df.loc[:sim_time,tag] - act_df.loc[start:start+sim_time,:].reset_index()[tag]) / act_df.loc[start:start+sim_time,:].reset_index()[tag]) )*100, ' %')
                vals.append(mean_pct)
                cols.append(tag)
                st.pyplot(fig)

            state_error_df = pd.DataFrame(data=[vals], columns=cols)
            st.subheader("Average Error on State: " + str(np.mean(abs(state_error_df.values[0][:]))) + "%" + ' +- ' + str(np.std(abs(state_error_df.values[0][:]))))
            st.dataframe(state_error_df.T)

        with tab7:
            # reward
            plot_df = actions_df
            vals = []
            cols = []
            
            for tag in [ x for x in plot_df.columns if ('index' not in x)]:
                fig = plt.figure()
                plt.plot(act_df.loc[start:start+sim_time,:].reset_index(drop=True)[tag])
                plt.plot(plot_df.loc[:sim_time,tag])
                plt.legend(['Train data', 'DDM Sim data'])
                #plt.ylim(reward_constraints[tag]['low'],reward_constraints[tag]['high'])
                plt.title(tag)

                mean_pct = np.mean( ((plot_df.loc[:sim_time,tag] - act_df.loc[start:start+sim_time,:].reset_index()[tag]) / act_df.loc[start:start+sim_time,:].reset_index()[tag]) )*100
                st.write('Mean pct error: ', mean_pct, ' %')
                #print('Max pct error: ', np.max( ((plot_df.loc[:sim_time,tag] - act_df.loc[start:start+sim_time,:].reset_index()[tag]) / act_df.loc[start:start+sim_time,:].reset_index()[tag]) )*100, ' %')
                vals.append(mean_pct)
                cols.append(tag)
                st.pyplot(fig)

            action_error_df = pd.DataFrame(data=[vals], columns=cols)
            st.subheader("Average Error on State: " + str(np.mean(abs(action_error_df.values[0][:]))) + "%" + ' +- ' + str(np.std(abs(action_error_df.values[0][:]))))
            st.dataframe(action_error_df.T)

        with tab8:
            tag2 = st.selectbox('Choose a State variable to analyze its trajectory:', state_df.columns)
            fig = plt.figure(figsize=(20,5))
            plt.plot(obs_dfs.index, obs_dfs[tag2])
            plt.title(tag2)
            st.pyplot(fig)

            #tag3 = st.selectbox('Choose a Reward variable to analyze its trajectory:', y_list_df.columns)
            #fig = plt.figure(figsize=(20,5))
            #plt.plot(y_dfs.index, y_dfs[tag3])
            #plt.title(tag3)
            #st.pyplot(fig)

            tag4 = st.selectbox('Choose a Action variable to analyze its trajectory:', act_dfs.columns)
            fig = plt.figure(figsize=(20,5))
            plt.plot(act_dfs.index, act_dfs[tag4])
            plt.title(tag4)
            st.pyplot(fig)


if choice == "Composabl Copilot":
    openai_api_key = st.text_input('Your OpenAI Key for ChatGPT:')
    project_name = st.selectbox('What is your project name', ['starship','cstr'])
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
    def search(project='starship'):
        # User asks a question
        USER_QUESTION = f"I want to build autonomous system from a business case called {project}. Which skills should I create to achive success on my system ? \
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

    def get_env_state(project_name='starship'):
        obs_state = {
            'starship': {
                'x': {'low': -400, "high": 400},
                'x_speed': {'low': -100, "high": 100},
                'y': {'low': 0, "high": 1000},
                'y_speed': {'low': -1000, "high": 100},
                'angle': {'low': -3.15 *2, "high": 3.15 *2},
                'angle_speed': {"low": 3, "high": 3}},
            'cstr': {
                'T':{'low': 200, "high": 500},
                'Tc':{'low': 200, "high": 500},
                'Ca':{'low': 0, "high": 12},
                'Cref':{'low': 0, "high": 12},
                'Tref':{'low': 200, "high": 500}
            }
        }
        return obs_state[project_name]

    def get_env_actions(project_name='starship'):
        actions = {
            'starship':{
                'thrust': {'low': 0.4, "high": 1},
                'angle': {'low': -3.14, "high": 3.14}},
            'cstr':{
                'dTc': {'low': -10, "high": 10}
            }
        }
        return actions[project_name]
    
    function_descriptions = [
        {
            "name": "get_env_state",
            "description": "Get the env state variables.",
            "parameters": {
                "type": "object",
                "properties": {
                },
            }
        },
        {
            "name": "get_env_actions",
            "description": "Get the env action variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    #"ticker": {
                    #    "type": "string",
                    #    "description": "id of the stock, which is referred as ticker in the API"
                    #},
                },
                #"required": ["ticker"]
            }
        }
    ]

    def function_call(ai_response):
        function_call = ai_response["choices"][0]["message"]["function_call"]
        function_name = function_call["name"]
        arguments = function_call["arguments"]
        
        if function_name == "get_env_state":
            return get_env_state()
        elif function_name == "get_env_actions":
            #ticker = eval(arguments).get("ticker")
            transformed_obs = get_env_actions()
            return transformed_obs
        else:
            return
        
    def ask_function_calling(query):
        messages = []
        messages.append({"role": "system", "content": "Answer user questions as a Reinforcement Learning Engineer, giving python functions as answers."})
        messages.append({"role": "user", "content": query})
        #messages = [{"role": "user", "content": query}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions = function_descriptions,
            function_call="auto",
            temperature=0.2
        )

        #print('Response: ', response)

        while response["choices"][0]["finish_reason"] == "function_call":
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
        else:
            return response["choices"][0]["message"]["content"]

    st.title("Composabl Copilot")
    st.write("This is Composabl Copilot, a tool that you can use to design your autonomous system with the help of LLM.")
    explain = st.text_area('Explain your sim')

    prompt = f"""I have an autonomous system with these variables in the observation state of my simulation: transformed_obs = {get_env_state(project_name)} 
            and these actions: action = {get_env_actions(project_name)}. 
            You have access only to variables from observation state and action state, if you want to create calculated variables you have to declare it.
            Give custom name for each skill.
            Don't create negative rewards, because it is harmful for the model. Don't create rewards that are too big, because it is harmful for the model.
            Don't generate a combination of skills class.
            Give me some skills with proper reward to build this system, programmed in python, and following this structure:
            class skill(): def compute_reward(self, transformed_obs, action): return reward. \
                Here is a brief explanation for my use case: """
    prompt += explain

    teacher_class_template = '''
    class {class_name}(Teacher):
        def __init__(self):
            self.obs_history = None
            self.reward_history = []
            self.last_reward = 0

        def transform_obs(self, obs, action):
            return obs

        def transform_action(self, transformed_obs, action):
            return action

        def filtered_observation_space(self):
            return {sensors_vars}

        {reward_function}

        def compute_action_mask(self, transformed_obs, action):
            return None

        def compute_success_criteria(self, transformed_obs, action):
            return len(self.obs_history) > 100

        def compute_termination(self, transformed_obs, action):
            return False
        '''
    if st.button("Ask GPT", type='primary'):
        answer = ask_function_calling(prompt)
        st.write(answer)

        def replace_values_template(file_path, path_save, replace_dict):
            with open(file_path, 'r') as file:
                content = file.read()
                for key, value in replace_dict.items():
                    content = content.replace('{' + key + '}', value)
            with open(path_save, 'w') as file:
                file.write(content)

        ### Build Teachers
        # extract function from gpt answer
        def extract_function_from_string(code, function_name):
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return ast.unparse(node)
            return None
        
        def extract_class_from_string(code):
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name
            return None
        
        # extract code from gpt answer
        code_snippets = re.findall(r'```(.*?)```', answer, re.DOTALL)
        teacher_list = []
        teachers_classes = ''
        for snippet in code_snippets:
            lines = snippet.split('\n')[1:]  # Remove the first line
            dedented_snippet = textwrap.dedent('\n'.join(lines))
            extracted_function = extract_function_from_string(dedented_snippet, "compute_reward")
            class_name = extract_class_from_string(dedented_snippet)
            teacher_list.append(class_name)

            content = teacher_class_template.replace('{reward_function}', extracted_function.replace('\n', '\n        ')).replace('{class_name}', class_name)
            
            teachers_classes += content + '\n'

        state_vars = get_env_state(project_name)
        values_to_replace = {
            'sensors_vars': str(list(state_vars.keys())),
            'teacher_classes': textwrap.dedent(teachers_classes)
        } 
        
        replace_values_template('teacher_template.py', 'agent/'+ project_name + '/teacher.py' , values_to_replace)


        ##### Build Agent
        
        skill_declaration_template = '''
        {skill_name}_skill = Skill("{skill_name}", {skill_name}, trainable=True)
        for scenario_dict in default_scenarios:
            {skill_name}_skill.add_scenario(Scenario(scenario_dict))
            '''

        create_skill = ''
        for tc in teacher_list:
            create_skill += skill_declaration_template.replace('{skill_name}', tc) + '\n'

        add_skill = ''
        for tc in teacher_list:
            add_skill += 'agent.add_skill({}_skill) \n    '.format(tc)


        values_to_replace = {
            'sensors_vars': str(list(state_vars.keys())),
            'teacher_list': ','.join(teacher_list),
            'skills_vars_list': str([x + '_skill' for x in teacher_list]).replace("'",''),
            'skills_declaration': textwrap.indent(textwrap.dedent(create_skill),'   '),
            'add_skills_declaration': add_skill
        } 
        
        replace_values_template('agent_template.py', 'agent/'+ project_name + '/agent.py' , values_to_replace)

        # Prepare Zip File for download
        def zipdir(path, ziph, project):
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
            
        fp.close()
        #try:
        #    os.remove(project_name + '_llm_agent' + ".zip")
        #    st.success("File deleted successfully!")
        #except Exception as e:
        #    st.error(f"Error deleting the file: {e}")
