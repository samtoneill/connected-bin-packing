import sys
sys.path.append('binpack')

import gurobipy as gp
import binpack as bp
import pandas as pd
import os
import time
import pickle

cwd = os.getcwd()
experiment_name = 'test_data_4-18'
model_prefix = 'model_2d_reverse'
results_folder = os.path.join(cwd, 'results', experiment_name, model_prefix)
log_file_name = '{}_solutions.csv'.format(experiment_name)
print(results_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if os.path.isfile(os.path.join(results_folder,log_file_name)):
    df_results = pd.read_csv(os.path.join(results_folder,log_file_name), encoding='utf-8', index_col='graph_id')
else:
    df_results = bp.create_dataframe_log()

# reading csv file 
items_df = pd.read_csv("{}_items.csv".format(experiment_name)) 
connections_df = pd.read_csv("{}_connections.csv".format(experiment_name)) 
containers_df = pd.read_csv("{}_containers_2d.csv".format(experiment_name)) 

graph_id_min = items_df['graph_id'].min()
graph_id_max = items_df['graph_id'].max()


graphs = [(i,items_df[items_df['graph_id'] == i], connections_df[connections_df['graph_id'] == i]) for i in range(graph_id_min, graph_id_max+1)]
for graph in graphs[::-1]:
    graph_id = graph[0]
    model_name = '{}_{}'.format(model_prefix, graph_id)
    items_df_i = graph[1]
    connections_df_i = graph[2]
    items = {
        'Name' : items_df_i['Name'].to_list(),
        'Width' : items_df_i['Width'].to_list(),
        'Height' : items_df_i['Height'].to_list(),
        'Depth' : items_df_i['Depth'].to_list(),
        'x_fixed' : items_df_i['x_fixed'].to_list(),
        'y_fixed' : items_df_i['y_fixed'].to_list(),
        'container_fixed' : items_df_i['container_fixed'].to_list()
    }
    connections = {
        'i' : connections_df_i['source'].to_list(),
        'j' : connections_df_i['target'].to_list(),
        'st_id' : connections_df_i['st_id'].to_list(),
        'cij' : connections_df_i['cost'].to_list(),
        'xsi' : connections_df_i['xsi'].to_list(),
        'ysi' : connections_df_i['ysi'].to_list(),
        'xsj' : connections_df_i['xsj'].to_list(),
        'ysj' : connections_df_i['ysj'].to_list()
    }
    containers = {
        'Width' : containers_df['Width'].to_list(),
        'Height' : containers_df['Height'].to_list(),
        'x' : containers_df['x'].to_list(),
        'y' : containers_df['y'].to_list(),
        'cost' : containers_df['cost'].to_list()
    }       


    m = bp.create_2d_model(items, connections, containers, model_name=model_name, pandas_log=True, graph_id=graph_id)
    
    # Set parameters
    m.Params.timelimit = 200.0
    m.Params.Heuristics = 0.1
    # use this to compute the infeasible subset
    #m.computeIIS()
    #m.write('error.ilp')
    bp.optimize_model(m)

    bp.log_solution_for_remaining_time(m)
    #try:
    #bp.print_solution(m)
    #fig, ax = bp.plot_solution_2d(items, connections, containers, m, plot_items=True, plot_connections=False, show=False, legend=True, figsize=(21,21))
    #fig.savefig(os.path.join(results_folder,'{}_{}_items.png'.format(m.modelName, m._no_items)))
    fig, ax = bp.plot_solution_2d(items, connections, containers, m, plot_items=True, plot_connections=True, show=True, legend=False, figsize=(25,25))
    fig.savefig(os.path.join(results_folder,'{}_{}_items_w_connections.pdf'.format(m.modelName, m._no_items)))
    m.write(os.path.join(results_folder,'{}_{}_items.sol'.format(m.modelName, m._no_items)))
    m.write(os.path.join(results_folder,'{}_{}_items.lp'.format(m.modelName, m._no_items)))
    df_results = df_results.append(m._dataframe_results,ignore_index=True)
    #except:
    #    print('Error')

df_results.to_csv(os.path.join(results_folder,'{}_solutions.csv'.format(experiment_name)))
