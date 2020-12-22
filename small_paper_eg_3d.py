import sys
sys.path.append('binpack')

import gurobipy as gp
import binpack as bp
import pandas as pd
import matplotlib.pyplot as plt


experiment_name = 'small_paper_eg'
# reading csv file 
items_df = pd.read_csv("{}_items.csv".format(experiment_name)) 
connections_df = pd.read_csv("{}_connections.csv".format(experiment_name)) 
containers_df = pd.read_csv("{}_containers.csv".format(experiment_name)) 
model_prefix = 'model_3d_opt'
graph_id_min = items_df['graph_id'].min()
graph_id_max = items_df['graph_id'].max()
df_results = pd.DataFrame()

import os
if not os.path.exists('results/{}/{}'.format(experiment_name, model_prefix)):
    os.makedirs('results/{}/{}'.format(experiment_name, model_prefix))

graphs = [(i,items_df[items_df['graph_id'] == i], connections_df[connections_df['graph_id'] == i]) for i in range(graph_id_min, graph_id_max+1)]
for graph in graphs:
    model_name = '{}_{}'.format(model_prefix, graph[0])
    items_df_i = graph[1]
    connections_df_i = graph[2]
    items = {
        'Name' : items_df_i['Name'].to_list(),
        'Width' : items_df_i['Width'].to_list(),
        'Height' : items_df_i['Height'].to_list(),
        'Depth' : items_df_i['Depth'].to_list(),
        'x_fixed' : items_df_i['x_fixed'].to_list(),
        'y_fixed' : items_df_i['y_fixed'].to_list(),
        'z_fixed' : items_df_i['z_fixed'].to_list(),
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
        'ysj' : connections_df_i['ysj'].to_list(),
        'zsi' : connections_df_i['zsi'].to_list(),
        'zsj' : connections_df_i['zsj'].to_list()
    }
    containers = {
        'Width' : containers_df['Width'].to_list(),
        'Height' : containers_df['Height'].to_list(),
        'Depth' : containers_df['Depth'].to_list(),
        'x' : containers_df['x'].to_list(),
        'y' : containers_df['y'].to_list(),
        'z' : containers_df['z'].to_list(),
        'cost' : containers_df['cost'].to_list()
    }


    m = bp.create_3d_model(items, connections, containers, model_name=model_name, margin=1, pandas_log=True)

    # Set parameters
    #m.Params.timelimit = 0.01
    m.Params.timelimit = 1000.0
    m.Params.Heuristics = 0.1

    bp.optimize_model(m)
    bp.print_solution(m)
    bp.save_items_for_cad(items,containers,m,filename=m.modelName,filepath= 'results/{}/{}'.format(experiment_name, model_prefix))

    #try:
    # aligned view elev=0, azim=90
    # view 1 elev=30, azim=165
    fig, ax = bp.plot_solution_3d(items, connections, containers, m,plot_connections=True, 
                                    show=False, legend=False, lims=[3,12,4],elev=30, azim=165) 
    ax.set_box_aspect([3,6,3])
    fig.show()
    bbox = fig.bbox_inches.from_bounds(0.5, 0.5, 5.5, 4)
    fig.savefig('results/{}/{}/{}_view1.pdf'.format(experiment_name, model_prefix, m.modelName),bbox_inches=bbox)
    fig, ax = bp.plot_solution_3d(items, connections, containers, m, plot_items=True,plot_connections=True, 
                                    show=False, legend=False, lims=[3,12,4],elev=0, azim=90) 
    ax.set_box_aspect([3,6,3])   
    fig.show() 
    bbox = fig.bbox_inches.from_bounds(1.5, 1, 3, 3)
    fig.savefig('results/{}/{}/{}_view2.pdf'.format(experiment_name, model_prefix, m.modelName),bbox_inches=bbox)
    m.write(os.path.join('results/{}/{}/{}.lp'.format(experiment_name, model_prefix, m.modelName)))
    df_results = df_results.append(m._dataframe_results)
    print(m._dataframe_results.head())

    plt.show()
    #except:
    #    print('Error')

df_results.to_csv('results/{}/{}/{}_solutions.csv'.format(experiment_name,model_prefix,experiment_name))
