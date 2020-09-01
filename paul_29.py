import sys
sys.path.append('binpack')

import gurobipy as gp
import binpack as bp
import pandas as pd

# name the experiment and the model
experiment_name = 'paul_29'
model_prefix = 'model_3d'

# reading csv files into panda dataframes, note that this is based on the experiment name
items_df = pd.read_csv("{}_items.csv".format(experiment_name)) 
connections_df = pd.read_csv("{}_connections.csv".format(experiment_name)) 
containers_df = pd.read_csv("{}_containers.csv".format(experiment_name)) 


# set up the results folder for output
import os
if not os.path.exists('results/{}/{}'.format(experiment_name, model_prefix)):
    os.makedirs('results/{}/{}'.format(experiment_name, model_prefix))

# read the panda dataframes into python dictionaries
items = {
    'Name' : items_df['Name'].to_list(),
    'Width' : items_df['Width'].to_list(),
    'Height' : items_df['Height'].to_list(),
    'Depth' : items_df['Depth'].to_list(),
    'x_fixed' : items_df['x_fixed'].to_list(),
    'y_fixed' : items_df['y_fixed'].to_list(),
    'z_fixed' : items_df['z_fixed'].to_list(),
    'container_fixed' : items_df['container_fixed'].to_list()
}
connections = {
    'i' : connections_df['source'].to_list(),
    'j' : connections_df['target'].to_list(),
    'st_id' : connections_df['st_id'].to_list(),
    'cij' : connections_df['cost'].to_list(),
    'xsi' : connections_df['xsi'].to_list(),
    'ysi' : connections_df['ysi'].to_list(),
    'xsj' : connections_df['xsj'].to_list(),
    'ysj' : connections_df['ysj'].to_list(),
    'zsi' : connections_df['zsi'].to_list(),
    'zsj' : connections_df['zsj'].to_list()
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

# create the gurobi model

m = bp.create_3d_model(items, connections, containers, model_name=model_prefix)

# Set parameters
m.Params.timelimit = 1000.0
m.Params.Heuristics = 0.5

# optimise the model
bp.optimize_model(m)

# save the item locations for input to CAD
bp.save_items_for_cad(items,containers,m,filename=m.modelName,filepath= 'results/{}/{}'.format(experiment_name, model_prefix))
# print the value of the gurobi variables
bp.print_items(items,containers,m)

# plot and then save
fig, ax = bp.plot_solution_3d(items, connections, containers, m, plot_items=True, plot_connections=True, show=True, legend=False, lims=[40,40,40])
fig.savefig('results/{}/{}/{}.png'.format(experiment_name, model_prefix, m.modelName))