import sys
sys.path.append("binpack")

import gurobipy as gp
import binpack as bp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

items_dict = {
    'Name'  : ['a','b','c','d'],
    'Width'  : [4,3,6,7],
    'Height' : [5,7,4,7],
    'Depth'  : [5,5,5,5],
    'x_fixed' : [1, 'inf', 'inf', 1],
    'y_fixed' : [30, 'inf', 'inf', 1],
    'z_fixed' : [1, 'inf', 'inf', 1],
    'container_fixed' : ['inf', 'inf', 0, 0]
}
connections_dict = {
    'i' : [0,0,1,1],
    'j' : [1,2,2,3],
    'st_id' : [0,0,0,0],
    'cij' : [1,1,1,1],
    'xsi' : [0,0,0,0],
    'ysi' : [0,0,0,0],
    'zsi' : [0,0,0,0],
    'xsj' : [0,0,0,0],
    'ysj' : [0,0,0,0],
    'zsj' : [0,0,0,0]
}
containers_dict = {
    'Width' : [8],
    'Height' : [40],
    'Depth' : [12],
    'x' : [0],
    'y' : [0],
    'z' : [0],
    'cost' : [0]
}

m = bp.create_3d_model(items_dict, connections_dict, containers_dict, model_name='test_3d')
    
# Set parameters
m.Params.timelimit = 600.0
m.Params.Heuristics = 0.5

bp.optimize_model(m)
bp.print_solution(m)
m.write("{}.lp".format(m.model_name))
#m.write("{}.sol".format(m.model_name))
print(m.getAttr("SolCount"))
print(m.getAttr("NodeCount"))
print(m.getAttr("ObjBound"))
print(m.getAttr("ObjVal"))
print(m.getAttr("IterCount"))
print('No Constraints = {}'.format(m.getAttr("NumConstrs")))
print('No Variables = {}'.format(m.getAttr("NumVars")))


    

fig, ax = bp.plot_solution_3d(items_dict, connections_dict, containers_dict, m, plot_items=True,plot_connections=True, show=True, legend=False, lims=[8,40,12])


