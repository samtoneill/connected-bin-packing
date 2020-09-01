import sys
sys.path.append("binpack")

import gurobipy as gp
import binpack as bp

# note that containers are indexed from 0
items = {
    'Width' : [4,3,6,8],
    'Height' : [5,7,4,7],
    #'x_fixed' : ['inf', 'inf', 3, 1],
    'x_fixed' : ['inf', 'inf', 'inf', 'inf'],
    #'y_fixed' : ['inf', 'inf', 4, 1],
    'y_fixed' : ['inf', 'inf', 'inf', 'inf'],
    #'container_fixed' : ['inf', 'inf', 1, 0]
    'container_fixed' : ['inf', 'inf', 'inf', 'inf']
}
connections = {
    'i' : [0,0,0,1,3,2],
    'j' : [1,2,3,3,1,0],
    'cij' : [1,1,1,1,1,2,3],
    'xsi' : [0,0,1,0,0,0,0],
    'ysi' : [0,0,1,0,0,0,0],
    'xsj' : [0,0,0,0,0,0,0],
    'ysj' : [0,0,0,0,0,0,0]
}
# containers = {
#     'Width' : [8,8],
#     'Height' : [11,11],
#     'x' : [0,0],
#     'y' : [0,11],
#     'cost' : [0,0]
# }
containers = {
    'Width' : [20,20],
    'Height' : [11,21],
    'x' : [0,0],
    'y' : [0,15],
    'cost' : [0,100]
}
m = bp.create_2d_model(items, connections, containers, model_name='test_2d_no_log')
    
# Set parameters
m.Params.timelimit = 600.0
m.Params.Heuristics = 0.5

bp.optimize_model(m)

m.write("{}.lp".format(m.model_name))
m.write("{}.sol".format(m.model_name))

fig, ax = bp.plot_solution_2d(items, connections, containers, m, show=False)
fig.savefig('{}.png'.format(m.modelName))

m = bp.create_2d_model(items, connections, containers, model_name='test_2d_log', pandas_log=True)
    
# Set parameters
m.Params.timelimit = 600.0
m.Params.Heuristics = 0.5

bp.optimize_model(m)
bp.print_solution(m)
bp.log_solution_for_remaining_time(m)
print(m._dataframe_results)
m.write("{}.lp".format(m.model_name))
m.write("{}.sol".format(m.model_name))
bp.plot_solution_2d(items,connections,containers,m)
print('No Constraints = {}'.format(m.getAttr("NumConstrs")))
print('No Variables = {}'.format(m.getAttr("NumVars")))
