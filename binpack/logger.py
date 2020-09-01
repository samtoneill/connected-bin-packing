import pandas as pd
from gurobipy import GRB


def create_dataframe_log():
    return pd.DataFrame()

def dataframe_results_row(graph_id, no_items, no_connections, no_constraints, no_variables, run_time, obj_val, 
                            obj_bound, nodes_explored, feasible_sols, simplex_iterations):
    return pd.DataFrame.from_dict(
        {
        'graph_id': graph_id,
        'No Items': no_items,
        'No Connections': no_connections,
        'No Constraints': no_constraints,
        'No Variabes': no_variables,
        'Runtime':  [run_time],  
        'ObjVal':   [obj_val],
        'BstBnd':   obj_bound,
        'Gap':      (obj_val - obj_bound)/obj_val,
        'Nodes Explored': nodes_explored,
        'No Feasible Solutions': feasible_sols,
        'Simplex Iterations': simplex_iterations
        }
    )

def log_solution_by_time(model, where):
    if where == GRB.Callback.MIP:
        if model.cbGet(GRB.Callback.RUNTIME) > (model._log_time*model._log_iteration - 0.1):
            model._dataframe_results = model._dataframe_results.append(
                dataframe_results_row(
                    graph_id = model._graph_id,
                    no_items = model._no_items,
                    no_connections = model._no_connections,
                    no_constraints = model.getAttr("NumConstrs"),
                    no_variables = model.getAttr("NumVars"),
                    run_time = model.cbGet(GRB.Callback.RUNTIME),
                    obj_val = model.cbGet(GRB.Callback.MIP_OBJBST),
                    obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND),
                    nodes_explored = model.cbGet(GRB.Callback.MIP_NODCNT),
                    feasible_sols = model.cbGet(GRB.Callback.MIP_SOLCNT),
                    simplex_iterations = model.cbGet(GRB.Callback.MIP_ITRCNT)
                ),
                ignore_index=True
            )
            model._log_iteration += 1

# log the soltion for the remaining time left of the time limit, necessary for comparing results
def log_solution_for_remaining_time(m):
    m._dataframe_results = m._dataframe_results.append(
        dataframe_results_row(
            graph_id = m._graph_id,
            no_items = m._no_items,
            no_connections = m._no_connections,
            no_constraints = m.getAttr("NumConstrs"),
            no_variables = m.getAttr("NumVars"),
            run_time = m.Runtime,
            obj_val = m.objVal,
            obj_bound = m.objBound,
            nodes_explored = m.nodeCount,
            feasible_sols = m.solCount,
            simplex_iterations = m.iterCount
        ),
        ignore_index=True
    )
    for i in range(m._log_iteration*m._log_time, int(m.Params.timelimit)+1, m._log_time):
        m._dataframe_results = m._dataframe_results.append(
            dataframe_results_row(
                graph_id = m._graph_id,
                no_items = m._no_items,
                no_connections = m._no_connections,
                no_constraints = m.getAttr("NumConstrs"),
                no_variables = m.getAttr("NumVars"),
                run_time = i,
                obj_val = m.getAttr("ObjVal"),
                obj_bound = m.getAttr("ObjBound"),
                nodes_explored = m.getAttr("NodeCount"),
                feasible_sols = m.getAttr("SolCount"),
                simplex_iterations = m.getAttr("IterCount")
            ),
            ignore_index=True
        )

