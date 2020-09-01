# import gurobi
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from logger import *


def create_2d_model(items_dict, connections_dict, containers_dict, model_name, 
                    margin=1, pandas_log=False, graph_id=None, log_time=2):

    
    # Create a new model
    m = gp.Model(model_name)
    
    m._pandas_log = pandas_log
    m._graph_id = graph_id
    m._log_time = log_time
    m._log_iteration = 0
    m._dataframe_results = pd.DataFrame()


    w = items_dict['Width']
    h = items_dict['Height']
    x_fixed = items_dict['x_fixed']
    y_fixed = items_dict['y_fixed']
    container_fixed = items_dict['container_fixed']

    s = connections_dict['i']
    t = connections_dict['j']
    cij = connections_dict['cij']
    xsi = connections_dict['xsi']
    ysi = connections_dict['ysi']
    xsj = connections_dict['xsj']
    ysj = connections_dict['ysj']

    W = containers_dict['Width']
    H = containers_dict['Height']
    
    # define module base co-ordinates
    mx = containers_dict['x']
    my = containers_dict['y']

    cK = containers_dict['cost']


    no_modules = len(mx)
    no_items = len(w)
    connections = [(i,j,xsi,ysi,xsj,ysj,cij) for (i,j,xsi,ysi,xsj,ysj,cij) in zip(s,t,xsi,ysi,xsj,ysj,cij)]
    V_prime = [(u,v) for u in range(no_items) for v in range(no_items) if u < v]

    M = 9999

    m._no_items = no_items
    m._no_connections= len(connections)

    x = m.addVars(range(no_items),
                    name="x")
    y = m.addVars(range(no_items),
                    name="y")

    xuv = m.addVars(V_prime, vtype=GRB.BINARY, name="xuv")
    yuv = m.addVars(V_prime, vtype=GRB.BINARY, name="yuv")
    Rij = m.addVars(zip(s,t), name="Rij")
    Bij = m.addVars(zip(s,t), name="Bij")

    r = m.addVars(range(no_items), 4, vtype=GRB.BINARY, name="r")

    mvk = m.addVars(range(no_items), range(no_modules), vtype=GRB.BINARY, name="mvk")
    mk = m.addVars(range(no_modules), vtype=GRB.BINARY, name="mk")
    nuvk = m.addVars([(u,v,k) for (u,v) in V_prime for k in  range(no_modules)], vtype=GRB.BINARY, name="nuvk")
    Nuv = m.addVars(V_prime, vtype=GRB.BINARY, name="Nuv")

    #-----------------------------------------------------------------------------------
    # Rotational Constraints
    #-----------------------------------------------------------------------------------

    m.addConstrs(
        (r.sum(v, '*') == 1 for v in range(no_items)), 
        name="onerotation"
    )

    #-----------------------------------------------------------------------------------
    # Container constraints
    #-----------------------------------------------------------------------------------

    # Item can be in only one container
    m.addConstrs(
        (mvk.sum(v, '*') == 1 for v in range(no_items)), 
        name="onecontainer"
    )   

    m.addConstrs((x[v] + (r[v,0]+r[v,2])*w[v] + (r[v,1]+r[v,3])*h[v] <= W[k] + (1-mk[k])*M for v in range(no_items) for k in range(no_modules)),name="boxxv")    
    m.addConstrs((y[v] + (r[v,0]+r[v,2])*h[v] + (r[v,1]+r[v,3])*w[v] <= H[k] + (1-mk[k])*M for v in range(no_items) for k in range(no_modules)),name="boxyv")

    # fix item in container
    m.addConstrs((mvk[v,container_fixed[v]] == 1 for v in range(no_items) if not container_fixed[v] == 'inf'), name = "container_fixed")


    #-----------------------------------------------------------------------------------
    # Fixed Items
    #-----------------------------------------------------------------------------------

    # constraint to fix item
    m.addConstrs((x[v] <= x_fixed[v] for v in range(no_items) if not x_fixed[v] == 'inf'), name="xv_high")
    m.addConstrs((x[v] >= x_fixed[v] for v in range(no_items) if not x_fixed[v] == 'inf'), name="xv_low")

    m.addConstrs((y[v] <= y_fixed[v] for v in range(no_items) if not y_fixed[v] == 'inf'), name="yv_high")
    m.addConstrs((y[v] >= y_fixed[v] for v in range(no_items) if not y_fixed[v] == 'inf'), name="yv_low")

    #-----------------------------------------------------------------------------------


    #-----------------------------------------------------------------------------------
    # Non-overlapping constraints
    #-----------------------------------------------------------------------------------
    # logical and for nij
    m.addConstrs(
        (nuvk[u,v,k] >= mvk[u,k] + mvk[v,k] - 1 for (u,v) in V_prime for k in range(no_modules)), 
        name='nuvklogical1'
    )
    m.addConstrs(
        (nuvk[u,v,k] <= mvk[u,k] for (u,v) in V_prime for k in range(no_modules)), 
        name='nuvklogical2'
        )
    m.addConstrs(
        (nuvk[u,v,k] <= mvk[v,k] for (u,v) in V_prime for k in range(no_modules)), 
        name='nuvklogical3'
    )

    m.addConstrs(
    (Nuv[u,v] == nuvk.sum(u,v, '*') for (u,v) in V_prime), 
    name='Nuvlogical1'
    )

    # x axis
    m.addConstrs((x[u]+ (r[u,0]+r[u,2])*w[u] + (r[u,1]+r[u,3])*h[u] + margin <=
                    x[v] + M*(xuv[u,v] + yuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapxuv")
    m.addConstrs((x[v]+ (r[v,0]+r[v,2])*w[v] + (r[v,1]+r[v,3])*h[v]  + margin <=
                    x[u] + M*(1-xuv[u,v] + yuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapxvu")

    # y axis
    m.addConstrs((y[u]+ (r[u,0]+r[u,2])*h[u] + (r[u,1]+r[u,3])*w[u]  + margin <=
                    y[v] + M*(1 + xuv[u,v] - yuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapyuv")
    m.addConstrs((y[v]+ (r[v,0]+r[v,2])*h[v] + (r[v,1]+r[v,3])*w[v] + margin <=
                    y[u] + M*(2 - xuv[u,v] - yuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapyvu")

    #-----------------------------------------------------------------------------------
    # Rectilinear constraints
    #-----------------------------------------------------------------------------------
    m.addConstrs(((x[i] + (r[i,0]+r[i,2])*w[i]/2 +(r[i,1]+r[i,3])*h[i]/2 + gp.quicksum(mvk[i,k]*mx[k] for k in range(no_modules)) ) 
                    - (x[j] + (r[j,0]+r[j,2])*w[j]/2 +(r[j,1]+r[j,3])*h[j]/2 + gp.quicksum(mvk[j,k]*mx[k] for k in range(no_modules))) 
                    + ( r[i,0]*xsi-r[i,1]*ysi-r[i,2]*xsi+r[i,3]*ysi ) 
                    - ( r[j,0]*xsj-r[j,1]*ysj-r[j,2]*xsj + r[j,3]*ysj ) 
                    <= Rij[i,j] 
                    for (i,j,xsi,ysi,xsj,ysj,cij) in connections), name="rectRij")

    m.addConstrs((-1*(
                    (x[i] + (r[i,0]+r[i,2])*w[i]/2 +(r[i,1]+r[i,3])*h[i]/2 + gp.quicksum(mvk[i,k]*mx[k] for k in range(no_modules))) 
                    - (x[j] + (r[j,0]+r[j,2])*w[j]/2 +(r[j,1]+r[j,3])*h[j]/2 + gp.quicksum(mvk[j,k]*mx[k] for k in range(no_modules)))  
                    + ( r[i,0]*xsi-r[i,1]*ysi-r[i,2]*xsi+r[i,3]*ysi ) 
                    - ( r[j,0]*xsj-r[j,1]*ysj-r[j,2]*xsj + r[j,3]*ysj ) 
                    )
                    <= Rij[i,j]
                    for (i,j,xsi,ysi,xsj,ysj,cij) in connections), name="rectRji")

    m.addConstrs(((y[i] + (r[i,0]+r[i,2])*h[i]/2 +(r[i,1]+r[i,3])*w[i]/2 + gp.quicksum(mvk[i,k]*my[k] for k in range(no_modules))) 
                    - (y[j] + (r[j,0]+r[j,2])*h[j]/2 +(r[j,1]+r[j,3])*w[j]/2  + gp.quicksum(mvk[j,k]*my[k] for k in range(no_modules)))  
                    + ( r[i,0]*ysi+r[i,1]*xsi-r[i,2]*ysi - r[i,3]*xsi )
                    - ( r[j,0]*ysj+r[j,1]*xsj-r[j,2]*ysj - r[j,3]*xsj )
                    <= Bij[i,j]
                    for (i,j,xsi,ysi,xsj,ysj,cij) in connections), name="rectBij")

    m.addConstrs((-1*(
                    (y[i] + (r[i,0]+r[i,2])*h[i]/2 +(r[i,1]+r[i,3])*w[i]/2 + gp.quicksum(mvk[i,k]*my[k] for k in range(no_modules))) 
                    - (y[j] + (r[j,0]+r[j,2])*h[j]/2 +(r[j,1]+r[j,3])*w[j]/2 + gp.quicksum(mvk[j,k]*my[k] for k in range(no_modules)))
                    + ( r[i,0]*ysi+r[i,1]*xsi-r[i,2]*ysi - r[i,3]*xsi )
                    - ( r[j,0]*ysj+r[j,1]*xsj-r[j,2]*ysj - r[j,3]*xsj )
                    )  
                    <= Bij[i,j]
                    for (i,j,xsi,ysi,xsj,ysj,cij) in connections), name="rectBji")

    #-----------------------------------------------------------------------------------
    # Container Used Constraints
    #-----------------------------------------------------------------------------------
    m.addConstrs(
        (mk[k] >= mvk[v,k] for v in range(no_items) for k in range(no_modules)), 
        name='mklogical1'
    )
    m.addConstrs(
        (mk[k] <= gp.quicksum(mvk[v,k] for v in range(no_items)) for k in range(no_modules)), 
        name='mklogical2'
        )

    # Set objective
    m.setObjective(gp.quicksum(cij*(Rij[i,j]+Bij[i,j]) for (i,j,xsi,ysi,xsj,ysj, cij) in connections) + gp.quicksum(cK[k]*mk[k] for k in range(no_modules)), GRB.MINIMIZE)
    
    # These can be used to minimise/maximise the number of containers_dict used, good for testing correctness
    #m.setObjective(gp.quicksum(Nuv[u,v] for u in range(no_items) for v in range(no_items) ), GRB.MINIMIZE)
    #m.setObjective(gp.quicksum(Nuv[u,v] for u in range(no_items) for v in range(no_items) ), GRB.MAXIMIZE)
    m.update()
    return m

def create_3d_model(items_dict, connections_dict, containers_dict, model_name, 
                    margin=1, pandas_log=False, graph_id=None, log_time=2):

    
    # Create a new model
    m = gp.Model(model_name)
    
    m._pandas_log = pandas_log
    m._graph_id = graph_id
    m._log_time = log_time
    m._log_iteration = 0
    m._dataframe_results = pd.DataFrame()

    name = items_dict['Name']
    w = items_dict['Width']
    h = items_dict['Height']
    d = items_dict['Depth']
    x_fixed = items_dict['x_fixed']
    y_fixed = items_dict['y_fixed']
    z_fixed = items_dict['z_fixed']
    container_fixed = items_dict['container_fixed']

    s = connections_dict['i']
    t = connections_dict['j']
    st_id = connections_dict['st_id']
    cij = connections_dict['cij']
    xsi = connections_dict['xsi']
    ysi = connections_dict['ysi']
    zsi = connections_dict['zsi']
    xsj = connections_dict['xsj']
    ysj = connections_dict['ysj']
    zsj = connections_dict['zsj']

    W = containers_dict['Width']
    H = containers_dict['Height']
    D = containers_dict['Depth']
    
    # define module base co-ordinates
    mx = containers_dict['x']
    my = containers_dict['y']
    mz = containers_dict['y']

    cK = containers_dict['cost']


    no_modules = len(mx)
    no_items = len(w)
    connections = [(i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in zip(s,t,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij)]
    V_prime = [(u,v) for u in range(no_items) for v in range(no_items) if u < v]
    
    M = 9999

    m._no_items = no_items
    m._no_connections= len(connections)

    x = m.addVars(range(no_items),
                    name="x")
    y = m.addVars(range(no_items),
                    name="y")

    z = m.addVars(range(no_items),
                    name="z")

    xuv = m.addVars(V_prime, vtype=GRB.BINARY, name="xuv")
    yuv = m.addVars(V_prime, vtype=GRB.BINARY, name="yuv")
    zuv = m.addVars(V_prime, vtype=GRB.BINARY, name="zuv")
    Rij = m.addVars(zip(s,t,st_id), name="Rij")
    Bij = m.addVars(zip(s,t,st_id), name="Bij")
    Fij = m.addVars(zip(s,t,st_id), name="Fij")

    r = m.addVars(range(no_items), 4, vtype=GRB.BINARY, name="r")

    mvk = m.addVars(range(no_items), range(no_modules), vtype=GRB.BINARY, name="mvk")
    mk = m.addVars(range(no_modules), vtype=GRB.BINARY, name="mk")
    nuvk = m.addVars([(u,v,k) for (u,v) in V_prime for k in  range(no_modules)], vtype=GRB.BINARY, name="nuvk")
    Nuv = m.addVars(V_prime, vtype=GRB.BINARY, name="Nuv")

    #-----------------------------------------------------------------------------------
    # Rotational Constraints
    #-----------------------------------------------------------------------------------

    m.addConstrs(
        (r.sum(v, '*') == 1 for v in range(no_items)), 
        name="onerotation"
    )

    #-----------------------------------------------------------------------------------
    # Container constraints
    #-----------------------------------------------------------------------------------

    # Item can be in only one container
    m.addConstrs(
        (mvk.sum(v, '*') == 1 for v in range(no_items)), 
        name="onecontainer"
    )   

    m.addConstrs((x[v] + (r[v,0]+r[v,2])*w[v] + (r[v,1]+r[v,3])*h[v] <= W[k] + (1-mk[k])*M for v in range(no_items) for k in range(no_modules)),name="boxxv")
    m.addConstrs((y[v] + (r[v,0]+r[v,2])*h[v] + (r[v,1]+r[v,3])*w[v] <= H[k] + (1-mk[k])*M for v in range(no_items) for k in range(no_modules)),name="boxyv")
    m.addConstrs((z[v] + d[v] <= D[k] + (1-mk[k])*M for v in range(no_items) for k in range(no_modules)),name="boxzv")

    # fix item in container
    m.addConstrs((mvk[v,int(container_fixed[v])] == 1 for v in range(no_items) if not str(container_fixed[v]) == 'inf'), name = "container_fixed")

    #-----------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------
    # Container constraints
    #-----------------------------------------------------------------------------------

    # constraint to fix item
    m.addConstrs((x[v] <= x_fixed[v] for v in range(no_items) if not str(x_fixed[v]) == 'inf'), name="xv_high")
    m.addConstrs((x[v] >= x_fixed[v] for v in range(no_items) if not str(x_fixed[v]) == 'inf'), name="xv_low")

    m.addConstrs((y[v] <= y_fixed[v] for v in range(no_items) if not str(y_fixed[v]) == 'inf'), name="yv_high")
    m.addConstrs((y[v] >= y_fixed[v] for v in range(no_items) if not str(y_fixed[v]) == 'inf'), name="yv_low")

    m.addConstrs((z[v] <= z_fixed[v] for v in range(no_items) if not str(z_fixed[v]) == 'inf'), name="zv_high")
    m.addConstrs((z[v] >= z_fixed[v] for v in range(no_items) if not str(z_fixed[v]) == 'inf'), name="zv_low")

    #-----------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------
    # Non-overlapping constraints
    #-----------------------------------------------------------------------------------
    # logical and for nij
    m.addConstrs(
        (nuvk[u,v,k] >= mvk[u,k] + mvk[v,k] - 1 for (u,v) in V_prime for k in range(no_modules)), 
        name='nuvklogical1'
    )
    m.addConstrs(
        (nuvk[u,v,k] <= mvk[u,k] for (u,v) in V_prime for k in range(no_modules)), 
        name='nuvklogical2'
        )
    m.addConstrs(
        (nuvk[u,v,k] <= mvk[v,k] for (u,v) in V_prime for k in range(no_modules)), 
        name='nuvklogical3'
    )

    m.addConstrs(
    (Nuv[u,v] == nuvk.sum(u,v, '*') for (u,v) in V_prime), 
    name='Nuvlogical1'
    )

    # x axis
    m.addConstrs((x[u]+ (r[u,0]+r[u,2])*w[u] + (r[u,1]+r[u,3])*h[u] + margin <=
                    x[v] + M*(xuv[u,v] + yuv[u,v] + zuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapxuv")
    m.addConstrs((x[v]+ (r[v,0]+r[v,2])*w[v] + (r[v,1]+r[v,3])*h[v]  + margin <=
                    x[u] + M*(1-xuv[u,v] + yuv[u,v] + zuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapxvu")

    # y axis
    m.addConstrs((y[u]+ (r[u,0]+r[u,2])*h[u] + (r[u,1]+r[u,3])*w[u]  + margin <=
                    y[v] + M*(1 + xuv[u,v] - yuv[u,v] + zuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapyuv")
    m.addConstrs((y[v]+ (r[v,0]+r[v,2])*h[v] + (r[v,1]+r[v,3])*w[v] + margin <=
                    y[u] + M*(2 - xuv[u,v] - yuv[u,v] + zuv[u,v]) + M*(1-Nuv[u,v])
                    for (u,v) in V_prime), name="overlapyvu")

    # z axis

    m.addConstrs((z[u]+ d[u]  + margin <=
                z[v] + M*(2 - xuv[u,v] + yuv[u,v] - zuv[u,v])
                for (u,v) in V_prime), name="overlapzuv")

    m.addConstrs((z[v] + d[v] + margin <=
                z[u] + M*(2 + xuv[u,v] - yuv[u,v] - zuv[u,v])
                for (u,v) in V_prime), name="overlapzvu")

    m.addConstrs((xuv[u,v]+yuv[u,v]+zuv[u,v] <= 2
                for (u,v) in V_prime), name="eight_to_six1")

    m.addConstrs((-1*(xuv[u,v]+yuv[u,v]+zuv[u,v] +(1-zuv[u,v])*M) <= -2
                for (u,v) in V_prime), name="eight_to_six2")

    #-----------------------------------------------------------------------------------
    # Rectilinear constraints
    #-----------------------------------------------------------------------------------

    m.addConstrs(((x[i] + (r[i,0]+r[i,2])*w[i]/2 +(r[i,1]+r[i,3])*h[i]/2 + gp.quicksum(mvk[i,k]*mx[k] for k in range(no_modules)) ) 
                    - (x[j] + (r[j,0]+r[j,2])*w[j]/2 +(r[j,1]+r[j,3])*h[j]/2 + gp.quicksum(mvk[j,k]*mx[k] for k in range(no_modules))) 
                    + ( r[i,0]*xsi-r[i,1]*ysi-r[i,2]*xsi+r[i,3]*ysi ) 
                    - ( r[j,0]*xsj-r[j,1]*ysj-r[j,2]*xsj + r[j,3]*ysj ) 
                    <= Rij[i,j,st_id] 
                    for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections), name="rectRij")

    m.addConstrs((-1*(
                    (x[i] + (r[i,0]+r[i,2])*w[i]/2 +(r[i,1]+r[i,3])*h[i]/2 + gp.quicksum(mvk[i,k]*mx[k] for k in range(no_modules))) 
                    - (x[j] + (r[j,0]+r[j,2])*w[j]/2 +(r[j,1]+r[j,3])*h[j]/2 + gp.quicksum(mvk[j,k]*mx[k] for k in range(no_modules)))  
                    + ( r[i,0]*xsi-r[i,1]*ysi-r[i,2]*xsi+r[i,3]*ysi ) 
                    - ( r[j,0]*xsj-r[j,1]*ysj-r[j,2]*xsj + r[j,3]*ysj ) 
                    )
                    <= Rij[i,j,st_id]
                    for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections), name="rectRji")

    m.addConstrs(((y[i] + (r[i,0]+r[i,2])*h[i]/2 +(r[i,1]+r[i,3])*w[i]/2 + gp.quicksum(mvk[i,k]*my[k] for k in range(no_modules))) 
                    - (y[j] + (r[j,0]+r[j,2])*h[j]/2 +(r[j,1]+r[j,3])*w[j]/2  + gp.quicksum(mvk[j,k]*my[k] for k in range(no_modules)))  
                    + ( r[i,0]*ysi+r[i,1]*xsi-r[i,2]*ysi - r[i,3]*xsi )
                    - ( r[j,0]*ysj+r[j,1]*xsj-r[j,2]*ysj - r[j,3]*xsj )
                    <= Bij[i,j,st_id]
                    for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections), name="rectBij")

    m.addConstrs((-1*(
                    (y[i] + (r[i,0]+r[i,2])*h[i]/2 +(r[i,1]+r[i,3])*w[i]/2 + gp.quicksum(mvk[i,k]*my[k] for k in range(no_modules))) 
                    - (y[j] + (r[j,0]+r[j,2])*h[j]/2 +(r[j,1]+r[j,3])*w[j]/2 + gp.quicksum(mvk[j,k]*my[k] for k in range(no_modules)))
                    + ( r[i,0]*ysi+r[i,1]*xsi-r[i,2]*ysi - r[i,3]*xsi )
                    - ( r[j,0]*ysj+r[j,1]*xsj-r[j,2]*ysj - r[j,3]*xsj )
                    )  
                    <= Bij[i,j,st_id]
                    for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections), name="rectBji")

                    
    m.addConstrs(((z[i] + gp.quicksum(mvk[i,k]*my[k] for k in range(no_modules))) 
                    - (z[j] + gp.quicksum(mvk[j,k]*my[k] for k in range(no_modules)))  
                    + zsi -zsj
                    <= Fij[i,j,st_id]
                    for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections), name="rectFij")

    m.addConstrs((-1*(
                    (z[i] + gp.quicksum(mvk[i,k]*my[k] for k in range(no_modules))) 
                    - (z[j] + gp.quicksum(mvk[j,k]*my[k] for k in range(no_modules)))  
                    + zsi -zsj
                    )
                    <= Fij[i,j,st_id]
                    for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections), name="rectFji")   

    #-----------------------------------------------------------------------------------
    # Container Used Constraints
    #-----------------------------------------------------------------------------------
    m.addConstrs(
        (mk[k] >= mvk[v,k] for v in range(no_items) for k in range(no_modules)), 
        name='mklogical1'
    )
    m.addConstrs(
        (mk[k] <= gp.quicksum(mvk[v,k] for v in range(no_items)) for k in range(no_modules)), 
        name='mklogical2'
        )

    # Set objective
    m.setObjective(gp.quicksum(cij*(Rij[i,j,st_id]+Bij[i,j,st_id]+Fij[i,j,st_id]) for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections) + gp.quicksum(cK[k]*mk[k] for k in range(no_modules)), GRB.MINIMIZE)
    #m.setObjective(gp.quicksum(cij*(Rij[i,j]+Bij[i,j]+Fij[i,j]) for (i,j,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections) , GRB.MINIMIZE)
    
    # These can be used to minimise/maximise the number of containers used, good for testing correctness
    #m.setObjective(gp.quicksum(Nuv[u,v] for u in range(no_items) for v in range(no_items) ), GRB.MINIMIZE)
    #m.setObjective(gp.quicksum(Nuv[u,v] for u in range(no_items) for v in range(no_items) ), GRB.MAXIMIZE)
    m.update()
    return m

def optimize_model(model):
    if model._pandas_log:
        # Optimize model
        model.optimize(log_solution_by_time)
    else:
        model.optimize()

def print_solution(model):
    #print solution
    for v in model.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % model.objVal)

def print_items(items,containers,m):
    for i in range(len(items['Width'])):
        k = 99
        for c in range(len(containers['Width'])):
            if m.getVarByName("mvk[{},{}]".format(i,c)).x:
                k = c
        r = 99
        for o in range(4):
            if m.getVarByName("r[{},{}]".format(i,o)).x:
                r = ['N','E','S','W'][o]
        print(i, m.getVarByName("x[{}]".format(i)).x, m.getVarByName("y[{}]".format(i)).x,m.getVarByName("z[{}]".format(i)).x,k,r)

def save_items_for_cad(items,containers,m,filename,filepath=None):
    lst = []
    for i in range(len(items['Width'])):

        x = float(m.getVarByName("x[{}]".format(i)).x)
        y = float(m.getVarByName("y[{}]".format(i)).x)
        z = float(m.getVarByName("z[{}]".format(i)).x)
        name = items['Name'][i]
        w = float(items['Width'][i])
        h = float(items['Height'][i])
        d = float(items['Depth'][i])

        k = 99
        cx = 0
        cy = 0
        cz = 0
        for c in range(len(containers['Width'])):
            if m.getVarByName("mvk[{},{}]".format(i,c)).x:
                k = c
                cx = float(containers['x'][c])
                cy = float(containers['y'][c])
                cz = float(containers['z'][c])
        r = 99
        for o in range(4):
            if m.getVarByName("r[{},{}]".format(i,o)).x:
                r = ['N','E','S','W'][o]
        print(x,y,z,w,h,d)
        if r == 'N':
                x = x+cx+w/2
                y = y+cy+h/2
        elif r == 'E':
                x = x+cx+h/2
                y = y+cy+w/2
        elif r == 'S':
                x = x+cx+w/2
                y = y+cy+h/2
        elif r == 'W':
                x = x+cx+h/2
                y = y+cy+w/2

        z = z+cz+d/2

        lst.append([name,x,y,z,r, 'EQUIPMENT'])
    df = pd.DataFrame(lst, columns=['Name', 'x', 'y', 'z', 'Direction','CAD Level'])
    if filepath:
        df.to_csv('{}/{}_cadinput.csv'.format(filepath,filename))
    else:
        df.to_csv('{}_cadinput.csv'.format(filename))