# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np

def cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plot_box(ax, w, h, x, y, rotated, edgecolor=None, facecolor=None, label=""):
    """
    Plot a 2-dimensional box.
    Parameters
    ----------
    ax:
    w:
    h:
    x:
    y:
    rotated:
    edgecolor:
    facecolor:
    label:
    Returns
    -------
    ax :
    """

    # This defines a unit square at (0,0)
    verts = np.array([
        [0., 0.], # left, bottom
        [0., 1.], # left, top
        [1., 1.], # right, top
        [1., 0.], # right, bottom
        [0., 0.], # ignored
        ])

    # create path from co-ordinates
    codes = [Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
            ]

    verts = scale(verts, w, h)
    if rotated:
        verts = rotation(verts, 90)
        verts = translation(verts, h, 0)
    
    verts = translation(verts, x, y)

    path = Path(verts, codes)

    if rotated:
        label = '{} rotated'.format(label)
    # plot the path
    patch = patches.PathPatch(path, edgecolor=edgecolor, facecolor=facecolor, lw=2, label=label,alpha=0.3)
    ax.add_patch(patch)


def plot_cube(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): 
        colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): 
        sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6, axis=0), **kwargs)

def plot_line_3d(ax,x1,x2,y1,y2,z1,z2,color, alpha=1):
    print('Plot 3d line...')
    ax.plot([x1,x2], [y1,y2], [z1,z2], 'o--', linewidth=2, color=color, alpha=alpha)

def plot_line_2d(ax,x1,x2,y1,y2,color, alpha=1):
    ax.plot([x1,x2], [y1,y2], 'o--', color=color, alpha=alpha)

def plot_solution_2d(items_dict, connections_dict, containers_dict, m, plot_items=True, plot_connections=True, show=True, legend=True,figsize=(10,10)):

    w = items_dict['Width']
    h = items_dict['Height']

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

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.grid()
    #ax.set_xticks(np.arange(0, sum(W), 1))
    #ax.set_yticks(np.arange(0, sum(H), 1))

    # plot the containers
    for (xk,yk,Wk,Hk) in zip(mx,my,W,H):
        plot_box(ax, Wk, Hk, xk, yk, 0, edgecolor='k',facecolor='grey')


    box_color=iter(cm.rainbow(np.linspace(0,1,no_items)))

    # plot the items, this includes the shift by module
    for i in range(no_items):
        (wi,hi,xi,yi,ri) = (w[i], h[i], 
            m.getVarByName('x[{}]'.format(i)).x,
            m.getVarByName('y[{}]'.format(i)).x,
            round(m.getVarByName('r[{},1]'.format(i)).x) + round(m.getVarByName('r[{},3]'.format(i)).x)
        )
        module_used = [round(m.getVarByName('mvk[{},{}]'.format(i,k)).x) for k in range(no_modules)]

        mx_shift = sum(module_used*np.array(mx))
        my_shift = sum(module_used*np.array(my))

        c=next(box_color)
        plot_box(ax, wi, hi, xi + mx_shift, yi + my_shift, ri, facecolor=c,label='Item {}'.format(i))

    if plot_connections:
        line_color=iter(cm.rainbow(np.linspace(0,1,len(connections))))
        # plot the connections, this includes the shift by module
        for (i,j,xsi,ysi,xsj,ysj,cij) in connections:
            (wi,hi,xi,yi,ri0,ri1,ri2,ri3) = (w[i], h[i], 
                m.getVarByName('x[{}]'.format(i)).x,
                m.getVarByName('y[{}]'.format(i)).x,
                round(m.getVarByName('r[{},0]'.format(i)).x),
                round(m.getVarByName('r[{},1]'.format(i)).x),
                round(m.getVarByName('r[{},2]'.format(i)).x),
                round(m.getVarByName('r[{},3]'.format(i)).x)
            )

            (wj,hj,xj,yj,rj0,rj1,rj2,rj3) = (w[j], h[j], 
                m.getVarByName('x[{}]'.format(j)).x,
                m.getVarByName('y[{}]'.format(j)).x,
                round(m.getVarByName('r[{},0]'.format(j)).x),
                round(m.getVarByName('r[{},1]'.format(j)).x),
                round(m.getVarByName('r[{},2]'.format(j)).x),
                round(m.getVarByName('r[{},3]'.format(j)).x)
            )
            if ri0:
                x1 = xi+wi/2 + xsi
                y1 = yi+hi/2 + ysi
            elif ri1:
                x1 = xi+hi/2 - ysi
                y1 = yi+wi/2 + xsi
            elif ri2:
                x1 = xi+wi/2 - xsi
                y1 = yi+hi/2 - ysi
            elif ri3:
                x1 = xi+hi/2 + ysi
                y1 = yi+wi/2 - xsi

            if rj0:
                x2 = xj+wj/2 + xsj
                y2 = yj+hj/2 + ysj
            elif rj1:
                x2 = xj+hj/2 - ysj
                y2 = yj+wj/2 + xsj
            elif rj2:
                x2 = xj+wj/2 - xsj
                y2 = yj+hj/2 - ysj
            elif rj3:
                x2 = xj+hj/2 + ysj
                y2 = yj+wj/2 - xsj

            module_used_i = [round(m.getVarByName('mvk[{},{}]'.format(i,k)).x) for k in range(no_modules)]
            module_used_j = [round(m.getVarByName('mvk[{},{}]'.format(j,k)).x) for k in range(no_modules)]

            mx_shift_i = sum(module_used_i*np.array(mx))
            my_shift_i = sum(module_used_i*np.array(my))

            mx_shift_j = sum(module_used_j*np.array(mx))
            my_shift_j = sum(module_used_j*np.array(my))

            x1 = x1 + mx_shift_i
            x2 = x2 + mx_shift_j
            y1 = y1 + my_shift_i
            y2 = y2 + my_shift_j
            c=next(line_color)
            plot_line_2d(ax,x1,x2,y1,y1,color='k') 
            plot_line_2d(ax,x2,x2,y1,y2,color='k')
            #plot_line_2d(ax,x1,x1,y1,y2) 
            #plot_line_2d(ax,x1,x2,y2,y2)
        
    if legend:
        plt.legend()
    if show:
        plt.show()

    return fig, ax

def plot_solution_3d(items_dict, connections_dict, containers_dict, m, plot_items=True, plot_connections=True, show=True, legend=True, lims=[1,1,1]):
    w = items_dict['Width']
    h = items_dict['Height']
    d = items_dict['Depth']

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
    mz = containers_dict['z']

    cK = containers_dict['cost']

    no_modules = len(mx)
    no_items = len(w)
    connections = [(i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in zip(s,t,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij)]
    print(connections)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    color=iter(cm.rainbow(np.linspace(0,1,no_items)))
    positions = []
    sizes = []
    
    # plot the containers
    for (xk,yk,zk,Wk,Hk,Dk) in zip(mx,my,mz,W,H,D):
        print(Wk,Hk,Dk)
        positions.append((xk,yk,zk))
        sizes.append((Wk,Hk,Dk))
    print(np.max(positions))
        
    pc = plot_cube(positions,sizes=sizes,facecolor='white',edgecolor="grey")
    pc.set_alpha(0.01)
    ax.add_collection3d(pc)    

    if plot_items:
        colors = []
        positions = []
        sizes = []
        for i in range(no_items):
            (wi,hi,di,xi,yi,zi,ri0,ri1,ri2,ri3) = (w[i], h[i], d[i],
                m.getVarByName('x[{}]'.format(i)).x,
                m.getVarByName('y[{}]'.format(i)).x,
                m.getVarByName('z[{}]'.format(i)).x,
                round(m.getVarByName('r[{},0]'.format(i)).x),
                round(m.getVarByName('r[{},1]'.format(i)).x),
                round(m.getVarByName('r[{},2]'.format(i)).x),
                round(m.getVarByName('r[{},3]'.format(i)).x)
            )
            c=next(color)

            module_used = [round(m.getVarByName('mvk[{},{}]'.format(i,k)).x) for k in range(no_modules)]

            mx_shift = sum(module_used*np.array(mx))
            my_shift = sum(module_used*np.array(my))
            mz_shift = sum(module_used*np.array(mz))

            positions.append((xi + mx_shift,yi + my_shift,zi + mz_shift))
            if ri1 or ri3:
                sizes.append((hi,wi,di))
            else:
                sizes.append((wi,hi,di))
            colors.append(c)

            ax.text(xi + mx_shift,yi + my_shift,zi + mz_shift, str(i))
    
    
    pc = plot_cube(positions,sizes=sizes,colors=colors,edgecolor="k")
    pc.set_alpha(0.075)
    ax.add_collection3d(pc)    

    line_color=iter(cm.jet(np.linspace(0,1,len(connections))))

    if plot_connections:    
        print('Plotting Connections...')
        print(connections)
        for (i,j,st_id,xsi,ysi,zsi,xsj,ysj,zsj,cij) in connections:
            print('Connection {},{}'.format(i,j))
            (wi,hi,di,xi,yi,zi,ri0,ri1,ri2,ri3) = (w[i], h[i], d[i],
                m.getVarByName('x[{}]'.format(i)).x,
                m.getVarByName('y[{}]'.format(i)).x,
                m.getVarByName('z[{}]'.format(i)).x,
                round(m.getVarByName('r[{},0]'.format(i)).x),
                round(m.getVarByName('r[{},1]'.format(i)).x),
                round(m.getVarByName('r[{},2]'.format(i)).x),
                round(m.getVarByName('r[{},3]'.format(i)).x)
            )
            (wj,hj,dj,xj,yj,zj,rj0,rj1,rj2,rj3) = (w[j], h[j], d[j],
                m.getVarByName('x[{}]'.format(j)).x,
                m.getVarByName('y[{}]'.format(j)).x,
                m.getVarByName('z[{}]'.format(j)).x,
                round(m.getVarByName('r[{},0]'.format(j)).x),
                round(m.getVarByName('r[{},1]'.format(j)).x),
                round(m.getVarByName('r[{},2]'.format(j)).x),
                round(m.getVarByName('r[{},3]'.format(j)).x)
            ) 
            if ri0:
                x1 = xi+wi/2 + xsi
                y1 = yi+hi/2 + ysi
            elif ri1:
                x1 = xi+hi/2 - ysi
                y1 = yi+wi/2 + xsi
            elif ri2:
                x1 = xi+wi/2 - xsi
                y1 = yi+hi/2 - ysi
            elif ri3:
                x1 = xi+hi/2 + ysi
                y1 = yi+wi/2 - xsi

            if rj0:
                x2 = xj+wj/2 + xsj
                y2 = yj+hj/2 + ysj
            elif rj1:
                x2 = xj+hj/2 - ysj
                y2 = yj+wj/2 + xsj
            elif rj2:
                x2 = xj+wj/2 - xsj
                y2 = yj+hj/2 - ysj
            elif rj3:
                x2 = xj+hj/2 + ysj
                y2 = yj+wj/2 - xsj
            
            z1 = zi + di/2 + zsi
            z2 = zj + dj/2 + zsj

            module_used_i = [round(m.getVarByName('mvk[{},{}]'.format(i,k)).x) for k in range(no_modules)]
            module_used_j = [round(m.getVarByName('mvk[{},{}]'.format(j,k)).x) for k in range(no_modules)]

            mx_shift_i = sum(module_used_i*np.array(mx))
            my_shift_i = sum(module_used_i*np.array(my))
            mz_shift_i = sum(module_used_i*np.array(mz))

            mx_shift_j = sum(module_used_j*np.array(mx))
            my_shift_j = sum(module_used_j*np.array(my))
            mz_shift_j = sum(module_used_j*np.array(mz))

            x1 = x1 + mx_shift_i
            x2 = x2 + mx_shift_j
            y1 = y1 + my_shift_i
            y2 = y2 + my_shift_j
            z1 = z1 + mz_shift_i
            z2 = z2 + mz_shift_j
            c=next(line_color)
            plot_line_3d(ax,x1,x2,y1,y1,z1,z1,color='k') 
            plot_line_3d(ax,x2,x2,y1,y2,z1,z1,color='k')
            plot_line_3d(ax,x2,x2,y2,y2,z1,z2,color='k')

    
    ax.set_xlim([0,lims[0]])
    ax.set_ylim([0,lims[1]])
    ax.set_zlim([0,lims[2]])
    #ADD CODE TO AUTOSCALE AXES
    
    # Hide grid lines
    ax.grid(False)

    if legend:
        plt.legend()
    if show:
        plt.show()

    return fig, ax


def rotation(verts, theta_d):
    theta = theta_d*2*np.pi/360
    T = np.array([[np.cos(theta),-np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]])

    return np.dot(T, verts.T).T

def scale(verts, a, d):
    T = np.array([[a,0], 
                  [0,d]])
    return np.dot(T, verts.T).T

def translation(verts, a, b):
    T = np.array([a,b])
    return verts + T

