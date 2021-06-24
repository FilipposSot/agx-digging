

# Python imports
import math
import numpy as np
from scipy.interpolate import splprep, splrep, splev, splint
import control
import scipy
from datetime import datetime

import matplotlib
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['mathtext.default'] = 'rm'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["legend.loc"] = 'upper left'
import matplotlib.colors as colors
import matplotlib.pyplot as plt



import pickle
import copy
import sys
import os
import argparse
import time
from collections import namedtuple
from agxPythonModules.utils.environment import simulation, root, application

sys.path.append(os.getenv("AGX_DIR") + "/data/python/tutorials")
from tutorial_utils import createHelpText
from agxPythonModules.utils.numpy_utils import BufferWrapper

np.set_printoptions(precision = 5, suppress = True)
np.set_printoptions(edgeitems=30, linewidth=100000)
np.core.arrayprint._line_width = 200

def plotData(t, x, u, s, e, t2=None, x2=None, u2=None, s2=None, e2=None, comparison=False):

    fig, axs = plt.subplots(5,2, figsize=(8,10))
    
    # if len(x.shape)==3:
    #     t = t.reshape(-1,t.shape[-1])
    #     x = x.reshape(-1,x.shape[-1])
    #     u = u.reshape(-1,u.shape[-1])
    #     s = s.reshape(-1,s.shape[-1])
    #     e = e.reshape(-1,e.shape[-1])

    for i in range( x.shape[0] - 1, -1, -1):
        print("i: ", i)

        axs[0,0].plot(t[0,:],x[i,:,0],'r',marker=".")
        axs[0,0].plot(t[0,:],x[i,:,1],'g',marker=".")
        axs[0,0].plot(t[0,:],x[i,:,2],'b',marker=".")
        axs[0,0].set_title("tip position")

        axs[1,0].plot(t[0,:],x[i,:,3],'r',marker=".")
        axs[1,0].plot(t[0,:],x[i,:,4],'g',marker=".")
        axs[1,0].plot(t[0,:],x[i,:,5],'b',marker=".")
        axs[1,0].set_title("tip velocity")

        axs[2,0].plot(t[0,:],e[i,:,0],'r',marker=".")
        axs[2,0].plot(t[0,:],e[i,:,1],'g',marker=".")
        axs[2,0].plot(t[0,:],e[i,:,2],'b',marker=".")
        axs[2,0].set_title("tip acceleration")

        axs[3,0].plot(t[0,:],e[i,:,3],'r', marker = ".")
        axs[3,0].plot(t[0,:],e[i,:,4],'g', marker = ".")
        axs[3,0].set_title("soil force")

        axs[4,0].plot(t[0,:],u[i,:,0],'r', marker = ".")
        axs[4,0].plot(t[0,:],u[i,:,1],'g', marker = ".")
        axs[4,0].plot(t[0,:],u[i,:,2],'b', marker = ".")
        axs[4,0].set_title("bucket force")


        axs[0,1].plot(t[0,:],e[i,:,5],'r')
        axs[0,1].set_title("Bucket Fill")

        # soil shape variables
        axs[2,1].plot(t[0,:],x[i,:,1],'k')
        axs[2,1].plot(t[0,:],s[i,:,0],'r', marker=".")
        axs[2,1].set_title("Soil height")

        axs[3,1].plot(t[0,:],s[i,:,1],'r', marker=".")
        axs[3,1].set_title("Soil gradient")

        if comparison:
            break

        # axs[2,1].plot(t[0,:],s[i,:,2],'r', marker=".")
        # axs[2,1].set_title("Soil Curvature")
    
    if comparison:

        axs[0,0].plot(t[0,:],x2[:,0],'r--')
        axs[0,0].plot(t[0,:],x2[:,1],'g--')
        axs[0,0].plot(t[0,:],x2[:,2],'b--')

        axs[1,0].plot(t[0,:],x2[:,3],'r--')
        axs[1,0].plot(t[0,:],x2[:,4],'g--')
        axs[1,0].plot(t[0,:],x2[:,5],'b--')

        axs[2,0].plot(t[0,:],e2[:,0],'r--')
        axs[2,0].plot(t[0,:],e2[:,1],'g--')
        axs[2,0].plot(t[0,:],e2[:,2],'b--')

        axs[3,0].plot(t[0,:],e2[:,3],'r--')
        axs[3,0].plot(t[0,:],e2[:,4],'g--')

        axs[4,0].plot(t[0,:],u2[:,0],'r--', marker = ".")
        axs[4,0].plot(t[0,:],u2[:,1],'g--', marker = ".")
        axs[4,0].plot(t[0,:],u2[:,2],'b--', marker = ".")

        axs[0,1].plot(t[0,:],e2[:,5],'r--')



    plt.subplots_adjust(left = 0.2, top = 0.89, hspace = 0.4)
    fig.tight_layout()
    plt.show()

def plotData2(k,t, x, u, s, e, t2=None, x2=None, u2=None, s2=None, e2=None, comparison=False):

    fig, axs = plt.subplots(5,2, figsize=(8,10))
    
    # if len(x.shape)==3:
    #     t = t.reshape(-1,t.shape[-1])
    #     x = x.reshape(-1,x.shape[-1])
    #     u = u.reshape(-1,u.shape[-1])
    #     s = s.reshape(-1,s.shape[-1])
    #     e = e.reshape(-1,e.shape[-1])

    for i in range( x.shape[0] - 1, -1, -1):
        print("i: ", i)
        color = 'black'
        axs[0,0].plot(t[0,::3],x[i,::3,0], color = 'black')
        axs[0,0].plot(t[0,::3],x[i,::3,1], color = 'tab:blue')
        axs[0,0].plot(t[0,::3],x[i,::3,2], color = 'tab:orange')
        # axs[0,0].set_title("tip position")

        axs[1,0].plot(t[0,::3],x[i,::3,3], color = 'black')
        axs[1,0].plot(t[0,::3],x[i,::3,4], color = 'tab:blue')
        axs[1,0].plot(t[0,::3],x[i,::3,5], color = 'tab:orange')
        # axs[1,0].set_title("tip velocity")

        axs[2,0].plot(t[0,::3],e[i,::3,0], color = 'black')
        axs[2,0].plot(t[0,::3],e[i,::3,1], color = 'tab:blue')

        axs[2,1].plot(t[0,::3],u[i,::3,0], color = 'black')
        axs[2,1].plot(t[0,::3],u[i,::3,1], color = 'tab:blue')
        axs[2,1].plot(t[0,::3],u[i,::3,2], color = 'tab:orange')
        # axs[3,0].set_title("bucket force")

        axs[1,1].plot(t[0,::3],e[i,::3,2],'k')
        # axs[4,0].set_title("Bucket Fill")

        # soil shape variables
        axs[0,1].plot(t[0,::3],x[i,::3,1],'tab:blue')
        axs[0,1].plot(t[0,::3],s[i,::3,0],'k')
        axs[0,1].plot(t[0,::3],e[i,::3,3],'k-.')
        # axs[2,1].set_title("Soil height")

        # axs[1,1].plot(t[0,:],s[i,:,1],'r', marker=".")
        # axs[3,1].set_title("Soil gradient")

        # axs[4,1].plot(u[i,:,0],u[i,:,2],'.')

        if comparison:
            break

        # axs[2,1].plot(t[0,:],s[i,:,2],'r', marker=".")
        # axs[2,1].set_title("Soil Curvature")
    
    if comparison:

        axs[0,0].plot(t[0,::3],x2[::3,0],'--',color = 'black')
        axs[0,0].plot(t[0,::3],x2[::3,1],'--',color = 'tab:blue')
        axs[0,0].plot(t[0,::3],x2[::3,2],'--',color = 'tab:orange')
        axs[0,0].set_title("tip position")

        axs[1,0].plot(t[0,::3],x2[::3,3], '--', color = 'black')
        axs[1,0].plot(t[0,::3],x2[::3,4], '--',color = 'tab:blue')
        axs[1,0].plot(t[0,::3],x2[::3,5], '--',color = 'tab:orange')
        axs[1,0].set_title("tip velocity")

        axs[2,0].plot(t[0,::3],e2[::3,0], '--', color = 'black')
        axs[2,0].plot(t[0,::3],e2[::3,1], '--',color = 'tab:blue')
        axs[2,0].set_title("soil force")

        axs[2,1].plot(t[0,::3],u2[::3,0],'--', color = 'black')
        axs[2,1].plot(t[0,::3],u2[::3,1],'--',color = 'tab:blue')
        axs[2,1].plot(t[0,::3],u2[::3,2],'--',color = 'tab:orange')
        axs[2,1].set_title("bucket force")

        axs[1,1].plot(t[0,::3],e2[::3,2],'k--')
        axs[1,1].set_title("Bucket Fill")




    plt.subplots_adjust(left = 0.2, top = 0.89, hspace = 0.4)
    fig.tight_layout()
    # plt.show()
    pickle.dump(fig, open(str(k)+'_sample_data_input_transform.fig.pickle', 'wb')) 

def plotData3(t, x, u, s, e, dfl, t2=None, x2=None, u2=None, s2=None, e2=None, comparison=False):

    fig, axs = plt.subplots(5,2, figsize=(8,10))
    
    y = []
    
    x_shape = x.shape

    for j in range(x_shape[0]):
        for i in range(x_shape[1]):
            y.append(dfl.g_Koop(x[j,i,:], e[j,i,:], s[j,i,:]))

    y = np.array(y)

    for i in range( x.shape[0] - 1, -1, -1):
        print("i: ", i)

        axs[0,0].plot(t[0,:],x[i,:,0],'r',marker=".")
        axs[0,0].plot(t[0,:],x[i,:,1],'g',marker=".")
        axs[0,0].plot(t[0,:],x[i,:,2],'b',marker=".")
        axs[0,0].set_title("tip position")

        axs[1,0].plot(t[0,:],x[i,:,3],'r',marker=".")
        axs[1,0].plot(t[0,:],x[i,:,4],'g',marker=".")
        axs[1,0].plot(t[0,:],x[i,:,5],'b',marker=".")
        axs[1,0].set_title("tip velocity")

        axs[2,0].plot(t[0,:],y[:,6],'r', marker = ".")
        axs[2,0].plot(t[0,:],y[:,7],'g', marker = ".")
        axs[2,0].set_title("soil force")

        axs[3,0].plot(t[0,:],u[i,:,0],'r', marker = ".")
        axs[3,0].plot(t[0,:],u[i,:,1],'g', marker = ".")
        axs[3,0].plot(t[0,:],u[i,:,2],'b', marker = ".")
        axs[3,0].set_title("bucket force")

        axs[4,0].plot(t[0,:],y[:,8],'r')
        axs[4,0].set_title("Bucket Fill")

        # soil shape variables
        axs[0,1].plot(t[0,:],y[:,9],'k')
        axs[0,1].plot(t[0,:],y[:,10],'r', marker=".")
        axs[0,1].set_title("f/m")

        # soil shape variables
        axs[1,1].plot(t[0,:],y[:,11],'k')
        axs[1,1].plot(t[0,:],y[:,12],'r', marker=".")
        axs[1,1].set_title("f trig")

        # axs[4,1].plot(u[i,:,0],u[i,:,2],'.')

        if comparison:
            break

        # axs[2,1].plot(t[0,:],s[i,:,2],'r', marker=".")
        # axs[2,1].set_title("Soil Curvature")
    
    if comparison:

        axs[0,0].plot(t[0,:],x2[:,0],'r--')
        axs[0,0].plot(t[0,:],x2[:,1],'g--')
        axs[0,0].plot(t[0,:],x2[:,2],'b--')
        axs[0,0].set_title("tip position")

        axs[1,0].plot(t[0,:],x2[:,3],'r--')
        axs[1,0].plot(t[0,:],x2[:,4],'g--')
        axs[1,0].plot(t[0,:],x2[:,5],'b--')
        axs[1,0].set_title("tip velocity")

        axs[2,0].plot(t[0,:],e2[:,0],'r--')
        axs[2,0].plot(t[0,:],e2[:,1],'g--')
        axs[2,0].set_title("soil force")

        axs[3,0].plot(t[0,:],u2[:,0],'r--')
        axs[3,0].plot(t[0,:],u2[:,1],'g--')
        axs[3,0].plot(t[0,:],u2[:,2],'b--')
        axs[3,0].set_title("bucket force")

        axs[4,0].plot(t[0,:],e2[:,2],'r--')
        axs[4,0].set_title("Bucket Fill")




    plt.subplots_adjust(left = 0.2, top = 0.89, hspace = 0.4)
    fig.tight_layout()
    plt.show()


def saveData(t, x, u, s, e):

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    
    file_name = 'data'+date_time+'.npz'

    np.savez( file_name,t = t,
                        x = x,
                        e = e,
                        s = s,
                        u = u)

def loadData(file_name):

    data = np.load(file_name)
    t = data['t']
    x = data['x']
    u = data['u']
    e = data['e']
    s = data['s']

    return t, x, u, s, e
