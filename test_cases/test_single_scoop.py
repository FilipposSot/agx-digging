# AGX Dynamics imports
import agx
import agxCollide
import agxPython
import agxSDK
import agxOSG
import agxIO
import agxUtil
import agxTerrain
import agxRender

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

from dfl.dfl.dfl_soil_agx import *
from dfl.dfl.mpcc import *
from dfl.dfl.dynamic_system import *

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

np.set_printoptions(precision = 5, suppress = True)
np.set_printoptions(edgeitems=30, linewidth=100000)
np.core.arrayprint._line_width = 200

# Import other necessary python classes and methods
from agx_simulation import *
from error_evaluation import *
from manage_data import *

class DiggingPlant():
    
    def __init__(self):

        # Linear part of states matrices
        self.n_x    = 6
        self.n_eta  = 7
        self.n_u    = 3

        self.n      = self.n_x + self.n_eta

        # # # User defined matrices for DFL
        self.A_cont_x  = np.array([[ 0., 0., 0., 1., 0., 0.],
                                   [ 0., 0., 0., 0., 1., 0.],
                                   [ 0., 0., 0., 0., 0., 1.],
                                   [ 0., 0., 0., 0., 0., 0.],
                                   [ 0., 0., 0., 0., 0., 0.],
                                   [ 0., 0., 0., 0., 0., 0.]])

        self.A_cont_eta = np.array([[ 0., 0., 0., 0., 0., 0.],
                                    [ 0., 0., 0., 0., 0., 0.],
                                    [ 0., 0., 0., 0., 0., 0.],
                                    [ 1., 0., 0., 0., 0., 0.],
                                    [ 0., 1., 0., 0., 0., 0.],
                                    [ 0., 0., 1., 0., 0., 0.]])

        self.B_cont_x = np.array([[ 0., 0., 0.],
                                  [ 0., 0., 0.],
                                  [ 0., 0., 0.],
                                  [ 0., 0., 0.],
                                  [ 0., 0., 0.],
                                  [ 0., 0., 0.]])

    def set_soil_surf(self, x, y):

        self.tck_sigma = splrep(x, y, s = 0)

    def soil_surf_eval(self, x):
        # Evaluate the spline soil surface and its derivatives
        
        surf     = splev(x, self.tck_sigma, der = 0, ext = 3)
        surf_d   = splev(x, self.tck_sigma, der = 1, ext = 3)
        surf_dd  = splev(x, self.tck_sigma, der = 2, ext = 3)
        surf_ddd = splev(x, self.tck_sigma, der = 3, ext = 3)

        return surf, surf_d, surf_dd, surf_ddd

    def draw_soil(self, ax, x_min, x_max):
        # for now lets keep this a very simple path: a circle
        x = np.linspace(x_min,x_max, 200)
        y = np.zeros(x.shape)
        
        for i in range(len(x)):
            y[i],_,_,_ = self.soil_surf_eval(x[i])
        
        ax.plot(x, y, 'k--')

        return ax

def main(args):

    dt_control = 0.01
    dt_data = 0.01
    T_traj_data = 5.5
    N_traj_data = 2
    
    plot_data = False
    save_data = False
    use_saved_data = False
    
    T_traj_test = 1.0
    N_traj_test = 1

    agx_sim = AgxSimulator(dt_data, dt_control)    
    agx_sim.model_has_surface_shape = True 
    plant = DiggingPlant()

    dfl = DFLSoil(plant, dt_data    = dt_data,
                         dt_control = dt_control)
    
    setattr(agx_sim, "dfl", dfl)
   
    if use_saved_data:
        t_data, x_data, u_data, s_data, e_data = loadData('data03_18_2021_12_35_38.npz')
    else:
        t_data, x_data, u_data, s_data, e_data, y_data = agx_sim.collectData(T = T_traj_data, N_traj = N_traj_data)

     
    fig, axs = plt.subplots(1,1, figsize=(8,10))

    axs.plot(t_data[0,::3],np.abs(np.divide(e_data[0,::3,0] + u_data[0,::3,0],u_data[0,::3,0])))#,marker=".")
    axs.plot(t_data[0,::3],np.abs(np.divide(e_data[0,::3,1] + u_data[0,::3,1],u_data[0,::3,1])))#,marker=".")
    # axs[1].plot(t_data[0,:],np.abs(np.divide(e_data[0,:,2]+u_data[0,:,2],u_data[0,:,2])),'g',marker=".")
    axs.legend([r'$\mathit{x}-$',r'$\mathit{y}$'],loc='upper right')
    # axs.set_title("surcharge height")
    axs.set_ylabel(r'Force Ratio')
    axs.set_xlabel(r'Time $(s)$')
    plt.grid(True)
    plt.show()
    exit()

    if save_data:
        saveData(t_data, x_data, u_data, s_data, e_data)
    
    if plot_data:
        plotData2(t_data, x_data, u_data, s_data, e_data)
   
    dfl.koop_poly_order = 1

    setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_4)
    setattr(agx_sim.dfl, "h_Koop", agx_sim.dfl.h_Koop_identity) 
    setattr(agx_sim.dfl, "h_Koop_inverse", agx_sim.dfl.h_Koop_identity_inverse) 
                 
    agx_sim.observable_type = 'dfl'

    agx_sim.dfl.regress_model_Koop_with_surf(x_data, e_data, u_data, s_data, N=10000)

    fig, axs = plt.subplots(1, 1)
    w,v = np.linalg.eig(sp.linalg.logm(agx_sim.dfl.K_x)/dt_data)
    print(abs(w))
    print(v)
    axs.plot(np.real(w), np.imag(w),'.',marker = 'o')
    plt.show()
   
    # exit()

    agx_sim.control_mode = "mpcc"

    agx_sim.Q_mpcc         = 50.*sparse.diags([1.,10.])
    agx_sim.R_mpcc         = 1.*sparse.diags([1., 1., 1., 0.001])
    agx_sim.q_theta_mpcc   = 3.0 

    t_gt_mpc_1, x_gt_mpc_1, u_gt_mpc_1, s_gt_mpc_1, e_gt_mpc_1, y_gt_mpc_1= agx_sim.collectData(T = 9.5, N_traj = 1)

    print(y_gt_mpc_1.shape)

    plotData2(t_gt_mpc_1, x_gt_mpc_1, u_gt_mpc_1, s_gt_mpc_1, e_gt_mpc_1)

    fig = plt.figure(figsize=[6.4, 2.8])
    ax = fig.add_subplot(1, 1, 1)    
    
    ax.plot(x_gt_mpc_1[0, :, 0], x_gt_mpc_1[0, :, 1], color = 'mediumblue')

    ax = agx_sim.mpcc.draw_soil(ax,-5, 5)
    ax = agx_sim.mpcc.draw_path(ax, -10, -5)

    ax.quiver(x_gt_mpc_1[0, ::10, 0],
              x_gt_mpc_1[0, ::10, 1],
          -0.5*np.cos(x_gt_mpc_1[0, ::10, 2]),
           0.5*np.sin(x_gt_mpc_1[0, ::10, 2]), units = 'xy',headwidth = 0.0, 
           headlength = 0.0, scale = 2.2,width = 0.005,color = 'mediumblue')

    ax.set_ylabel(r'$\mathit{y}$ $(m)$')
    ax.set_xlabel(r'$\mathit{x}$ $(m)$')

    ax.axis('equal')
    plt.tight_layout()
    ax.set_xlim(-4.5,0.0)  
    ax.set_ylim(np.amin(x_gt_mpc_1[0, :, 1])-0.5, np.amax(x_gt_mpc_1[0, :, 1]) + 0.5)
    plt.show()
    
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    
    pickle.dump(fig, open('Figure_mpcc'+date_time+'.fig.pickle', 'wb')) 

    # exit()

    print(t_gt_mpc_1.shape)
    print(u_gt_mpc_1.shape)
    print(y_gt_mpc_1.shape)

    # fig = plt.figure(figsize=[6.4, 2.8])
    # ax = fig.add_subplot(1, 1, 1)    

    # ax.plot(t_gt_mpc_1[0,:], u_gt_mpc_1[0, :, 0], color = 'mediumblue')
    
    # for i in range(y_gt_mpc_1.shape[1]-70):
    #     if i%30 ==0:
    #         ax.plot(t_gt_mpc_1[0,i:i+50], y_gt_mpc_1[0, i, :, 0], color = 'lightsteelblue')

    # plt.show()

    # fig = plt.figure(figsize=[6.4, 2.8])
    # ax = fig.add_subplot(1, 1, 1)    

    # ax.plot(t_gt_mpc_1[0,:], u_gt_mpc_1[0, :, 1], color = 'mediumblue')
    
    # for i in range(y_gt_mpc_1.shape[1]-70):
    #     if i%30 ==0:
    #         ax.plot(t_gt_mpc_1[0,i:i+50], y_gt_mpc_1[0, i, :, 1], color = 'lightsteelblue')

    # plt.show()

    # pickle.dump(fig, open('Figure_mpcc_optimization.fig.pickle', 'wb')) 



    fig, axs = plt.subplots(3,1, figsize=(8,10))
    axs[0].plot(t_gt_mpc_1[0,::3],y_gt_mpc_1[0,::3,3], marker=".",color='black')
    axs[0].plot(t_gt_mpc_1[0,::3],y_gt_mpc_1[0,::3,4], marker=".",color='tab:blue')
    axs[0].plot(t_gt_mpc_1[0,::3],y_gt_mpc_1[0,::3,5], marker=".",color='tab:orange')
    axs[0].legend([r'$\sigma_x$',r'$\sigma_y$',r'$\sigma_\phi$'],loc='upper right')
    axs[0].set_ylabel(r'Sliding variable')
    axs[0].set_xlabel(r'Time $(s)$')
    
    axs[1].plot(y_gt_mpc_1[0,:,0],y_gt_mpc_1[0,:,1],'k.',color='black')
    m, b = np.polyfit(y_gt_mpc_1[0,:,0], y_gt_mpc_1[0,:,1], 1)
    axs[1].plot(np.array([0,np.amax(y_gt_mpc_1[0,:,0])]),
                np.array([0,m*np.amax(y_gt_mpc_1[0,:,0])+b]),'k--',color='black')

    axs[1].set_xlabel(r'Total Aggregate Mass (kg)')
    axs[1].set_ylabel(r'$\mathit{V_{soil}}$ $(m^3)$')

    axs[2].plot(x_gt_mpc_1[0,:,2],x_gt_mpc_1[0,:,5], 'b')

    axs[2].set_title("surcharge height")

    # pickle.dump(fig, open('FigureSliding.fig.pickle', 'wb')) 

    plt.show()

    exit()


    fig = plt.figure(figsize=[6.4, 2.8])
    ax = fig.add_subplot(1, 1, 1)    
    
    # ax.plot(np.array([-8,5]),np.array([0,0]), color = 'black')

    ax.plot(x_gt_mpc_1[0, :, 0], x_gt_mpc_1[0, :, 1], color = 'lightsteelblue')



    ax = agx_sim.mpcc.draw_soil(ax,-5, 5)
    ax = agx_sim.mpcc.draw_path(ax, -10, -5)

    ax.set_ylabel(r'$\mathit{y}$ $(m)$')
    ax.set_xlabel(r'$\mathit{x}$ $(m)$')

    ax.axis('equal')
    plt.tight_layout()
    ax.set_xlim(-4.5,0.0)  
    ax.set_ylim(np.amin(x_gt_mpc_1[0, :, 1])-0.5, np.amax(x_gt_mpc_1[0, :, 1]) + 0.5)
    plt.show()


# Entry point when this script is loaded with python
if agxPython.getContext() is None:
    init = agx.AutoInit()
    main(sys.argv)