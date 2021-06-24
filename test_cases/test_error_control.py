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
    T_traj_data = 5.0
    N_traj_data = 1
    
    plot_data = False
    save_data = False
    use_saved_data = True
    
    T_traj_test = 5.0
    N_traj_test = 1

    agx_sim = AgxSimulator(dt_data, dt_control)    
    agx_sim.model_has_surface_shape = True 
    plant = DiggingPlant()

    dfl = DFLSoil(plant, dt_data    = dt_data,
                         dt_control = dt_control)
    setattr(agx_sim, "dfl", dfl)
   
    if use_saved_data:
        t_data, x_data, u_data, s_data, e_data = loadData('data03_11_2021_11_07_17.npz')
        print(t_data.shape)
        print(x_data.shape)
        print(u_data.shape)
        print(s_data.shape)
        print(e_data.shape)
        
    else:
        t_data, x_data, u_data, s_data, e_data, y_data = agx_sim.collectData(T = T_traj_data, N_traj = N_traj_data)
   

    if save_data:
        saveData(t_data, x_data, u_data, s_data, e_data)
    
    if plot_data:
        plotData2(t_data, x_data, u_data, s_data, e_data)
    
    dfl.koop_poly_order = 1

    setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_2)

    agx_sim.dfl.regress_model_Koop_with_surf(x_data,e_data,u_data,s_data)
   
    agx_sim.control_mode = "mpcc"
    agx_sim.Q_mpcc         = 50.*sparse.diags([1.,10.])
    agx_sim.R_mpcc         = 1.*sparse.diags([1., 1., 1., 0.001])
    agx_sim.q_theta_mpcc   = 1.0 

    evaluate_error_dataset_size_control(agx_sim, t_data, x_data, u_data, s_data, e_data)


# Entry point when this script is loaded with python
if agxPython.getContext() is None:
    init = agx.AutoInit()
    main(sys.argv)