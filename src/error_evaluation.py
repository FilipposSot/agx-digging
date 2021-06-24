"""
Helper functions to evaluate models 
"""

# Python imports
import math
import numpy as np
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
from scipy.spatial.distance import cdist

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

def path_error(x_gt, agx_sim):

    theta_array = np.linspace(-10, 0, num=2000)
    for i in range(len(theta_array)):
        x_path, y_path = agx_sim.spl_path.path_eval(theta_array[i], d=0)
        if x_path > x_gt[0,0,0]:
            path_initial = theta_array[i]
            break
    path = []
    theta_array = np.linspace(path_initial, 0, num=2000)
    for i in range(len(theta_array)):
        x_path, y_path = agx_sim.spl_path.path_eval(theta_array[i], d=0)
        path.append(np.array([ x_path, y_path ]))
        if x_path > x_gt[0,-1, 0]:
            path_final = theta_array[i]
            break
    path = np.array(path)

    mean_error = np.mean(np.amin(cdist(path,x_gt[0, :, :2]),axis=0))

    return mean_error

def evaluate_error_dataset_size(t_train, x_train, u_train, s_train, e_train ):

    N_data_array = np.array([50,75,150,200,250,300,400,500,750,1000,1250,1500,2000,5000,10000,20000])
    
    sum_error_dfl_total  = np.zeros((len(N_data_array), 6))

    N_train     = 10
    N_tests     = 15
    N_samples   = 300
    
    for i_test in range(N_tests):
        
        t_test, x_test, u_test, s_test, e_test = agx_sim.collectData(T = 5., N_traj = 1)

        k_horizon = 20

        k_0_array = np.random.randint(low = 0, high = x_test.shape[1]-k_horizon , size = N_samples)

        for i_dataset in range(len(N_data_array)):
            
            N_data = N_data_array[i_dataset]
            
            print(i_test,N_data )

            for i_training in range(N_train):
                
                dfl.koop_poly_order = 1
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta)
                n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train, e_train, u_train, s_train, N=N_data)

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_dfl = np.zeros((k_horizon + 1,n_koop ))
                    y_dfl[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_dfl[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_dfl[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_dfl      =  y_dfl[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    # y_minus_mean_dfl =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_dfl_total[i_dataset,:]         += y_error_dfl**2
                    # sum_normalization_dfl_total[i_dataset,:] += y_minus_mean_dfl**2

    n_total = N_tests*N_train*N_samples

    fig, axs = plt.subplots(3,2, figsize=(8,10))
    axs[0,0].plot(N_data_array ,  sum_error_dfl_total[:,0]/n_total,'k',marker=".")
    axs[0,0].grid(True)
    axs[0,0].set_ylabel(r'$\mathit{MSE_x}$   $(m^2)$')
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlabel('Training Dataset Size')    

    axs[1,0].plot(N_data_array ,  sum_error_dfl_total[:,1]/n_total,'k',marker=".")
    axs[1,0].grid(True)
    axs[1,0].set_ylabel(r'$\mathit{MSE_y}$   $(m^2)$')
    axs[1,0].set_yscale('log')
    axs[1,0].set_xlabel('Training Dataset Size')    

    axs[2,0].plot(N_data_array ,  sum_error_dfl_total[:,2]/n_total,'k',marker=".")
    axs[2,0].grid(True)
    axs[2,0].set_ylabel(r'$\mathit{MSE_\phi}$   $(rad^2)$')
    axs[2,0].set_yscale('log')
    axs[2,0].set_xlabel('Training Dataset Size')    
    
    pickle.dump(fig, open('FigureDataset.fig.pickle', 'wb')) 
    plt.show()

    return y_error_dfl

def evaluate_error_dataset_size_control(agx_sim, t_train, x_train, u_train, s_train, e_train ):

    N_data_array = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000])

    sum_error_total_dfl  = np.zeros(len(N_data_array))
    sum_error_total_koop_x  = np.zeros(len(N_data_array))

    N_train     = 10
    N_tests     = 1
    
    for i_dataset in range(len(N_data_array)):
        
        N_data = N_data_array[i_dataset]
        
        for i_training in range(N_train):

            print(N_data , i_training)
            
            agx_sim.dfl.koop_poly_order = 1
            agx_sim.observable_type = 'dfl'
            setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x_eta_2)
            n_koop = agx_sim.dfl.g_Koop(x_train[0,0,:], e_train[0,0,:], s_train[0,0,:]).shape[0]
            agx_sim.dfl.regress_model_Koop_with_surf(x_train, e_train, u_train, s_train, N = N_data)

            for i_test in range(N_tests):
                t_gt, x_gt, u_gt, s_gt, e_gt , _= agx_sim.collectData(T = 7.5, N_traj = 1)
                error = path_error(x_gt, agx_sim)
                sum_error_total_dfl[i_dataset] += error
            
            agx_sim.dfl.koop_poly_order = 1
            agx_sim.observable_type = 'x'
            setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x)
            n_koop = agx_sim.dfl.g_Koop(x_train[0,0,:], e_train[0,0,:], s_train[0,0,:]).shape[0]
            agx_sim.dfl.regress_model_Koop_with_surf(x_train, e_train, u_train, s_train, N = N_data)
            
            for i_test in range(N_tests):
                t_gt, x_gt, u_gt, s_gt, e_gt , _= agx_sim.collectData(T = 7.5, N_traj = 1)
                error = path_error(x_gt, agx_sim)
                sum_error_total_koop_x[i_dataset] += error


    n_total = N_tests*N_train

    fig, axs = plt.subplots(3,1, figsize=(8,10))
    axs[0].plot(N_data_array ,  sum_error_total_dfl/n_total, color = 'black', marker=".")
    axs[0].plot(N_data_array ,  sum_error_total_koop_x/n_total, color = 'tab:blue', marker=".")
    axs[0].grid(True)
    axs[0].set_ylabel(r'Mean Path Error $(m)$')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Training Dataset Size')    
   
    pickle.dump(fig, open('FigureDataset_control_4.fig.pickle', 'wb')) 
    plt.show()

    return y_error_dfl

def evaluate_error_modified(t_train, x_train, u_train, s_train, e_train ):
    
    k_horizon_array = np.array([1,2,5,10,15,20,30,40,50])

    sum_error_koop_1_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_1_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_2_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_2_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_3_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_3_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_4_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_4_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_5_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_5_total  = np.zeros((len(k_horizon_array), 6))


    N_train = 10
    N_tests = 10
    N_samples = 300


    for i_training in range(N_train):
        print("Train Number: ", i_training)
        train_indices = np.random.choice(range(100), size = 10, replace = False)

        # t_train, x_train, u_train, s_train, e_train = agx_sim.collectData(T = 8.0,
        #                                                                   N_traj = 8)


        # plotData2(t_train, x_train, u_train, s_train, e_train)

        # mean_train = np.concatenate((np.mean(np.mean(x_train[:,:,:3], axis = 1), axis = 0),np.mean(np.mean(e_train[:,:,:3], axis = 1), axis = 0)))
        mean_train = np.mean(np.mean(x_train[:,:,:6], axis = 1), axis = 0)

        # perform testing
        for i_tests in range(N_tests):

            print("Test Number: ", i_tests)

            t_test, x_test, u_test, s_test, e_test = agx_sim.collectData(T = 7.5, N_traj = 1)
            
            # if i_tests == 0:
               
                # # DFL plotting to evaluate model            
                # y_dfl = np.zeros((x_test.shape[1],plant.n))
                # y_dfl[0,:] = np.concatenate((x_test[-1,0,:],e_test[-1,0,:]))
                # for i in range(x_test.shape[1] - 1):
                #     y_dfl[i+1,:] = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[i,:], u_test[-1,i,:])

                # plotData2(t_test, x_test, u_test, s_test, e_test,
                #  t_test, y_dfl[:,: plant.n_x], u_test[-1,:,:], s_test, y_dfl[:,plant.n_x :], comparison = True)

            # # DFL plotting to evaluate model   
            # dfl.koop_poly_order = 1
            # setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_2)
            # n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
            
            # agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])
            # y_koop      = np.zeros((x_test.shape[1],n_koop))
            # y_koop[0,:] = agx_sim.dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:])
            
            # for i in range(x_test.shape[1] - 1):
            #     y_koop[i+1,:] = agx_sim.dfl.f_disc_koop(0.0, y_koop[i,:], u_test[-1,i,:])

            # plotData3(t_test, x_test, u_test, s_test, e_test,
            #  t_test, y_koop[:,: plant.n_x], u_test[-1,:,:], s_test, y_koop[:,plant.n_x :], comparison = True)


            for i_horizon in range(len(k_horizon_array)):
                k_horizon = k_horizon_array[i_horizon]
                k_0_array = np.random.randint(low = 0, high = x_test.shape[1]-k_horizon , size = N_samples)


                ################################################################################################
                dfl.koop_poly_order = 1
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_2)
                n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_1 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_1[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_1[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_1[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_1      =  y_koop_1[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_1 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_1_total[i_horizon,:]         += y_error_koop_1**2
                    sum_normalization_koop_1_total[i_horizon,:] += y_minus_mean_koop_1**2
                
                #################################################################################################

                
                dfl.koop_poly_order = 2
                setattr(dfl, "g_Koop", dfl.g_Koop_x)
                n_koop = dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])
                # simulate koopman 1
                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_2 = np.zeros((k_horizon+1,n_koop ))
                    y_koop_2[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_2[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_2[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_2      = y_koop_2[-1,:6]  - x_test[-1, k_0 + k_horizon,:6]#np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_2 =  mean_train - x_test[-1, k_0 + k_horizon,:6] #np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_2_total[i_horizon,:]         += y_error_koop_2**2
                    sum_normalization_koop_2_total[i_horizon,:] += y_minus_mean_koop_2**2

                #################################################################################################
                dfl.koop_poly_order = 2
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_2)
                n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_3 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_3[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_3[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_3[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_3      =  y_koop_3[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_3 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_3_total[i_horizon,:]         += y_error_koop_3**2
                    sum_normalization_koop_3_total[i_horizon,:] += y_minus_mean_koop_3**2
                
                #################################################################################################

                
                dfl.koop_poly_order = 1
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_3)
                n_koop = dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])
                # simulate koopman 1
                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_4 = np.zeros((k_horizon+1,n_koop ))
                    y_koop_4[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_4[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_4[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_4      = y_koop_4[-1,:6]  - x_test[-1, k_0 + k_horizon,:6]#np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_4 =  mean_train - x_test[-1, k_0 + k_horizon,:6] #np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_4_total[i_horizon,:]         += y_error_koop_4**2
                    sum_normalization_koop_4_total[i_horizon,:] += y_minus_mean_koop_4**2

                #################################################################################################
                dfl.koop_poly_order = 2
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_3)
                n_koop = dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])
                # simulate koopman 1
                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_5 = np.zeros((k_horizon+1,n_koop ))
                    y_koop_5[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_5[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_5[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_5      = y_koop_5[-1,:6]  - x_test[-1, k_0 + k_horizon,:6]#np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_5 =  mean_train - x_test[-1, k_0 + k_horizon,:6] #np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_5_total[i_horizon,:]         += y_error_koop_5**2
                    sum_normalization_koop_5_total[i_horizon,:] += y_minus_mean_koop_5**2

                #################################################################################################
               

    y_koop_1_nmse = np.divide( sum_error_koop_1_total, sum_normalization_koop_1_total )
    y_koop_2_nmse = np.divide( sum_error_koop_2_total, sum_normalization_koop_2_total )

    n_total = N_train*N_tests*N_samples

    if True:
        # dfl, x only + poly2, dfl + poly2, dfl naive, dfl naive poly2
        fig, axs = plt.subplots(3,2, figsize=(8,10))
        axs[0,0].plot( k_horizon_array,  sum_error_koop_1_total[:,0]/n_total,color = 'black', marker=".")
        axs[0,0].plot( k_horizon_array,  sum_error_koop_2_total[:,0]/n_total,color = 'tab:blue', marker=".")
        axs[0,0].plot( k_horizon_array,  sum_error_koop_3_total[:,0]/n_total,color = 'tab:orange', marker=".")
        axs[0,0].plot( k_horizon_array,  sum_error_koop_4_total[:,0]/n_total,color = 'tab:green', marker=".")      
        axs[0,0].plot( k_horizon_array,  sum_error_koop_5_total[:,0]/n_total,color = 'tab:purple', marker=".")
        axs[0,0].grid(True)
        axs[0,0].set_ylabel(r'$\mathit{MSE_x}$   $(m^2)$')
        axs[0,0].set_yscale('log')

        axs[1,0].plot( k_horizon_array,  sum_error_koop_1_total[:,1]/n_total,color = 'black', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_2_total[:,1]/n_total,color = 'tab:blue', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_3_total[:,1]/n_total,color = 'tab:orange', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_4_total[:,1]/n_total,color = 'tab:green', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_5_total[:,1]/n_total,color = 'tab:purple', marker=".")
        axs[1,0].grid(True)
        axs[1,0].set_ylabel(r'$\mathit{MSE_y} $   $(m^2)$')
        axs[1,0].set_yscale('log')

        axs[2,0].plot( k_horizon_array,  sum_error_koop_1_total[:,2]/n_total,color = 'black', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_2_total[:,2]/n_total,color = 'tab:blue', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_3_total[:,2]/n_total,color = 'tab:orange', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_4_total[:,2]/n_total,color = 'tab:green', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_5_total[:,2]/n_total,color = 'tab:purple', marker=".")
        axs[2,0].grid(True)
        axs[2,0].set_ylabel(r'$\mathit{MSE_\phi}$   $(rad^2)$')
        axs[2,0].set_xlabel('Time horizon, (steps)')
        axs[2,0].set_yscale('log')

        axs[0,1].plot( k_horizon_array,  sum_error_koop_1_total[:,3]/n_total,color = 'black', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_2_total[:,3]/n_total,color = 'tab:blue', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_3_total[:,3]/n_total,color = 'tab:orange', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_4_total[:,3]/n_total,color = 'tab:green', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_5_total[:,3]/n_total,color = 'tab:purple', marker=".")
        axs[0,1].grid(True)
        axs[0,1].set_ylabel(r'$\mathit{MSE_{v_x}} $   $(m^2 s^{-2})$')
        axs[0,1].set_yscale('log')

        axs[1,1].plot( k_horizon_array,  sum_error_koop_1_total[:,4]/n_total,color = 'black', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_2_total[:,4]/n_total,color = 'tab:blue', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_3_total[:,4]/n_total,color = 'tab:orange', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_4_total[:,4]/n_total,color = 'tab:green', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_5_total[:,4]/n_total,color = 'tab:purple', marker=".")
        axs[1,1].grid(True)
        axs[1,1].set_ylabel(r'$\mathit{MSE_{v_y}} $   $(m^2 s^{-2})$')
        axs[1,1].set_yscale('log')

        axs[2,1].plot( k_horizon_array,  sum_error_koop_1_total[:,5]/n_total,color = 'black', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_2_total[:,5]/n_total,color = 'tab:blue', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_3_total[:,5]/n_total,color = 'tab:orange', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_4_total[:,5]/n_total,color = 'tab:green', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_5_total[:,5]/n_total,color = 'tab:purple', marker=".")
        axs[2,1].grid(True)
        axs[2,1].set_ylabel(r'$\mathit{MSE_\omega} $   $(rad^2 s^{-2})$')
        axs[2,1].set_xlabel('Time horizon, (steps)')
        axs[2,1].set_yscale('log')

        pickle.dump(fig, open('Figure_error_MSE.fig.pickle', 'wb')) 
        plt.show()


        plt.show()

    return y_error_dfl

def evaluate_error_eta(agx_sim, t_train, x_train, u_train, s_train, e_train ):
    
    k_horizon_array = np.array([1,2,5,10,20,35,50,100])

    sum_error_koop_1_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_1_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_2_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_2_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_3_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_3_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_4_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_4_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_5_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_5_total  = np.zeros((len(k_horizon_array), 6))


    N_train = 5
    N_tests = 5
    N_samples = 100
    N_data = 5000

    for i_training in range(N_train):
        print("Train Number: ", i_training)
        train_indices = np.random.choice(range(10), size = 10, replace = False)

        # t_train, x_train, u_train, s_train, e_train = agx_sim.collectData(T = 8.0,
        #                                                                   N_traj = 8)


        # plotData2(t_train, x_train, u_train, s_train, e_train)

        # mean_train = np.concatenate((np.mean(np.mean(x_train[:,:,:3], axis = 1), axis = 0),np.mean(np.mean(e_train[:,:,:3], axis = 1), axis = 0)))
        mean_train = np.mean(np.mean(x_train[:,:,:6], axis = 1), axis = 0)

        # perform testing
        for i_tests in range(N_tests):

            print("Test Number: ", i_tests)

            t_test, x_test, u_test, s_test, e_test, _ = agx_sim.collectData(T = 5.0, N_traj = 1)
            
            for i_horizon in range(len(k_horizon_array)):
                k_horizon = k_horizon_array[i_horizon]
                k_0_array = np.random.randint(low = 0, high = x_test.shape[1]-k_horizon , size = N_samples)


                ################################################################################################
                agx_sim.dfl.koop_poly_order = 1
                setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x_eta_4)
                setattr(agx_sim.dfl, "h_Koop", agx_sim.dfl.h_Koop_identity) 
                setattr(agx_sim.dfl, "h_Koop_inverse", agx_sim.dfl.h_Koop_identity_inverse) 
                n_koop = agx_sim.dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:], N = N_data)

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_1 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_1[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_1[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_1[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_1      =  y_koop_1[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_1 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_1_total[i_horizon,:]         += y_error_koop_1**2
                    sum_normalization_koop_1_total[i_horizon,:] += y_minus_mean_koop_1**2
                
                #################################################################################################

                
                agx_sim.dfl.koop_poly_order = 1
                setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x)
                setattr(agx_sim.dfl, "h_Koop", agx_sim.dfl.h_Koop_identity) 
                setattr(agx_sim.dfl, "h_Koop_inverse", agx_sim.dfl.h_Koop_identity_inverse) 
                n_koop = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:], N = N_data)
                # simulate koopman 1
                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_2 = np.zeros((k_horizon+1,n_koop ))
                    y_koop_2[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_2[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_2[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_2      = y_koop_2[-1,:6]  - x_test[-1, k_0 + k_horizon,:6]#np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_2 =  mean_train - x_test[-1, k_0 + k_horizon,:6] #np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_2_total[i_horizon,:]         += y_error_koop_2**2
                    sum_normalization_koop_2_total[i_horizon,:] += y_minus_mean_koop_2**2

                #################################################################################################
                agx_sim.dfl.koop_poly_order = 1
                setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x_eta_5)
                setattr(agx_sim.dfl, "h_Koop", agx_sim.dfl.h_Koop_identity) 
                setattr(agx_sim.dfl, "h_Koop_inverse", agx_sim.dfl.h_Koop_identity_inverse) 
                n_koop = agx_sim.dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:], N = N_data)

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_3 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_3[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_3[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_3[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_3      =  y_koop_3[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_3 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_3_total[i_horizon,:]         += y_error_koop_3**2
                    sum_normalization_koop_3_total[i_horizon,:] += y_minus_mean_koop_3**2           
                
                #################################################################################################

                agx_sim.dfl.koop_poly_order = 2
                setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x)
                setattr(agx_sim.dfl, "h_Koop", agx_sim.dfl.h_Koop_identity) 
                setattr(agx_sim.dfl, "h_Koop_inverse", agx_sim.dfl.h_Koop_identity_inverse) 
                n_koop = agx_sim.dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:], N = N_data)

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_4 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_4[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_4[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_4[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_4      =  y_koop_4[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_4 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_4_total[i_horizon,:]         += y_error_koop_4**2
                    sum_normalization_koop_4_total[i_horizon,:] += y_minus_mean_koop_4**2         

    y_koop_1_nmse = np.divide( sum_error_koop_1_total, sum_normalization_koop_1_total )
    y_koop_2_nmse = np.divide( sum_error_koop_2_total, sum_normalization_koop_2_total )

    n_total = N_train*N_tests*N_samples

    if True:
        # dfl, x only + poly2, dfl + poly2, dfl naive, dfl naive poly2
        fig, axs = plt.subplots(3,2, figsize=(8,10))
        axs[0,0].plot( k_horizon_array,  sum_error_koop_1_total[:,0]/n_total,color = 'black', marker=".")
        axs[0,0].plot( k_horizon_array,  sum_error_koop_2_total[:,0]/n_total,color = 'tab:blue', marker=".")
        axs[0,0].plot( k_horizon_array,  sum_error_koop_3_total[:,0]/n_total,color = 'tab:orange', marker=".")
        axs[0,0].plot( k_horizon_array,  sum_error_koop_4_total[:,0]/n_total,color = 'tab:green', marker=".")
        axs[0,0].grid(True)
        axs[0,0].set_ylabel(r'$\mathit{MSE_x}$   $(m^2)$')
        axs[0,0].set_yscale('log')

        axs[1,0].plot( k_horizon_array,  sum_error_koop_1_total[:,1]/n_total,color = 'black', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_2_total[:,1]/n_total,color = 'tab:blue', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_3_total[:,1]/n_total,color = 'tab:orange', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_4_total[:,1]/n_total,color = 'tab:green', marker=".")
        axs[1,0].grid(True)
        axs[1,0].set_ylabel(r'$\mathit{MSE_y} $   $(m^2)$')
        axs[1,0].set_yscale('log')

        axs[2,0].plot( k_horizon_array,  sum_error_koop_1_total[:,2]/n_total,color = 'black', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_2_total[:,2]/n_total,color = 'tab:blue', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_3_total[:,2]/n_total,color = 'tab:orange', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_4_total[:,2]/n_total,color = 'tab:green', marker=".")
        axs[2,0].grid(True)
        axs[2,0].set_ylabel(r'$\mathit{MSE_\phi}$   $(rad^2)$')
        axs[2,0].set_xlabel('Time horizon, (steps)')
        axs[2,0].set_yscale('log')

        axs[0,1].plot( k_horizon_array,  sum_error_koop_1_total[:,3]/n_total,color = 'black', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_2_total[:,3]/n_total,color = 'tab:blue', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_3_total[:,3]/n_total,color = 'tab:orange', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_4_total[:,3]/n_total,color = 'tab:green', marker=".")
        axs[0,1].grid(True)
        axs[0,1].set_ylabel(r'$\mathit{MSE_{v_x}} $   $(m^2 s^{-2})$')
        axs[0,1].set_yscale('log')

        axs[1,1].plot( k_horizon_array,  sum_error_koop_1_total[:,4]/n_total,color = 'black', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_2_total[:,4]/n_total,color = 'tab:blue', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_3_total[:,4]/n_total,color = 'tab:orange', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_4_total[:,4]/n_total,color = 'tab:green', marker=".")
        axs[1,1].grid(True)
        axs[1,1].set_ylabel(r'$\mathit{MSE_{v_y}} $   $(m^2 s^{-2})$')
        axs[1,1].set_yscale('log')

        axs[2,1].plot( k_horizon_array,  sum_error_koop_1_total[:,5]/n_total,color = 'black', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_2_total[:,5]/n_total,color = 'tab:blue', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_3_total[:,5]/n_total,color = 'tab:orange', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_4_total[:,5]/n_total,color = 'tab:green', marker=".")
        axs[2,1].grid(True)
        axs[2,1].set_ylabel(r'$\mathit{MSE_\omega} $   $(rad^2 s^{-2})$')
        axs[2,1].set_xlabel('Time horizon, (steps)')
        axs[2,1].set_yscale('log')

        pickle.dump(fig, open('Figure_error_MSE_eta_types_momentum.fig.pickle', 'wb')) 
        plt.show()

    return None

def evaluate_error_u_transform(agx_sim, t_train, x_train, u_train, s_train, e_train ):
    
    k_horizon_array = np.array([1,2,5,10,20,35,50])

    sum_error_koop_1_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_1_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_2_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_2_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_3_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_3_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_4_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_4_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_5_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_5_total  = np.zeros((len(k_horizon_array), 6))


    N_train = 5
    N_tests = 5
    N_samples = 200
    N_data = 10000

    for i_training in range(N_train):
        print("Train Number: ", i_training)
        train_indices = np.random.choice(range(100), size = 20, replace = False)

        # t_train, x_train, u_train, s_train, e_train = agx_sim.collectData(T = 8.0,
        #                                                                   N_traj = 8)


        # plotData2(t_train, x_train, u_train, s_train, e_train)

        # mean_train = np.concatenate((np.mean(np.mean(x_train[:,:,:3], axis = 1), axis = 0),np.mean(np.mean(e_train[:,:,:3], axis = 1), axis = 0)))
        mean_train = np.mean(np.mean(x_train[:,:,:6], axis = 1), axis = 0)

        # perform testing
        for i_tests in range(N_tests):

            print("Test Number: ", i_tests)

            t_test, x_test, u_test, s_test, e_test, _ = agx_sim.collectData(T = 5.0, N_traj = 1)
            
            for i_horizon in range(len(k_horizon_array)):
                k_horizon = k_horizon_array[i_horizon]
                k_0_array = np.random.randint(low = 0, high = x_test.shape[1]-k_horizon , size = N_samples)


                ################################################################################################

                agx_sim.dfl.koop_poly_order = 1
                setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x_eta_4)
                setattr(agx_sim.dfl,  "h_Koop", agx_sim.dfl.h_Koop_1) 
                setattr(agx_sim.dfl,  "h_Koop_inverse", agx_sim.dfl.h_Koop_1_inverse) 
                n_koop = agx_sim.dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:], N = N_data)

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_1 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_1[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_1[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_1[j,:], agx_sim.dfl.h_Koop(x_test[0,k_0 + j,:], e_test[0,k_0 + j,:], s_test[0,k_0 + j,:], u_test[0,k_0 + j,:]))
                    
                    y_error_koop_1      =  y_koop_1[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_1 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_1_total[i_horizon,:]         += y_error_koop_1**2
                    sum_normalization_koop_1_total[i_horizon,:] += y_minus_mean_koop_1**2
                
                #################################################################################################

                agx_sim.dfl.koop_poly_order = 1
                setattr(agx_sim.dfl, "g_Koop", agx_sim.dfl.g_Koop_x_eta_4)
                setattr(agx_sim.dfl, "h_Koop", agx_sim.dfl.h_Koop_identity) 
                setattr(agx_sim.dfl, "h_Koop_inverse", agx_sim.dfl.h_Koop_identity_inverse) 
                n_koop = agx_sim.dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:], N = N_data)

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_2 = np.zeros((k_horizon+1,n_koop ))
                    y_koop_2[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_2[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_2[j,:], agx_sim.dfl.h_Koop(x_test[0,k_0 + j,:], e_test[0,k_0 + j,:], s_test[0,k_0 + j,:], u_test[0,k_0 + j,:]))
                    
                    y_error_koop_2      = y_koop_2[-1,:6]  - x_test[-1, k_0 + k_horizon,:6]#np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_2 =  mean_train - x_test[-1, k_0 + k_horizon,:6] #np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_2_total[i_horizon,:]         += y_error_koop_2**2
                    sum_normalization_koop_2_total[i_horizon,:] += y_minus_mean_koop_2**2 

    y_koop_1_nmse = np.divide( sum_error_koop_1_total, sum_normalization_koop_1_total )
    y_koop_2_nmse = np.divide( sum_error_koop_2_total, sum_normalization_koop_2_total )

    n_total = N_train*N_tests*N_samples

    if True:
        # dfl, x only + poly2, dfl + poly2, dfl naive, dfl naive poly2
        fig, axs = plt.subplots(3,2, figsize=(8,10))
        axs[0,0].plot( k_horizon_array,  sum_error_koop_1_total[:,0]/n_total,color = 'black', marker=".")
        axs[0,0].plot( k_horizon_array,  sum_error_koop_2_total[:,0]/n_total,color = 'tab:blue', marker=".")
        axs[0,0].grid(True)
        axs[0,0].set_ylabel(r'$\mathit{MSE_x}$   $(m^2)$')
        axs[0,0].set_yscale('log')

        axs[1,0].plot( k_horizon_array,  sum_error_koop_1_total[:,1]/n_total,color = 'black', marker=".")
        axs[1,0].plot( k_horizon_array,  sum_error_koop_2_total[:,1]/n_total,color = 'tab:blue', marker=".")
        axs[1,0].grid(True)
        axs[1,0].set_ylabel(r'$\mathit{MSE_y} $   $(m^2)$')
        axs[1,0].set_yscale('log')

        axs[2,0].plot( k_horizon_array,  sum_error_koop_1_total[:,2]/n_total,color = 'black', marker=".")
        axs[2,0].plot( k_horizon_array,  sum_error_koop_2_total[:,2]/n_total,color = 'tab:blue', marker=".")
        axs[2,0].grid(True)
        axs[2,0].set_ylabel(r'$\mathit{MSE_\phi}$   $(rad^2)$')
        axs[2,0].set_xlabel('Time horizon, (steps)')
        axs[2,0].set_yscale('log')

        axs[0,1].plot( k_horizon_array,  sum_error_koop_1_total[:,3]/n_total,color = 'black', marker=".")
        axs[0,1].plot( k_horizon_array,  sum_error_koop_2_total[:,3]/n_total,color = 'tab:blue', marker=".")
        axs[0,1].grid(True)
        axs[0,1].set_ylabel(r'$\mathit{MSE_{v_x}} $   $(m^2 s^{-2})$')
        axs[0,1].set_yscale('log')

        axs[1,1].plot( k_horizon_array,  sum_error_koop_1_total[:,4]/n_total,color = 'black', marker=".")
        axs[1,1].plot( k_horizon_array,  sum_error_koop_2_total[:,4]/n_total,color = 'tab:blue', marker=".")
        axs[1,1].grid(True)
        axs[1,1].set_ylabel(r'$\mathit{MSE_{v_y}} $   $(m^2 s^{-2})$')
        axs[1,1].set_yscale('log')

        axs[2,1].plot( k_horizon_array,  sum_error_koop_1_total[:,5]/n_total,color = 'black', marker=".")
        axs[2,1].plot( k_horizon_array,  sum_error_koop_2_total[:,5]/n_total,color = 'tab:blue', marker=".")
        axs[2,1].grid(True)
        axs[2,1].set_ylabel(r'$\mathit{MSE_\omega} $   $(rad^2 s^{-2})$')
        axs[2,1].set_xlabel('Time horizon, (steps)')
        axs[2,1].set_yscale('log')

        pickle.dump(fig, open('Figure_error_MSE_eta_input_transform.fig.pickle', 'wb')) 
        plt.show()

    return None

def evaluate_error(t_train, x_train, u_train, s_train, e_train ):
    
    k_horizon_array = np.array([1,2,5,10,15,20,30,40,50])

    sum_error_dfl_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_dfl_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_1_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_1_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_2_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_2_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_3_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_3_total  = np.zeros((len(k_horizon_array), 6))

    sum_error_koop_4_total  = np.zeros((len(k_horizon_array), 6))
    sum_normalization_koop_4_total  = np.zeros((len(k_horizon_array), 6))

    N_train = 10
    N_tests = 10
    N_samples = 300


    for i_training in range(N_train):
        print("Train Number: ", i_training)
        train_indices = np.random.choice(range(100),size = 10, replace = False)

        # t_train, x_train, u_train, s_train, e_train = agx_sim.collectData(T = 8.0,
        #                                                                   N_traj = 8)


        # plotData2(t_train, x_train, u_train, s_train, e_train)

        # mean_train = np.concatenate((np.mean(np.mean(x_train[:,:,:3], axis = 1), axis = 0),np.mean(np.mean(e_train[:,:,:3], axis = 1), axis = 0)))
        mean_train = np.mean(np.mean(x_train[:,:,:6], axis = 1), axis = 0)


        # agx_sim.dfl.regress_model_custom(x_train, e_train, u_train, s_train)    

        # # # DFL plotting to evaluate model            
        # y_dfl = np.zeros((x_train.shape[1],plant.n))
        # y_dfl[0,:] = np.concatenate((x_train[-1,0,:],e_train[-1,0,:]))
        # for i in range(x_train.shape[1] - 1):
        #     y_dfl[i+1,:] = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[i,:], u_train[-1,i,:])
       
        # plotData(t_train, x_train, u_train, s_train, e_train,
        #  t_train, y_dfl[:,: plant.n_x], u_train[-1,:,:], s_train, y_dfl[:,plant.n_x :], comparison = True)
        
        # perform testing
        for i_tests in range(N_tests):

            print("Test Number: ", i_tests)

            t_test, x_test, u_test, s_test, e_test = agx_sim.collectData(T = 7.5, N_traj = 1)
            
            # if i_tests == 0:
               
            #     # # DFL plotting to evaluate model            
            #     # y_dfl = np.zeros((x_test.shape[1],plant.n))
            #     # y_dfl[0,:] = np.concatenate((x_test[-1,0,:],e_test[-1,0,:]))
            #     # for i in range(x_test.shape[1] - 1):
            #     #     y_dfl[i+1,:] = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[i,:], u_test[-1,i,:])

            #     # plotData2(t_test, x_test, u_test, s_test, e_test,
            #     #  t_test, y_dfl[:,: plant.n_x], u_test[-1,:,:], s_test, y_dfl[:,plant.n_x :], comparison = True)

            #     # DFL plotting to evaluate model   
            #     dfl.koop_poly_order = 1
            #     setattr(dfl, "g_Koop", dfl.g_Koop_x_eta)
            #     n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                
            #     agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])
            #     y_koop      = np.zeros((x_test.shape[1],n_koop))
            #     y_koop[0,:] = agx_sim.dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:])
                
            #     for i in range(x_test.shape[1] - 1):
            #         y_koop[i+1,:] = agx_sim.dfl.f_disc_koop(0.0, y_koop[i,:], u_test[-1,i,:])

            #     plotData2(t_test, x_test, u_test, s_test, e_test,
            #      t_test, y_koop[:,: plant.n_x], u_test[-1,:,:], s_test, y_koop[:,plant.n_x :], comparison = True)


            for i_horizon in range(len(k_horizon_array)):
                k_horizon = k_horizon_array[i_horizon]
                k_0_array = np.random.randint(low = 0, high = x_test.shape[1]-k_horizon , size = N_samples)

                # # simulate DFL
                # for i_sample in range(N_samples):
                    
                #     k_0 = k_0_array[i_sample]
                #     y_dfl = np.zeros((k_horizon+1,plant.n))
                #     y_dfl[0,:] = np.concatenate((x_test[0,k_0,:], e_test[0,k_0,:]))

                #     for j in range(k_horizon):
                #         y_dfl[j+1,:]  = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[j,:], u_test[0,k_0 + j,:])

                #     # plt.plot(y_dfl[:,1])
                #     # plt.plot(x_test[-1,k_0:k_0+k_horizon+1,1])
                #     # print(x_test[-1, k_0+k_horizon,3])
                #     # plt.show()

                #     y_error_dfl  = y_dfl[-1,:6] - x_test[-1, k_0 + k_horizon,:6]# - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                #     y_minus_mean_dfl =  mean_train - x_test[-1, k_0 + k_horizon,:6]# - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                #     sum_error_dfl_total[i_horizon,:]         += y_error_dfl**2
                #     sum_normalization_dfl_total[i_horizon,:] += y_minus_mean_dfl**2
                dfl.koop_poly_order = 1
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta)
                n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_dfl = np.zeros((k_horizon + 1,n_koop ))
                    y_dfl[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_dfl[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_dfl[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_dfl      =  y_dfl[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_dfl =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_dfl_total[i_horizon,:]         += y_error_dfl**2
                    sum_normalization_dfl_total[i_horizon,:] += y_minus_mean_dfl**2
                
                #################################################################################################

                
                dfl.koop_poly_order = 3
                setattr(dfl, "g_Koop", dfl.g_Koop_x)
                n_koop = dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])
                # simulate koopman 1
                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_1 = np.zeros((k_horizon+1,n_koop ))
                    y_koop_1[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_1[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_1[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_1      = y_koop_1[-1,:6]  - x_test[-1, k_0 + k_horizon,:6]#np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_1 =  mean_train - x_test[-1, k_0 + k_horizon,:6] #np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_1_total[i_horizon,:]         += y_error_koop_1**2
                    sum_normalization_koop_1_total[i_horizon,:] += y_minus_mean_koop_1**2

                #################################################################################################

                dfl.koop_poly_order = 4
                setattr(dfl, "g_Koop", dfl.g_Koop_x)
                n_koop = dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_2 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_2[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_2[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_2[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_2      =  y_koop_2[-1,:6] - x_test[-1, k_0 + k_horizon,:6]#-np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_2 =  mean_train - x_test[-1, k_0 + k_horizon,:6]     #-  np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_2_total[i_horizon,:]         += y_error_koop_2**2
                    sum_normalization_koop_2_total[i_horizon,:] += y_minus_mean_koop_2**2

                ##################################################################################################
                dfl.koop_poly_order = 2
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta)
                n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_3 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_3[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_3[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_3[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_3      =  y_koop_3[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_3 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_3_total[i_horizon,:]         += y_error_koop_3**2
                    sum_normalization_koop_3_total[i_horizon,:] += y_minus_mean_koop_3**2
                
                ##################################################################################################

                dfl.koop_poly_order = 3 
                setattr(dfl, "g_Koop", dfl.g_Koop_x_eta)
                n_koop = dfl.g_Koop(x_test[0,0,:], e_test[0,0,:], s_test[0,0,:]).shape[0]
                agx_sim.dfl.regress_model_Koop_with_surf(x_train[ train_indices,:,:], e_train[ train_indices,:,:], u_train[ train_indices,:,:], s_train[ train_indices,:,:])

                for i_sample in range(N_samples):
                    
                    k_0 = k_0_array[i_sample]
                    y_koop_4 = np.zeros((k_horizon + 1,n_koop ))
                    y_koop_4[0,:] = agx_sim.dfl.g_Koop(x_test[0,k_0,:], e_test[0,k_0,:], s_test[0,k_0,:])

                    for j in range(k_horizon ):
                        y_koop_4[j+1,:]  = agx_sim.dfl.f_disc_koop(0.0, y_koop_4[j,:], u_test[0,k_0 + j,:])
                    
                    y_error_koop_4      =  y_koop_4[-1,:6] - x_test[-1, k_0 + k_horizon,:6] # - np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    y_minus_mean_koop_4 =  mean_train      - x_test[-1, k_0 + k_horizon,:6] #- np.concatenate((x_test[-1, k_0 + k_horizon,:3],e_test[-1, k_0 + k_horizon,:3]))
                    sum_error_koop_4_total[i_horizon,:]         += y_error_koop_4**2
                    sum_normalization_koop_4_total[i_horizon,:] += y_minus_mean_koop_4**2

    y_dfl_nmse    = np.divide( sum_error_dfl_total, sum_normalization_dfl_total )
    y_koop_1_nmse = np.divide( sum_error_koop_1_total, sum_normalization_koop_1_total )
    y_koop_2_nmse = np.divide( sum_error_koop_2_total, sum_normalization_koop_2_total )
    y_koop_3_nmse = np.divide( sum_error_koop_3_total, sum_normalization_koop_3_total )
    y_koop_4_nmse = np.divide( sum_error_koop_4_total, sum_normalization_koop_4_total )

    n_total = N_tests*N_samples*len(k_horizon_array)
    print(n_total)

    fig, axs = plt.subplots(3,2, figsize=(8,10))
    axs[0,0].plot( k_horizon_array,  sum_error_dfl_total[:,0]/n_total,'k',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_1_total[:,0]/n_total,'r',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_2_total[:,0]/n_total,'g',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_3_total[:,0]/n_total,'b',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_4_total[:,0]/n_total,'m',marker=".")
    axs[0,0].grid(True)
    axs[0,0].set_ylabel(r'$\mathit{MSE_x}$   $(m^2)$')

    axs[1,0].plot( k_horizon_array,  sum_error_dfl_total[:,1]/n_total,'k',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_1_total[:,1]/n_total,'r',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_2_total[:,1]/n_total,'g',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_3_total[:,1]/n_total,'b',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_4_total[:,1]/n_total,'m',marker=".")
    axs[1,0].grid(True)
    axs[1,0].set_ylabel(r'$\mathit{MSE_y} $   $(m^2)$')

    axs[2,0].plot( k_horizon_array,  sum_error_dfl_total[:,2]/n_total,'k',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_1_total[:,2]/n_total,'r',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_2_total[:,2]/n_total,'g',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_3_total[:,2]/n_total,'b',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_4_total[:,2]/n_total,'m',marker=".")
    axs[2,0].grid(True)
    axs[2,0].set_ylabel(r'$\mathit{MSE_\phi}$   $(rad^2)$')
    axs[2,0].set_xlabel('Time horizon, (steps)')

    axs[0,1].plot( k_horizon_array,  sum_error_dfl_total[:,3]/n_total,'k',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_1_total[:,3]/n_total,'r',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_2_total[:,3]/n_total,'g',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_3_total[:,3]/n_total,'b',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_4_total[:,3]/n_total,'m',marker=".")
    axs[0,1].grid(True)
    axs[0,1].set_ylabel(r'$\mathit{MSE_{v_x}} $   $(m^2 s^{-2})$')

    axs[1,1].plot( k_horizon_array,  sum_error_dfl_total[:,4]/n_total,'k',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_1_total[:,4]/n_total,'r',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_2_total[:,4]/n_total,'g',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_3_total[:,4]/n_total,'b',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_4_total[:,4]/n_total,'m',marker=".")
    axs[1,1].grid(True)
    axs[1,1].set_ylabel(r'$\mathit{MSE_{v_y}} $   $(m^2 s^{-2})$')

    axs[2,1].plot( k_horizon_array, sum_error_dfl_total[:,5]/n_total,'k',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_1_total[:,5]/n_total,'r',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_2_total[:,5]/n_total,'g',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_3_total[:,5]/n_total,'b',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_4_total[:,5]/n_total,'m',marker=".")
    axs[2,1].grid(True)
    axs[2,1].set_ylabel(r'$\mathit{MSE_\omega} $   $(rad^2 s^{-2})$')
    axs[2,1].set_xlabel('Time horizon, (steps)')    
    pickle.dump(fig, open('Figure1.fig.pickle', 'wb')) 
    plt.show()



    fig, axs = plt.subplots(3,2, figsize=(8,10))

    axs[0,0].plot( k_horizon_array,  sum_error_dfl_total[:,0]/n_total,'k',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_1_total[:,0]/n_total,'r',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_2_total[:,0]/n_total,'g',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_3_total[:,0]/n_total,'b',marker=".")
    axs[0,0].plot( k_horizon_array,  sum_error_koop_4_total[:,0]/n_total,'m',marker=".")
    axs[0,0].grid(True)
    axs[0,0].set_ylabel(r'$\mathit{MSE_x} $   $(m^2)$')
    axs[0,0].set_yscale('log')

    axs[1,0].plot( k_horizon_array,  sum_error_dfl_total[:,1]/n_total,'k',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_1_total[:,1]/n_total,'r',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_2_total[:,1]/n_total,'g',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_3_total[:,1]/n_total,'b',marker=".")
    axs[1,0].plot( k_horizon_array,  sum_error_koop_4_total[:,1]/n_total,'m',marker=".")
    axs[1,0].grid(True)
    axs[1,0].set_ylabel(r'$\mathit{MSE_y}$   $(m^2)$')
    axs[1,0].set_yscale('log')

    axs[2,0].plot( k_horizon_array,  sum_error_dfl_total[:,2]/n_total,'k',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_1_total[:,2]/n_total,'r',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_2_total[:,2]/n_total,'g',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_3_total[:,2]/n_total,'b',marker=".")
    axs[2,0].plot( k_horizon_array,  sum_error_koop_4_total[:,2]/n_total,'m',marker=".")
    axs[2,0].grid(True)
    axs[2,0].set_ylabel(r'$\mathit{MSE_\phi}$   $(rad^2)$')
    axs[2,0].set_xlabel('Time horizon, (steps)')
    axs[2,0].set_yscale('log')

    axs[0,1].plot( k_horizon_array,  sum_error_dfl_total[:,3]/n_total,'k',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_1_total[:,3]/n_total,'r',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_2_total[:,3]/n_total,'g',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_3_total[:,3]/n_total,'b',marker=".")
    axs[0,1].plot( k_horizon_array,  sum_error_koop_4_total[:,3]/n_total,'m',marker=".")
    axs[0,1].grid(True)
    axs[0,1].set_ylabel(r'$\mathit{MSE_{v_x}}$   $(m^2 s^{-2})$')
    axs[0,1].set_yscale('log')

    axs[1,1].plot( k_horizon_array,  sum_error_dfl_total[:,4]/n_total,'k',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_1_total[:,4]/n_total,'r',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_2_total[:,4]/n_total,'g',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_3_total[:,4]/n_total,'b',marker=".")
    axs[1,1].plot( k_horizon_array,  sum_error_koop_4_total[:,4]/n_total,'m',marker=".")
    axs[1,1].grid(True)
    axs[1,1].set_ylabel(r'$\mathit{MSE_{v_y}} $   $(m^2 s^{-2})$')
    axs[1,1].set_yscale('log')

    axs[2,1].plot( k_horizon_array, sum_error_dfl_total[:,5]/n_total,'k',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_1_total[:,5]/n_total,'r',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_2_total[:,5]/n_total,'g',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_3_total[:,5]/n_total,'b',marker=".")
    axs[2,1].plot( k_horizon_array, sum_error_koop_4_total[:,5]/n_total,'m',marker=".")
    axs[2,1].grid(True)
    axs[2,1].set_ylabel(r'$\mathit{MSE_\omega} $   $(rad^2 s^{-2})$')
    axs[2,1].set_yscale('log')
    axs[2,1].set_xlabel('Time horizon, (steps)')
    pickle.dump(fig, open('Figure2.fig.pickle', 'wb')) 
    plt.show()

    fig, axs = plt.subplots(3,2, figsize=(8,10))

    axs[0,0].plot( k_horizon_array,  y_dfl_nmse[:,0],'k',marker=".")
    axs[0,0].plot( k_horizon_array,  y_koop_1_nmse[:,0],'r',marker=".")
    axs[0,0].plot( k_horizon_array,  y_koop_2_nmse[:,0],'g',marker=".")
    axs[0,0].plot( k_horizon_array,  y_koop_3_nmse[:,0],'b',marker=".")
    axs[0,0].plot( k_horizon_array,  y_koop_4_nmse[:,0],'m',marker=".")
    axs[0,0].set_title("x")

    axs[1,0].plot( k_horizon_array,  y_dfl_nmse[:,1],'k',marker=".")
    axs[1,0].plot( k_horizon_array,  y_koop_1_nmse[:,1],'r',marker=".")
    axs[1,0].plot( k_horizon_array,  y_koop_2_nmse[:,1],'g',marker=".")
    axs[1,0].plot( k_horizon_array,  y_koop_3_nmse[:,1],'b',marker=".")
    axs[1,0].plot( k_horizon_array,  y_koop_4_nmse[:,1],'m',marker=".")
    axs[1,0].set_title("y")

    axs[2,0].plot( k_horizon_array,  y_dfl_nmse[:,2],'k',marker=".")
    axs[2,0].plot( k_horizon_array,  y_koop_1_nmse[:,2],'r',marker=".")
    axs[2,0].plot( k_horizon_array,  y_koop_2_nmse[:,2],'g',marker=".")
    axs[2,0].plot( k_horizon_array,  y_koop_3_nmse[:,2],'b',marker=".")
    axs[2,0].plot( k_horizon_array,  y_koop_4_nmse[:,2],'m',marker=".")
    axs[2,0].set_title("theta")

    axs[0,1].plot( k_horizon_array,  y_dfl_nmse[:,3],'k',marker=".")
    axs[0,1].plot( k_horizon_array,  y_koop_1_nmse[:,3],'r',marker=".")
    axs[0,1].plot( k_horizon_array,  y_koop_2_nmse[:,3],'g',marker=".")
    axs[0,1].plot( k_horizon_array,  y_koop_3_nmse[:,3],'b',marker=".")
    axs[0,1].plot( k_horizon_array,  y_koop_4_nmse[:,3],'m',marker=".")
    axs[0,1].set_title("v_x")

    axs[1,1].plot( k_horizon_array,  y_dfl_nmse[:,4],'k',marker=".")
    axs[1,1].plot( k_horizon_array,  y_koop_1_nmse[:,4],'r',marker=".")
    axs[1,1].plot( k_horizon_array,  y_koop_2_nmse[:,4],'g',marker=".")
    axs[1,1].plot( k_horizon_array,  y_koop_3_nmse[:,4],'b',marker=".")
    axs[1,1].plot( k_horizon_array,  y_koop_4_nmse[:,4],'m',marker=".")
    axs[1,1].set_title("v_y")
    
    axs[2,1].plot( k_horizon_array,  y_dfl_nmse[:,5],'k',marker=".")
    axs[2,1].plot( k_horizon_array,  y_koop_1_nmse[:,5],'r',marker=".")
    axs[2,1].plot( k_horizon_array,  y_koop_2_nmse[:,5],'g',marker=".")
    axs[2,1].plot( k_horizon_array,  y_koop_3_nmse[:,5],'b',marker=".")
    axs[2,1].plot( k_horizon_array,  y_koop_4_nmse[:,5],'m',marker=".")
    axs[2,1].set_title("omega")

    plt.show()

    return y_error_dfl

