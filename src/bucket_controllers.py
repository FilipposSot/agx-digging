"""
Various controllers for the digging environment
"""

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
from agxPythonModules.utils.numpy_utils import BufferWrapper

def measureBucketState(lockSphere, shovel):

    # Measure Kinematic Quantities of the bucket
    pos_com = shovel.getRigidBody().getCmPosition()
    pos_com = np.array([pos_com[0],pos_com[1],pos_com[2]])

    pos_tip = lockSphere.getPosition()    
    pos_tip = np.array([pos_tip[0],pos_tip[1],pos_tip[2]])

    vel_tip = lockSphere.getVelocity()     
    vel_tip = np.array([vel_tip[0],vel_tip[1],vel_tip[2]])

    acl_tip = lockSphere.getAcceleration()    
    acl_tip = np.array([acl_tip[0],acl_tip[1],acl_tip[2]])

    omega = lockSphere.getAngularVelocity()
    omega = np.array([omega[0],omega[1],omega[2]])
    
    alpha = lockSphere.getAngularAcceleration()
    alpha = np.array([alpha[0],alpha[1],alpha[2]])

    r       = (pos_tip - pos_com)               
    ang_tip = np.arctan2(-r[2], r[0])   

    return pos_tip, vel_tip, acl_tip, ang_tip, omega, alpha 

def measureSoilQuantities(shovel, terrain):
       
    # Measure Forces from soil
    penForce_tot = agx.Vec3()
    sepForce_tot = agx.Vec3()
    deformer_tot = agx.Vec3()
    subForce_tot = agx.Vec3()

    penForce = agx.Vec3()
    penTorque = agx.Vec3()
    
    terrain.getPenetrationForce( shovel, penForce, penTorque )
    penForce_tot = penForce
    sepForce_tot = terrain.getSeparationContactForce(shovel )
    subForce_tot = terrain.getContactForce( shovel )
    deformer_tot = terrain.getDeformationContactForce( shovel )

    soil_force = penForce_tot + sepForce_tot + deformer_tot

    fill = terrain.getDynamicMass(shovel)     
    # fill   = ter.getLastDeadLoadFraction(shov)

    return soil_force, fill 


class ForceDriverPID(agxSDK.StepEventListener):
    
    def __init__(self, app, lockSphere, lock, hinge, prismatic_x, prismatic_z, terrain, shovel, operations, dt_control):
        super(ForceDriverPID, self).__init__()
        self.lockSphere = lockSphere
        self.lock = lock
        self.hinge = hinge

        self.prismatic_x = prismatic_x
        self.prismatic_z = prismatic_z

        self.bucket = shovel.getRigidBody()
        self.shovel = shovel
        self.terrain = terrain
        self.operations = operations
        self.forceLimit = 5e4
        self.dt_control = dt_control

        lock.setEnableComputeForces(True)
        
        self.theta_d = -0.2
        self.ang_d = 1.55
        self.v_d = 0.2

        self.soil_force_last = agx.Vec3(0.,0.,0.)

        self.force  = self.operations[0]
        self.torque = 0.0

        self.t_last_control =  -100.
        self.t_last_setpoint = -100.

        self.integ_e_x = 0.0
        self.integ_e_z = 0.0
        self.integ_e_omega = 0.0
        self.integ_e_ang = 0.0

    def setBodyForce(self, force):
        self.lockSphere.setForce(force)

    def setBodyTorque(self, torque):

        torque = agx.Vec3(0., torque, 0.)
        self.lockSphere.setTorque(torque)

    def post(self,t):
        
        self.soil_force_last = self.terrain.getSeparationContactForce(self.shovel)

    def pre(self, t):

        force  = self.operations[0]
        torque = self.operations[1]
        
        # Measure all the states
        self.pos_tip, self.vel_tip, self.acl_tip, self.ang_tip, self.omega, self.alpha = measureBucketState(self.lockSphere, self.shovel)
 
        mass =  self.terrain.getDynamicMass(self.shovel)
        
        s_nom, s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom = self.soilShapeEvaluator.soil_surf_eval(self.pos_tip[0])

        D = s_nom - self.pos_tip[2]

        if (t-self.t_last_setpoint) > .5:
            
            self.t_last_setpoint = t
            # generate pseudo-random velocity set point
            if D < 0.15:
                # self.theta_d = np.random.uniform(low = -0.45, high = -0.05)
                self.theta_d = np.random.uniform(low = -0.55, high = -0.12)

            elif D >= 0.15 and D < 0.4:
                self.theta_d = np.random.uniform(low = -0.25, high = 0.20)
            elif D >= 0.4:
                self.theta_d = np.random.uniform(low = -0.1 , high = 0.45)
            
            self.v_d = np.random.uniform(low = 0.3, high = 1.0)
            # self.omega_d = np.random.uniform(low = -0.5, high = 0.5)
            self.ang_d = np.clip(self.ang_d + np.random.uniform(low = -0.01, high = 0.01), 1.35, 1.75)
            # self.ang_d = 1.55
            self.v_x_d = np.cos(self.theta_d)*self.v_d
            self.v_z_d = np.sin(self.theta_d)*self.v_d
        
        if (t-self.t_last_control) >= self.dt_control:

            self.t_last_control = t

            # PID - VELOCITY CONTROL
            # calculate errors and error integral

            e_v_x = self.vel_tip[0] - self.v_x_d
            e_v_z = self.vel_tip[2] - self.v_z_d 
            # e_omega = self.omega - self.omega_d

            e_ang = self.ang_tip - self.ang_d 

            self.integ_e_x += e_v_x
            self.integ_e_z += e_v_z
            self.integ_e_ang += e_ang

            pos     = self.bucket.getCmPosition()   
            pos_tip = self.lockSphere.getPosition()   
            r       = pos_tip - pos

            mass = self.bucket.getMassProperties().getMass() - 3.28

            force[0]  = -1000.*(1.*(e_v_x) + .5*self.integ_e_x)
            force[2]  = -1500.*(1.*(e_v_z) + .5*self.integ_e_z) + 10.0*mass
            torque    = -0.8*5000.*(1.*np.sign(e_ang)*e_ang**2   + .01*self.integ_e_ang + 0.20*self.omega[1]) + 10.0*mass*r[0]

            self.force  = force 
            self.torque = torque

            self.force[0] += np.random.uniform(low = -5, high = 5)
            self.force[2] += np.random.uniform(low = -5, high = 5)
            self.torque   += np.random.uniform(low = -5, high = 5)
            
        self.setBodyForce(self.force)
        self.setBodyTorque(self.torque)


class ForceDriverDFL(agxSDK.StepEventListener):
    
    def __init__(self, app, lockSphere, lock, hinge, terrain, shovel, dt_control):
        super(ForceDriverDFL, self).__init__()
        self.lockSphere = lockSphere
        self.lock = lock
        self.hinge = hinge
        self.bucket = shovel.getRigidBody()
        self.shovel = shovel
        self.terrain = terrain
        self.forceLimit = 5e4

        self.dt_control = dt_control
        
        # These are variables which are stored for analysis purposes
        self.sliding_x = 0.0
        self.sliding_z = 0.0
        self.sliding_phi = 0.0

        self.sliding_x_u = 0.0
        self.sliding_z_u = 0.0
        self.sliding_phi_u = 0.0

        self.e_x    = 0.0
        self.e_v_x  = 0.0
        self.e_z    = 0.0
        self.e_v_z  = 0.0
        self.e_phi  = 0.0
        self.e_v_phi = 0.0

        self.x_d    = 0.0
        self.v_x_d  = 0.0
        self.z_d    = 0.0
        self.v_z_d  = 0.0
        self.phi_d  = 0.0
        self.v_phi_d = 0.0

        lock.setEnableComputeForces(True)

        self.v_x_low  =  1.0
        self.v_x_high =  2.0
        self.v_z_low  = -0.3
        self.v_z_high = -0.1

        self.v_x_d = 1.0
        self.v_z_d = 0.0

        self.soil_force_last =  agx.Vec3( 0.0, 0.0, 0.0 )
        
        self.force  = agx.Vec3( 0.0, 0.0, 0.0 )
        self.torque = 0.0
        self.omega_d = 0.0

        self.t_last_control = -100.0
        self.t_last_setpoint = 0.0

        self.integ_e_x = 0.0
        self.integ_e_z = 0.0
        self.integ_e_omega = 0.0
        self.integ_e_ang = 0.0

        self.pid_array = []


    def setBodyForce(self, force):
        self.lockSphere.setForce(self.terrain.getTransform().transformVector(force))

    def setBodyTorque(self, torque):

        torque =  agx.Vec3(0., torque, 0.)
        new_torque = self.terrain.getTransform().transformVector(torque)
        self.lockSphere.setTorque(new_torque)

    # def setRotation(self, omega):
    #     self.hinge.getMotor1D().setEnable(True)
    #     self.hinge.getMotor1D().setSpeed( omega)


    def post(self, t):
                
        self.soil_force_last, _ = measureSoilQuantities(self.shovel, self.terrain)

    def pre(self, t):

        force  = agx.Vec3( 0.0, 0.0, 0.0)
        force_pid  = agx.Vec3( 0.0, 0.0, 0.0)

        torque = 0.0
        torque_pid = 0.0

        # self.pos, self.vel, self.acl, self.angle, self.omega, self.alpha, self.fill = self.measureState()
        self.pos_tip, self.vel_tip, self.acl_tip, self.angle_tip, self.omega, self.alpha = measureBucketState(self.lockSphere, self.shovel)
        self.fill =  self.terrain.getDynamicMass(self.shovel)

        agx_pbhf = self.terrain.getSoilParticleBoundedHeightField()
        hf_grid_bucket = self.terrain.getClosestGridPoint(agx.Vec3(   self.pos_tip[0],    self.pos_tip[1],    self.pos_tip[2]))
        x_hf_bucket,  y_hf_bucket = hf_grid_bucket[0], hf_grid_bucket[1]
        z_surcharge_hf = agx_pbhf.getHeight(x_hf_bucket,  y_hf_bucket)

    
        x = np.array([ self.pos_tip[0], self.pos_tip[2], self.angle_tip,  self.vel_tip[0], self.vel_tip[2], self.omega[1]])
        eta = np.array([self.scaling*self.soil_force_last[0],  self.scaling*self.soil_force_last[2], self.scaling*self.fill, z_surcharge_hf])
        xi =  self.dfl.g_Koop(x, eta, None )
       
        e_x     = self.pos_tip[0] - self.last_x_opt[0]
        e_z     = self.pos_tip[2] - self.last_x_opt[1]
        e_ang   = self.angle_tip  - self.last_x_opt[2]

        e_v_x   = self.vel_tip[0] - self.last_x_opt[3]
        e_v_z   = self.vel_tip[2] - self.last_x_opt[4]
        e_omega = self.omega[1]   - self.last_x_opt[5]

        self.integ_e_x += e_v_x
        self.integ_e_z += e_v_z
        self.integ_e_ang += e_ang

        sliding_x       =  10.*e_x   +  e_v_x 
        sliding_z       =  10.*e_z   +  e_v_z 
        sliding_ang     =  10.*e_ang +  e_omega 

        self.e_x = e_x
        self.e_v_x = e_v_x
        self.e_z = e_z
        self.e_v_z = e_v_z
        self.e_phi = e_ang
        self.e_v_phi = e_omega 

        self.x_d     = self.last_x_opt[0]
        self.v_x_d   = self.last_x_opt[3]
        self.z_d     = self.last_x_opt[1]
        self.v_z_d   = self.last_x_opt[4]
        self.phi_d   = self.last_x_opt[2]
        self.v_phi_d = self.last_x_opt[5]

        
        self.sliding_x      = sliding_x
        self.sliding_z      = sliding_z
        self.sliding_phi    = sliding_ang

        u_sliding_x     = -7000*np.clip(sliding_x,-1,1)
        u_sliding_z     = -7000*np.clip(sliding_z,-1,1)
        u_sliding_ang   = -7000*np.clip(sliding_ang,-1,1)

        self.sliding_x_u      = self.scaling*u_sliding_x 
        self.sliding_z_u      = self.scaling*u_sliding_z 
        self.sliding_phi_u    = self.scaling*u_sliding_ang 

        ####################### MPCC CONTROL #########################
        Ups, x_opt_last,x_opt_whole,u_opt_whole = self.mpcc.control_function(xi, t)
        
        U = self.dfl.h_Koop_inverse(x , eta , None, Ups)

        self.last_x_opt = x_opt_last
        self.x_opt_whole = x_opt_whole
        self.u_opt_whole = u_opt_whole
        
        pos     = self.bucket.getCmPosition()   
        pos_tip = self.lockSphere.getPosition()   
        r       = pos_tip - pos

        mass = self.bucket.getMassProperties().getMass() - 3.28

        force[0] =  u_sliding_x   + U[0]/self.scaling
        force[2] =  u_sliding_z   + U[1]/self.scaling + 10.0*mass
        torque   =  u_sliding_ang + U[2]/self.scaling + 10.0*mass*r[0]

        self.pid_array.append(np.array([ sliding_x, sliding_z, sliding_ang]))

        self.t_last_control = t
        
        self.force  = force
        self.torque = torque

        self.setBodyForce(self.force)
        self.setBodyTorque(self.torque)
        # self.setRotation(self.omega_d)