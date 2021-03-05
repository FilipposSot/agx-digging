"""
Name: Trenching tutorial for agxTerrain
Description:

agxTerrain is a library consisting of classes that implement a deformable terrain model based a symmetric
3D grid data structure with an overlapping surface heightfield that can be deformed by interacting Shovel
objects performing digging motions, converting solid mass to dynamic mass which can be moved.

This script demonstrates how to
  - Setup agxTerrain for trenching scenarios
  - Achieving vertical side walls ( creating sharp ditches ) during excavation by adjusting ground
    compaction and angle of repose compaction scaling
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
from agxPythonModules.utils.numpy_utils import BufferWrapper

np.set_printoptions(precision = 5, suppress = True)
np.set_printoptions(edgeitems=30, linewidth=100000)
np.core.arrayprint._line_width = 200

# Defualt shovel settings
default = {
    'length': 0.6,
    'width': 0.6,
    'height': 0.45,
    'thickness': 0.02,
    'topFraction': 0.25
}

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
 
    return K, X, eigVals

class SoilSurfaceEvaluator():
    def __init__(self, x, z):
        self.set_soil_surf(x,z)

    def set_soil_surf(self, x, z):
        self.tck_sigma = splrep(x, z, s = 0)

    def soil_surf_eval(self, x):
        # Evaluate the spline soil surface and its derivatives
        surf     = splev(x, self.tck_sigma, der = 0, ext=3)
        surf_d   = splev(x, self.tck_sigma, der = 1, ext=3)
        surf_dd  = splev(x, self.tck_sigma, der = 2, ext=3)
        surf_ddd = splev(x, self.tck_sigma, der = 3, ext=3)

        return surf, surf_d, surf_dd, surf_ddd

    def draw_soil(self, ax, x_min, x_max):
        # for now lets keep this a very simple path: a circle
        x = np.linspace(x_min,x_max, 200)
        y = np.zeros(x.shape)
        
        for i in range(len(x)):
            y[i],_,_,_ = self.soil_surf_eval(x[i])
        
        ax.plot(x, y, 'k--')

        return ax

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

class AgxSimulator():

    def __init__(self, dt_data, dt_control):
        self.control_mode = "data_collection"
        self.dt_control = dt_control
        self.dt_data = dt_data

        self.scaling = .001
        
        self.set_height_from_previous = False
        self.consecutive_scoop_i = 0



    # def setupCamera(self,  app):
    #     cameraData                   = app.getCameraData()
    #     cameraData.eye               = agx.Vec3( 1.7190326542940962E+01, -1.4658716770523059E-01, 1.2635000298378865E+01 )
    #     cameraData.center            = agx.Vec3( 3.4621315146672371E-01, -1.2032941390018395E-01, -4.7443399198018110E-01 )
    #     cameraData.up                = agx.Vec3( -6.1418716626367531E-01, 2.8103832342862844E-04, 7.8916034227174481E-01 )
    #     cameraData.nearClippingPlane = 0.1
    #     cameraData.farClippingPlane  = 5000
    #     app.applyCameraData( cameraData )

    def setupCamera(self,  app):    
        cameraData                   = app.getCameraData()
        # cameraData.eye               = agx.Vec3( 15.  , -15. , 13. )
        # cameraData.center            = agx.Vec3( 0.   ,  0.  , 0. )
        # cameraData.up                = agx.Vec3( -0.6 ,  0.5 , 0.8 )
        # cameraData.nearClippingPlane = 0.1
        # cameraData.farClippingPlane  = 5000

        cameraData.eye               = agx.Vec3( -3.  , -10. , 0.4)
        cameraData.center            = agx.Vec3( -3.   ,  0.  , 0. )
        cameraData.up                = agx.Vec3( 0.0, 0.0, 0.0)
        cameraData.nearClippingPlane = 10
        cameraData.farClippingPlane  = 5000
        app.applyCameraData( cameraData )

    def setHeightField(self, agx_heightField, np_heightField):

        hf_size = agx_heightField.getSize()
        # print(hf_size[0])
        # Set the height field heights
        for i in range(0, agx_heightField.getResolutionX()):
            for j in range(0, agx_heightField.getResolutionY()):
                agx_heightField.setHeight(i, j, np_heightField[i,j])

        return agx_heightField

    def createBucket(self, spec):
        length = spec['length'] if 'length' in spec else default['length']
        width = spec['width'] if 'width' in spec else default['width']
        height = spec['height'] if 'height' in spec else default['height']
        thickness = spec['thickness'] if 'thickness' in spec else default['thickness']
        topFraction = spec['topFraction'] if 'topFraction' in spec else default['topFraction']

        def createSlopedSides(length, height, thickness, topFraction):
            fracLength = (1.0 - topFraction) * length
            points = [agx.Vec3(0.0, 0.0, -thickness),
                    agx.Vec3(-2.0 * fracLength, 0.0, -thickness),
                    agx.Vec3(0.0, 2.0 * height, -thickness),
                    agx.Vec3(0.0, 0.0, thickness),
                    agx.Vec3(-2.0 * fracLength, 0.0, thickness),
                    agx.Vec3(0.0, 2.0 * height, thickness)]
            vec3Vector = agx.Vec3Vector()
            for point in points:
                vec3Vector.append(point)

            side = agxUtil.createConvexFromVerticesOnly(vec3Vector)

            return side

        def createCuttingEdge(length, width, height, thickness, topFraction):
            fracLength = (1.0 - topFraction) * length
            tanSlope = fracLength / height
            edgeLength = tanSlope * thickness

            points = [agx.Vec3(0.0, -width, 0.0),
                    agx.Vec3(2.0 * edgeLength, -width, 0.0),
                    agx.Vec3(2.0 * edgeLength, -width, thickness * 2.0),
                    agx.Vec3(0.0, width, 0.0),
                    agx.Vec3(2.0 * edgeLength, width, 0.0),
                    agx.Vec3(2.0 * edgeLength, width, thickness * 2.0)]

            vec3Vector = agx.Vec3Vector()
            for point in points:
                vec3Vector.append(point)

            edge = agxUtil.createConvexFromVerticesOnly(vec3Vector)

            return edge

        fracLength = (1.0 - topFraction) * length
        tanSlope = fracLength / height
        edgeLength = tanSlope * thickness

        bucket = agx.RigidBody()

        bottomPlate = agxCollide.Geometry(agxCollide.Box(length - edgeLength, width, thickness))
        leftSide = agxCollide.Geometry(agxCollide.Box(length * topFraction, height, thickness))
        rightSide = agxCollide.Geometry(agxCollide.Box(length * topFraction, height, thickness))
        backPlate = agxCollide.Geometry(agxCollide.Box(height, width + thickness, thickness))
        topPlate = agxCollide.Geometry(agxCollide.Box(length * topFraction, width, thickness))
        leftSlopedSide = agxCollide.Geometry(createSlopedSides(length, height, thickness, topFraction))
        rightSlopedSide = agxCollide.Geometry(createSlopedSides(length, height, thickness, topFraction))
        edge = agxCollide.Geometry(createCuttingEdge(length, width, height, thickness, topFraction))

        bucket.add(bottomPlate)
        bucket.add(leftSide)
        bucket.add(rightSide)
        bucket.add(backPlate)
        bucket.add(topPlate)
        bucket.add(leftSlopedSide)
        bucket.add(rightSlopedSide)
        bucket.add(edge)

        bottomPlate.setLocalPosition(edgeLength, 0.0, -height + thickness)

        leftSide.setLocalRotation(agx.Quat(agx.PI_2, agx.Vec3.X_AXIS()))
        leftSide.setLocalPosition(length * (1.0 - topFraction), width, 0.0)

        rightSide.setLocalRotation(agx.Quat(agx.PI_2, agx.Vec3.X_AXIS()))
        rightSide.setLocalPosition(length * (1.0 - topFraction), -width, 0.0)

        backPlate.setLocalRotation(agx.Quat(agx.PI_2, agx.Vec3.Y_AXIS()))
        backPlate.setLocalPosition(length + thickness, 0.0, 0.0)

        topPlate.setLocalPosition(length * (1.0 - topFraction), 0.0, height - thickness)

        leftSlopedSide.setLocalRotation(agx.Quat(agx.PI_2, agx.Vec3.X_AXIS()))
        leftSlopedSide.setLocalPosition(length * (1.0 - topFraction * 2.0), width, -height)

        rightSlopedSide.setLocalRotation(agx.Quat(agx.PI_2, agx.Vec3.X_AXIS()))
        rightSlopedSide.setLocalPosition(length * (1.0 - topFraction * 2.0), -width, -height)

        edge.setLocalPosition(-length, 0.0, -height)

        cuttingEdge = agx.Line(agx.Vec3(-length, width, -height + thickness), agx.Vec3(-length, -width, -height + thickness))
        topEdge = agx.Line(agx.Vec3((1.0 - topFraction * 2.0), width, height), agx.Vec3((1.0 - topFraction * 2.0), -width, height))
        forwardVector = agx.Vec3(-1.0, 0.0, 0.0)

        return cuttingEdge, topEdge, forwardVector, bucket

    def createRandomHeightfield(self, n_x, n_y, r, random_heaps = 7):

        np_HeightField = np.zeros((n_x, n_y))
        
        x = np.linspace(0.0, r*(n_x-1), n_x)
        y = np.linspace(0.0, r*(n_y-1), n_y)
        X, Y = np.meshgrid(x, y)

        # add/remove some random heaps
        for i in range(random_heaps):

            heap_height = np.random.uniform(0.3,1.0,1)
            heap_sigma  = np.random.uniform(2.0,3.0,1)

            x_c = np.random.uniform(low = 0.0, high = n_x*r, size = 1)
            y_c = np.random.uniform(low = 0.0, high = n_y*r, size = 1)

            surf_heap_i = heap_height*np.exp(-(np.square(X-x_c) + np.square(Y-y_c))/heap_sigma**2)
            np_HeightField = np_HeightField + -np.sign(np.random.uniform(-3,3,1))*surf_heap_i

        # np_HeightField = np_HeightField + np.random.uniform(low = -noise_level, high = noise_level, size = surface.shape)

        return np_HeightField

    def createRandomHeightfield(self, n_x, n_y, r, random_heaps = 5):

        np_HeightField = np.zeros((n_x, n_y))
        
        x = np.linspace(0.0, r*(n_x-1), n_x)
        y = np.linspace(0.0, r*(n_y-1), n_y)
        X, Y = np.meshgrid(x, y)
        heights = [1.0,-1.0,1.5,-1.5,0.5]
        sigmas  = [2.5,2.5,2.5,2.5,2.5]
        x_c      = [6.0,6.0,6.0,6.0,6.0]
        y_c      = [3.0,5.0,8.0,10.0,11.0]

        # add/remove some random heaps
        for i in range(random_heaps):

            surf_heap_i = heights[i]*np.exp(-(np.square(X-x_c[i]) + np.square(Y-y_c[i]))/sigmas[i]**2)
            np_HeightField = np_HeightField + surf_heap_i

        # np_HeightField = np_HeightField + np.random.uniform(low = -noise_level, high = noise_level, size = surface.shape)

        return np_HeightField

    def createFixedHeightfield(self, n_x, n_y, r, random_heaps = 5):

        np_HeightField = np.zeros((n_x, n_y))
        
        x = np.linspace(0.0, r*(n_x-1), n_x)
        y = np.linspace(0.0, r*(n_y-1), n_y)
        X, Y = np.meshgrid(x, y)

        heap_height = 1.0# 1.5 #0.5
        heap_sigma = 3.5

        x_c =  0.5*n_x*r
        y_c =  0.5*n_y*r

        surf_heap_i = heap_height*np.exp(-(np.square(X-x_c) + np.square(Y-y_c))/heap_sigma**2)
        # np_HeightField = np_HeightField + -1*surf_heap_i
        np_HeightField = np_HeightField #- 0.2*Y #0.2*np.sign(Y-3)
        # np_HeightField = np_HeightField + np.random.uniform(low = -noise_level, high = noise_level, size = surface.shape)

        return np_HeightField

    def extractSoilSurface(self, ter, sim):

        hf = ter.getHeightField()
        pos = sim.getRigidBodies()[0].getPosition()
        hf_grid_bucket = ter.getClosestGridPoint(pos)
        x_hf_bucket,  y_hf_bucket = hf_grid_bucket[0], hf_grid_bucket[1]
                  
        x_hf = np.arange(0,ter.getResolutionX())*ter.getElementSize() + ter.getSurfacePositionWorld(agx.Vec2i(0,0))[0]
        z_hf = np.zeros(x_hf.shape)
        
        for hf_x_index in range(ter.getResolutionX()):
            z_hf[hf_x_index] = hf.getHeight(hf_x_index, y_hf_bucket)

        return x_hf, z_hf

    def createSimulation(self, no_graphics = True):
        
        ap = argparse.ArgumentParser()

        # the --noApp will run this without a graphics window
        ap.add_argument('--noApp', action='store_true')
        args1, args2 = ap.parse_known_args()
        args1 = vars(args1)

        if no_graphics:
            app = None
        else:
            # Creates an Example Application
            app = agxOSG.ExampleApplication()
            app.init(agxIO.ArgumentParser([sys.executable] + args2))

        # Create a simulation
        sim = agxSDK.Simulation()

        return sim, app

    def setScene(self, sim, app):
        
        # Start by cleaning up the simulation from any previous content
        sim.cleanup()

        # Next build a scene with some content
        ter, shov, driver, locksphere = self.buildTheScene(app, sim)

        if app:
            # Initialize the simulation in the application
            app.initSimulation(sim, True)

        return sim, app, ter, shov, driver, locksphere

    def buildTheScene(self, app, sim):
        '''
        With this function we will actually create our content
        It will be called from the loop below.
        '''
        scene = 1
        root = None
        # Try getting ExampleApplication
        if app:
            root = app.getSceneRoot()
            app.getSceneDecorator().setText(0, "Scene: {}".format(scene))

        ter, shov, driver, locksphere = self.buildScene1(app, sim, root)

        if app:
            self.setupCamera(app)

        return ter, shov, driver, locksphere

    def buildScene1(self, app, sim, root):

        # Create the Terrain
        num_cells_x = 80
        num_cells_y = 80
        cell_size   = 0.15
        max_depth   = 1.0

        # terrain = agxTerrain.Terrain( num_cells_x, num_cells_y, cell_size, max_depth )
        # sim.add( terrain )

        agx_heightField = agxCollide.HeightField(num_cells_x, num_cells_y, (num_cells_x-1)*cell_size, (num_cells_y-1)*cell_size)
        
        # Dummy numpy height field
        if self.control_mode == "data_collection":
            np_heightField = self.createFixedHeightfield(num_cells_x, num_cells_y, cell_size)
        elif self.control_mode == "mpcc" or self.control_mode == "trajectory_control":
            np_heightField = self.createFixedHeightfield(num_cells_x, num_cells_y, cell_size)


        if self.set_height_from_previous:
            agx_heightField = self.agx_heightField_previous
        else:
            agx_heightField = self.setHeightField(agx_heightField,np_heightField)


        terrain = agxTerrain.Terrain.createFromHeightField(agx_heightField, 5.0)
        sim.add(terrain)
        
        G = agx.Vec3(0, 0, -10.0)
        sim.setUniformGravity(G)
              
        terrain.loadLibraryMaterial("sand_1")
        # terrain.loadMaterialFile("sand_2.json")

        terrainMaterial = terrain.getTerrainMaterial()
        terrainMaterial.getBulkProperties().setSwellFactor( 1.00 )
        
        compactionProperties = terrainMaterial.getCompactionProperties()
        compactionProperties.setAngleOfReposeCompactionRate( 500.0 )

        terrain.setCompaction( 1.05 )
        
        # The trenching will reach the bounds of the terrain so we simply remove particles out of bounds
        # to get rid of the mateiral in a practical way.
        terrain.getProperties().setDeleteSoilParticlesOutsideBounds( True )

        if app:
            # Setup a renderer for the terrain
            renderer = agxOSG.TerrainVoxelRenderer( terrain, root )

            renderer.setRenderHeightField( True )
            # We choose to render the compaction of the soil to visually denote excavated
            # soil from compacted ground
            # renderer.setRenderCompaction( True, agx.RangeReal( 1.0, 1.05 ) )
            renderer.setRenderHeights(True, agx.RangeReal(-0.4,0.1))
            # renderer.setRenderHeights(True, agx.RangeReal(-0.5,0.5))
            renderer.setRenderVoxelSolidMass(False)
            renderer.setRenderVoxelFluidMass(True)
            renderer.setRenderNodes(False)
            renderer.setRenderVoxelBoundingBox(False)
            renderer.setRenderSoilParticlesMesh(True)

            sim.add( renderer )

        # Set contact materials of the terrain and shovel
        # This contact material governs the resistance that the shovel will feel when digging into the terrain
        # [ Shovel - Terrain ] contact material
        shovelMaterial = agx.Material( "shovel_material" )

        terrainMaterial = terrain.getMaterial( agxTerrain.Terrain.MaterialType_TERRAIN )
        shovelTerrainContactMaterial = agx.ContactMaterial( shovelMaterial, terrainMaterial )
        shovelTerrainContactMaterial.setYoungsModulus( 1e8 )
        shovelTerrainContactMaterial.setRestitution( 0.0 )
        shovelTerrainContactMaterial.setFrictionCoefficient( 0.4 )
        sim.add( shovelTerrainContactMaterial )

        # Create the trenching shovel body creation, do setup in the Terrain object and
        # constrain it to a kinematic that will drive the motion

        # Create the bucket rigid body
        cuttingEdge, topEdge, forwardVector, bucket = self.createBucket( default )

        sim.add( bucket )

        # Create the Shovel object using the previously defined cutting and top edge
        shovel = agxTerrain.Shovel( bucket, topEdge, cuttingEdge, forwardVector )
        agxUtil.setBodyMaterial( bucket, shovelMaterial )

        # Set a margin around the bounding box of the shovel where particles are not to be merged
        shovel.setNoMergeExtensionDistance( 0.1 )

        # Add the shovel to the terrain
        terrain.add( shovel )
        
        if app and self.consecutive_scoop_i < 4:
            # Create visual representation of the shovel
            node = agxOSG.createVisual(bucket, root)
            agxOSG.setDiffuseColor(node, agxRender.Color.Gold())
            agxOSG.setAlpha(node, 1.0)

        # Set initial bucket rotation
        if self.control_mode == "mpcc":
            angle_bucket_initial = -0.3*np.pi
        else:
            angle_bucket_initial = np.random.uniform(low = -0.25*np.pi, high = -0.35*np.pi)
            # angle_bucket_initial = -0.3*np.pi


        bucket.setRotation(agx.EulerAngles(0.0, angle_bucket_initial, agx.PI))

        # Get the offset of the bucket tip from the COM
        tip_offset = shovel.getCuttingEdgeWorld().p2
        
        # Set initial bucket rotation
        if self.control_mode == "mpcc" or self.control_mode == "trajectory_control":
            
            if self.consecutive_scoop_i == 0 :
                x_initial_tip = -4.0
            elif self.consecutive_scoop_i == 1:
                x_initial_tip = -3.6
            elif self.consecutive_scoop_i == 2:
                x_initial_tip = -3.3
            else:
                x_initial_tip = -2.6
        else:
            x_initial_tip = np.random.uniform(low = -4.5, high = -3.0)

        # find the soil height at the initial penetration location
        hf_grid_initial = terrain.getClosestGridPoint(agx.Vec3(x_initial_tip, 0.0, 0.0))
        height_initial = terrain.getHeight(hf_grid_initial) - 0.05
        
        # Set the initial bucket location such that it is just contacting the soil
        position = agx.Vec3(x_initial_tip-tip_offset[0], 0, height_initial-tip_offset[2]) 
        bucket.setPosition( terrain.getTransform().transformPoint( position ) )
        
        bucket.setVelocity(0.0, 0.0, 0.0)
        # bucket.setAngularVelocity(0.0, 0.05, 0.0)

        # Add a lockjoint between a kinematic sphere and the shovel
        # in order to have some compliance when moving the shovel
        # through the terrain
        offset = agx.Vec3( 0.0, 0.0, 0.0 )

        ## ADD ALL THE JOINTS TO CONTROL THE BUCKET (x,z,theta)
        sphere1 = agx.RigidBody(agxCollide.Geometry( agxCollide.Sphere( .1 )))
        sphere2 = agx.RigidBody(agxCollide.Geometry( agxCollide.Sphere( .1 )))
        sphere3 = agx.RigidBody(agxCollide.Geometry( agxCollide.Sphere( .1 )))

        sphere1.setMotionControl( agx.RigidBody.DYNAMICS )
        sphere2.setMotionControl( agx.RigidBody.DYNAMICS )
        sphere3.setMotionControl( agx.RigidBody.DYNAMICS )

        sphere1.getGeometries()[ 0 ].setEnableCollisions( False )
        sphere2.getGeometries()[ 0 ].setEnableCollisions( False )
        sphere3.getGeometries()[ 0 ].setEnableCollisions( False )

        # sphere1.setPosition( bucket.getCmFrame().transformPointToWorld( offset ))
        # sphere2.setPosition( bucket.getCmFrame().transformPointToWorld( offset ))
        # sphere3.setPosition( bucket.getCmFrame().transformPointToWorld( offset ))

        tip_position    = shovel.getCuttingEdgeWorld().p2 
        tip_position[1] = bucket.getCmPosition()[1] 

        sphere1.setPosition( tip_position )
        sphere2.setPosition( tip_position )
        sphere3.setPosition( tip_position )

        sphere1.getMassProperties().setMass(0.000001)
        sphere2.getMassProperties().setMass(0.000001)
        sphere3.getMassProperties().setMass(0.000001)

        # print('sphere mass: ', sphere1.getMassProperties().getMass())

        sim.add(sphere1)
        sim.add(sphere2)
        sim.add(sphere3)

        # if app:
        #     node1 = agxOSG.createVisual( sphere1, root)
        #     node2 = agxOSG.createVisual( sphere2, root)
        #     node3 = agxOSG.createVisual( sphere3, root)

        #     agxOSG.setDiffuseColor( node1, agxRender.Color.Red() )
        #     agxOSG.setDiffuseColor( node2, agxRender.Color.Green() )
        #     agxOSG.setDiffuseColor( node3, agxRender.Color.Blue() )

        #     agxOSG.setAlpha( node1 , 0.5 )
        #     agxOSG.setAlpha( node2 , 0.5 )
        #     agxOSG.setAlpha( node3 , 0.5 ) 

        # Set prismatic joint for x transalation world - sphere 1
        f1 = agx.Frame()
        f1.setLocalRotate(agx.EulerAngles(0, math.radians(90), 0))
        prismatic1 = agx.Prismatic(sphere1, f1)

        # Set prismatic joint for z transalation world - sphere 2
        f1 = agx.Frame()
        f2 = agx.Frame()
        f1.setLocalRotate(agx.EulerAngles(0, math.radians(180), 0))
        f2.setLocalRotate(agx.EulerAngles(0, math.radians(180), 0))
        prismatic2 = agx.Prismatic(sphere1, f1, sphere2, f2)

        # # Set hinge joint for rotation of the bucket
        f1 = agx.Frame()
        f1.setLocalRotate(agx.EulerAngles(-math.radians(90),0, 0))
        f2 = agx.Frame()
        f2.setLocalRotate(agx.EulerAngles(-math.radians(90),0, 0))
        hinge2 = agx.Hinge(sphere2, f1, sphere3, f2)

        # hinge2.getMotor1D().setEnable(True)
        # prismatic1.getMotor1D().setEnable(True)
        # prismatic2.getMotor1D().setEnable(True)

        # hinge2.getRange1D().setEnable( True );
        # hinge2.getRange1D().setRange( agx.RangeReal(-0.2, 0.2))
        
        sim.add(prismatic1)
        sim.add(prismatic2)
        sim.add(hinge2)

        lock = agx.LockJoint( sphere3, bucket )
        sim.add( lock )


        
        # lock2 = agx.LockJoint( sphere1, sphere3 )
        # sim.add(lock2)
        # Uncomment to lock rotations
        # sim.add(agx.LockJoint(sphere2,bucket))

        # constant force and torque
        operations = [agx.Vec3( 0.0, 0.0, 0.0), agx.Vec3( 0.0, 0.0, 0.0 )]

        # Extract soil shape along the bucket excavation direction
        x_hf,z_hf = self.extractSoilSurface(terrain, sim)          
        self.soilShapeEvaluator = SoilSurfaceEvaluator(x_hf, z_hf)
        setattr(self.dfl,"soilShapeEvaluator", self.soilShapeEvaluator)

        if self.control_mode == "data_collection":

            # create driver and add it to the simulation
            driver = ForceDriverPID(app,
                                    sphere3,
                                    lock,
                                    hinge2,
                                    prismatic1,
                                    prismatic2,
                                    terrain,
                                    shovel,
                                    operations,
                                    self.dt_control)
            
            # Add the current surface evaluator to the controller
            setattr(driver, "soilShapeEvaluator", self.soilShapeEvaluator)

            # Add the controller to the simulation
            sim.add(driver)
        
        elif self.control_mode == "trajectory_control":
            # create driver and add it to the simulation
            # create driver and add it to the simulation
            driver = ForceDriverTrajectory(app,
                                    sphere3,
                                    lock,
                                    hinge2,
                                    prismatic1,
                                    prismatic2,
                                    terrain,
                                    shovel,
                                    operations,
                                    self.dt_control)
            
            # Add the current surface evaluator to the controller
            setattr(driver, "soilShapeEvaluator", self.soilShapeEvaluator)
            setattr(driver, "dfl",self.dfl)

            x_path = x_initial_tip + np.array([0., 0.5, 1.5, 2.0, 2.5, 
                                                  3.0, 3.5])
            y_soil,_,_,_ = self.soilShapeEvaluator.soil_surf_eval(x_path)
            y_path = y_soil + np.array([ -0.07,  -0.25, -0.25, -0.25, -0.25,
                                             -0.25, -0.02])
            spl_path = spline_path(x_path,y_path)
            setattr(driver, "path_eval", spl_path.path_eval)

            # Add the controller to the simulation
            sim.add(driver)

        elif self.control_mode == "mpcc":
            # create driver and add it to the simulation
            driver = ForceDriverDFL(app,
                                    sphere3,
                                    lock,
                                    hinge2,
                                    terrain,
                                    shovel,
                                    self.dt_control)

            ################ MPCC CONTROLLER ############################
            # Add the current surface evaluator to the controller
            setattr(driver, "soilShapeEvaluator", self.soilShapeEvaluator)
            setattr(driver, "dfl", self.dfl)
            setattr(driver, "scaling",self.scaling)

            # x_array.append(np.array([pos[0], pos[2]]))
            # eta_array.append(np.array([vel[0], vel[2], soil_force[0], soil_force[2], fill_2]))
            
            # define the path to be followed by MPCC
            # x_path = x_initial_tip + np.array([0., 0.5, 1.5, 2.0, 2.5, 
            #                                   3.0, 3.5, 4.0, 4.5, 5.0,
            #                                   5.5, 6.0, 6.5, 7.0, 7.5,
            #                                   8.0, 8.5, 9.0, 9.5, 10.0])
            # y_soil,_,_,_ = self.soilShapeEvaluator.soil_surf_eval(x_path)
            # y_path = y_soil + np.array([ -0.07,  -0.45, -0.45, -0.45, -0.45,
            #                              -0.45,  -0.45, -0.45, -0.45, -0.45,
            #                              -0.45,  -0.45, -0.45, -0.45, -0.45,
            #                              -0.45,  -0.45, -0.45, -0.45, -0.45])
            x_path = x_initial_tip + np.array([0., 0.5, 1.5, 2.0, 2.5, 3.0, 3.5])

            y_soil,_,_,_ = self.soilShapeEvaluator.soil_surf_eval(x_path)
            y_path = y_soil + np.array([ -0.07, -0.35, -0.35, -0.35, -0.35, -0.25, -0.02])

            # x_min = np.array([ x_initial_tip-0.1 ,  -3. , 0.5 , -0.5 , -2.5 , -2.5 , -400, -400 , -400, -70000*self.scaling, -70000*self.scaling, 0.0])
            # x_max = np.array([ 2.                , 5.0  , 2.5 ,  2.5 ,  2.5 ,  2.5 ,  400,  400 ,  400,  70000*self.scaling,  70000*self.scaling, 3000.*self.scaling])
            
            x_min = np.array([ x_initial_tip-0.1 ,  -3. , 0.5 , -0.5 , -2.5 , -2.5 , -80000*self.scaling, -80000*self.scaling, 0.0               , -70000*self.scaling, -70000*self.scaling,-70000*self.scaling, -70000*self.scaling])
            x_max = np.array([ 2.                , 5.0  , 2.5 ,  2.5 ,  2.5 ,  2.5 ,  80000*self.scaling,  80000*self.scaling, 3000.*self.scaling,  70000*self.scaling,  70000*self.scaling, 70000*self.scaling,  70000*self.scaling])
            # x_min = np.array([ x_initial_tip-0.1 ,  -2. , 0.5 , -0.5 , -2.5 , -2.5 ,  -70000*self.scaling, -70000*self.scaling, 0.0])
            # x_max = np.array([ 2.                , 2.0  , 2.5 ,  2.5 ,  2.5 ,  2.5 ,   70000*self.scaling,  70000*self.scaling, 3000.*self.scaling])
            
            # x_min = np.concatenate((np.array([ x_initial_tip-0.1 ,  -2. , 0.5 , -0.5 , -2.5 , -2.5 ,  -70000*self.scaling, -70000*self.scaling, 0.0]),-10000000*np.ones(45)))
            # x_max = np.concatenate((np.array([ 2.                , 0.0  , 2.5 ,  2.5 ,  2.5 ,  2.5 ,   70000*self.scaling,  70000*self.scaling, 3000.*self.scaling]),10000000*np.ones(45)))
            
            u_min = np.array([ -100.*self.scaling   , -70000*self.scaling ,  -70000*self.scaling ])
            u_max = np.array([  15000.*self.scaling,   70000*self.scaling ,   70000*self.scaling ])

            if self.set_height_from_previous: 
                pass
            else:
                self.spl_path = spline_path(x_path,y_path)


            # instantiate the MPCC object
            mpcc = MPCC(np.zeros((self.dfl.plant.n, self.dfl.plant.n)),
                        np.zeros((self.dfl.plant.n, self.dfl.plant.n_u)),
                        x_min, x_max,
                        u_min, u_max,
                        dt = self.dt_data, N = 50)
            
            # # # # instantiate the MPCC object
            # mpcc = MPCC(np.zeros((54, 54)),
            #             np.zeros((54, self.dfl.plant.n_u)),
            #             x_min, x_max,
            #             u_min, u_max,
            #             dt = self.dt_data, N = 30)

            #  set the observation function, path object and linearization function
            setattr(mpcc, "path_eval",  self.spl_path .path_eval)
            setattr(mpcc, "get_soil_surface", self.soilShapeEvaluator.soil_surf_eval) 

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax = mpcc.draw_path(ax, -10, -5)
            ax = mpcc.draw_soil(ax, x_initial_tip -1, x_initial_tip + 5)
            ax.axis('equal')
            plt.show()

            if self.model_has_surface_shape:
                print('Linearizing with soil shape')
                # setattr(mpcc, "get_linearized_model", self.dfl.linearize_soil_dynamics)
                setattr(mpcc, "get_linearized_model", self.dfl.linearize_soil_dynamics_koop)
            else:
                setattr(mpcc, "get_linearized_model", self.dfl.linearize_soil_dynamics_no_surface)

            self.mpcc = copy.copy(mpcc)

            # define the MPCC cost matrices and coef.
            Q = 50.*sparse.diags([1.,10.])
            R = 1.*sparse.diags([1., 1., 1., 0.001])
            q_theta = 5.0 #self.q_theta_mpcc
            pos_tip, vel_tip, acl_tip, ang_tip, omega, alpha = measureBucketState(sphere3, shovel)
            

            x_i = np.array([ pos_tip[0], pos_tip[2], ang_tip,  vel_tip[0], vel_tip[2], omega[1]])
            eta_i = np.array([acl_tip[0], acl_tip[2], alpha[1], 0.,  0., 0.])
            path_initial = [-10,-9.5,-9.0,-8.5,-8.5]
            x_0 =  np.concatenate((self.dfl.g_Koop(x_i,eta_i,_), np.array([path_initial[self.consecutive_scoop_i]])))

            # x_0 = np.concatenate((np.array([pos_tip[0], pos_tip[2], ang_tip ,
            #                                 vel_tip[0], vel_tip[2], omega[1],
            #                                 acl_tip[0], acl_tip[2], alpha[1],
            #                                 0.        ,  0.       ,     0.]), np.array([-10.0])))
            driver.last_x_opt = x_0
            # x_0 = np.concatenate((np.array([pos_tip[0], pos_tip[2], ang_tip ,
            #                                 vel_tip[0], vel_tip[2], omega[1],
            #                                 0.        ,  0.       ,     0.]), np.array([-10.0])))

            # x_0 = np.concatenate((self.dfl.g_Koop(np.array([pos_tip[0], pos_tip[2], ang_tip]),
            #                                     np.array([ vel_tip[0], vel_tip[2], omega[1],
            #                                                  0.        ,  0.       ,     0.]),_), np.array([-10.0])))
            # x_0 = x_0 - np.concatenate((self.x_offset,self.e_offset,np.array([0.0])))
            print("x_0:", x_0)
            print("x_min:", x_min)
            print("x_max:", x_max)

            # set initial input (since input cost is differential)
            u_minus = np.array([0.0, 0.0, 0.0, 0.0])

            # sets up the new mpcc problem
            mpcc.setup_new_problem(Q, R, q_theta, x_0, u_minus)

            setattr(driver, "mpcc", mpcc)
            #####################################################################################
            # Add the controller to the simulation
            sim.add(driver)
        
        # Limit core usage to number of physical cores. Assume that HT/SMT is active
        # and divide max threads with 2.
        agx.setNumThreads( 0 )
        n = int(agx.getNumThreads() / 2 - 1)
        agx.setNumThreads( n )
        
        # Setup initial camera view
        if app:
            createHelpText(sim, app)

        return terrain, shovel, driver, sphere3
   
    def collectData(self, T = 5, N_traj = 3):
            
        # This is our simulation loop where we will
        # Step simulation, update graphics(if we are using an application window) 
        # We will step the simulation for 5 seconds

        T_data   = []
        X_data   = []
        U_data   = []
        S_data   = []
        Eta_data = []

        Y_data= []

        if self.control_mode == "mpcc":
            no_graphics = False
        else:
            no_graphics = False

        sim, app = self.createSimulation(no_graphics)

        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

        for i in range(N_traj):
            print(i)
            t_array = []
            x_array = []
            u_array = []
            s_array = []
            eta_array = []

            y_array = []

            # Set the simulation scene
            sim, app, ter, shov, driver, locksphere = self.setScene(sim, app)


            sim.setTimeStep(self.dt_data)
            if app:
                app.setTimeStep(self.dt_data)                    

            # Extract soil shape along the bucket excavation direction
            x_hf,z_hf = self.extractSoilSurface(ter, sim)          
            soilShapeEvaluator = SoilSurfaceEvaluator(x_hf, z_hf)
    
            # app.setupVideoCaptureRenderTotexture()
            # app.setAllowWindowResizing(False)

            # vc = app.getVideoServerCapture()
            # vc.setFilename("agx_mov")
            
            # print('----------------------------------------------------------')
            # print(vc.setOutputVideoCodec(1))
            # print('----------------------------------------------------------')


            # vc.setEnableSyncWithSimulation(True)
            # vc.startCapture()

            # time.sleep(10.0)
            myCapture = app.getImageCapture()
            myCapture.setMaxImages(20)
            myCapture.setFps(0.3)
            myCapture.setDirectoryPath('captured_images/'+date_time+'/')
            myCapture.setPrefix(str(self.consecutive_scoop_i)+'test')  
            myCapture.startCapture() 
            # myCapture.writeImage('image', 0)

            # Perform simulation
            while sim.getTimeStamp() <= T:

                # Measure all the states
                pos_tip, vel_tip, acl_tip, ang_tip, omega, alpha = measureBucketState(locksphere, shov)
                soil_force, fill = measureSoilQuantities(shov, ter)
                surf, surf_d, surf_dd, surf_ddd = soilShapeEvaluator.soil_surf_eval(pos_tip[0])
               
                if pos_tip[0] > -.55:
                    break

                t_array.append(sim.getTimeStamp())

                # Compose data in relevant arrays
                x_array.append(np.array([pos_tip[0], pos_tip[2], ang_tip, vel_tip[0], vel_tip[2], omega[1]]))
                # x_array.append(np.array([pos_tip[0], pos_tip[2], ang_tip ]))

                eta_array.append(np.array([acl_tip[0], acl_tip[2], alpha[1],
                                           self.scaling*soil_force[0],
                                           self.scaling*soil_force[2],
                                           self.scaling*fill]))

                ##############################################
                fill_dm  = ter.getDynamicMass(shov)  
                fill_dlf = ter.getLastDeadLoadFraction(shov)
                fill_am  = ter.getSoilAggregateMass(shov,0)
                
                agx_pbhf = ter.getSoilParticleBoundedHeightField()

                hf_grid_bucket = ter.getClosestGridPoint(agx.Vec3( pos_tip[0],  pos_tip[1],  pos_tip[2]))
                x_hf_bucket,  y_hf_bucket = hf_grid_bucket[0], hf_grid_bucket[1]

                z_hf = agx_pbhf.getHeight(x_hf_bucket,  y_hf_bucket)
                
                y_array.append(np.array([fill_dm, fill_dlf, fill_am, z_hf]))

                
                ##############################################    
                s_array.append(np.array([surf, surf_d, surf_dd]))

                # app.executeOneStepWithGraphics()
                # sim.stepForward()

                mass = shov.getRigidBody().getMassProperties().getMass() - 3.28

                p1 = shov.getRigidBody().getCmPosition()   
                p2 = locksphere.getPosition()   
                r  = p2 - p1
                

                # # Step the simulation forward
                if self.control_mode == "mpcc" or self.control_mode == "trajectory_control":
                    if app:
                        app.executeOneStepWithGraphics()
                    else: 
                        sim.stepForward()
                else:
                    # sim.stepForward()
                    app.executeOneStepWithGraphics()


                # Measure the used control inputs
                bucket_force = driver.force
                bucket_torque = driver.torque


                u_array.append(np.array([ self.scaling*bucket_force[0],
                                          self.scaling*(bucket_force[2] - 10.0*mass),
                                          self.scaling*(bucket_torque   - 10.0*mass*r[0])]))
            myCapture.stopCapture() 

            hf_final_agx = ter.getHeightField()
            self.agx_heightField_previous = hf_final_agx
            self.consecutive_scoop_i += 1

            t_array   = np.array(t_array)
            x_array   = np.array(x_array)
            u_array   = np.array(u_array)
            s_array   = np.array(s_array)
            eta_array = np.array(eta_array)

            y_array = np.array(y_array)

            T_data.append(t_array)
            X_data.append(x_array)
            U_data.append(u_array)
            S_data.append(s_array)
            Eta_data.append(eta_array)
            Y_data.append(y_array)

        T_data = np.array(T_data)
        X_data = np.array(X_data)
        U_data = np.array(U_data)
        S_data = np.array(S_data)
        Eta_data = np.array(Eta_data)
        Y_data = np.array(Y_data)

        # print('----------------------------------------------------------')
        # vc.stopCapture()        
        # print('----------------------------------------------------------')
        # # vc.closeExternalFFMPEGProcess()   
        
        # print('----------------------------------------------------------')
        # vc.stopProcess()

        # print(vc.getImageNum())

        return T_data, X_data, U_data, S_data, Eta_data, Y_data

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

        # self.prismatic_x.getLock1D().setEnable( False )
        # self.prismatic_z.getLock1D().setEnable( False )
        # self.hinge.getLock1D().setEnable( False )


    def setBodyForce(self, force):
        self.lockSphere.setForce(force)

    def setBodyTorque(self, torque):

        torque = agx.Vec3(0., torque, 0.)
        self.lockSphere.setTorque(torque)

    def post(self,t):
        
        self.soil_force_last = self.terrain.getSeparationContactForce(self.shovel)
        # print('Soil force in post integration: ',self.soil_force_last)

    def pre(self, t):

        force  = self.operations[0]
        torque = self.operations[1]
        
        # Measure all the states
        self.pos_tip, self.vel_tip, self.acl_tip, self.ang_tip, self.omega, self.alpha = measureBucketState(self.lockSphere, self.shovel)
 
        mass =  self.terrain.getDynamicMass(self.shovel)
        
        s_nom, s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom = self.soilShapeEvaluator.soil_surf_eval(self.pos_tip[0])

        D = s_nom - self.pos_tip[2]
        print(D)
        if (t-self.t_last_setpoint) > .5:
            
            self.t_last_setpoint = t
            # generate pseudo-random velocity set point
            if D < 0.15:
                self.theta_d = np.random.uniform(low = -0.45, high = -0.05)
                self.theta_d = np.random.uniform(low = -0.45, high = -0.35)

            elif D >= 0.15 and D < 0.3:
                self.theta_d = np.random.uniform(low = -0.25, high = 0.25)
            elif D >= 0.3:
                self.theta_d = np.random.uniform(low = -0.1 , high = 0.45)
            
            self.theta_d  += np.random.uniform(low = -0.1 , high = 0.1)

            self.v_d = np.random.uniform(low = 0.3, high = 1.0)
            # self.omega_d = np.random.uniform(low = -0.5, high = 0.5)
            self.ang_d = np.clip(self.ang_d + np.random.uniform(low = -0.01, high = 0.01), 1.45, 1.65)
            # self.ang_d = 1.55
            self.v_x_d = np.cos(self.theta_d)*self.v_d
            self.v_z_d = np.sin(self.theta_d)*self.v_d
        
        # if (t-self.t_last_control) >= self.dt_control:

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
        torque    = -5000.*(1.*np.sign(e_ang)*e_ang**2   + .01*self.integ_e_ang + 0.20*self.omega[1]) + 10.0*mass*r[0]

        self.force  = force 
        self.torque = torque

        # self.force[0] += np.random.uniform(low = -5000, high = 5000)
        # self.force[2] += np.random.uniform(low = -5000, high = 5000)
        # self.torque   += np.random.uniform(low = -5000, high = 5000)
        
        self.setBodyForce(self.force)
        self.setBodyTorque(self.torque)

class ForceDriverTrajectory(agxSDK.StepEventListener):
    
    def __init__(self, app, lockSphere, lock, hinge, prismatic_x, prismatic_z, terrain, shovel, operations, dt_control):
        super(ForceDriverTrajectory, self).__init__()
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
        self.sd = app.getSceneDecorator()

        self.theta_d = -0.2
        self.ang_d = 1.55
        self.v_d = 1.2

        self.path_progression = -10.

        self.soil_force_last = agx.Vec3(0.,0.,0.)

        self.force  = self.operations[0]
        self.torque = 0.0

        self.t_last_control =  -100.
        self.t_last_setpoint = -100.

        self.integ_e_x = 0.0
        self.integ_e_z = 0.0
        self.integ_e_omega = 0.0
        self.integ_e_ang = 0.0

        # self.prismatic_x.getLock1D().setEnable( False )
        # self.prismatic_z.getLock1D().setEnable( False )
        # self.hinge.getLock1D().setEnable( False )


    def setBodyForce(self, force):
        self.lockSphere.setForce(force)

    def setBodyTorque(self, torque):

        torque = agx.Vec3(0., torque, 0.)
        self.lockSphere.setTorque(torque)

    def post(self,t):
        
        self.soil_force_last = self.terrain.getSeparationContactForce(self.shovel)
        # print('Soil force in post integration: ',self.soil_force_last)

    def pre(self, t):

        force  = self.operations[0]
        torque = self.operations[1]
        
        # Measure all the states
        self.pos_tip, self.vel_tip, self.acl_tip, self.ang_tip, self.omega, self.alpha = measureBucketState(self.lockSphere, self.shovel)
 
        
        s_nom, s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom = self.soilShapeEvaluator.soil_surf_eval(self.pos_tip[0])
        self.path_progression += self.v_d*0.01
       
        x_d, y_d = self.path_eval(self.path_progression)
        dxds, dyds = self.path_eval(self.path_progression, d = 1)
        
        self.v_x_d = dxds*self.v_d
        self.v_z_d = dyds*self.v_d


        self.t_last_control = t

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
        torque    = -5000.*(1.*np.sign(e_ang)*e_ang**2   + .01*self.integ_e_ang + 0.20*self.omega[1]) + 10.0*mass*r[0]

        # # PID - VELOCITY CONTROL
        # # calculate errors and error integral

        # e_p_x = self.pos_tip[0] - x_d
        # e_p_z = self.pos_tip[2] - y_d 
        # e_ang = self.ang_tip - self.ang_d 

        # self.integ_e_x += e_p_x
        # self.integ_e_z += e_p_z
        # self.integ_e_ang += e_ang

        # pos     = self.bucket.getCmPosition()   
        # pos_tip = self.lockSphere.getPosition()   
        # r       = pos_tip - pos

        # mass = self.bucket.getMassProperties().getMass() - 3.28

        # force[0]  = -2000.*(1.*(e_p_x) + .05*self.integ_e_x)
        # force[2]  = -2000.*(1.*(e_p_z) + .05*self.integ_e_z) + 10.0*mass
        # torque    = -5000.*(1.*np.sign(e_ang)*e_ang**2   + .01*self.integ_e_ang + 0.20*self.omega[1]) + 10.0*mass*r[0]

        self.force  = force 
        self.torque = torque
        
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

        # if (t-self.t_last_control) > self.dt_control:

            # s_nom, s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom = self.soilShapeEvaluator.soil_surf_eval(self.pos[0])
            # A_lin, B_lin, K_lin = self.dfl.linearize_soil_dynamics(x_nom)
            
            # xi = np.array([self.pos[0], self.pos[2], self.vel[0], self.vel[2], self.fill])

            # xi = self.dfl.g_Koop(np.array([self.pos[0], self.pos[2]]),np.array([self.vel[0], self.vel[2], self.fill]))
            # xi = 
            # xi = xi - np.concatenate((self.x_offset,self.e_offset))           
        x = np.array([ self.pos_tip[0], self.pos_tip[2], self.angle_tip,  self.vel_tip[0], self.vel_tip[2], self.omega[1]])
        eta = np.array([self.acl_tip[0], self.acl_tip[2], self.alpha[1], self.scaling*self.soil_force_last[0],  self.scaling*self.soil_force_last[2], self.scaling*self.fill])
        xi =  self.dfl.g_Koop(x,eta, None )


        # xi =  np.array([ self.pos_tip[0], self.pos_tip[2], self.angle_tip,  self.vel_tip[0], self.vel_tip[2], self.omega[1], self.scaling*self.soil_force_last[0],  self.scaling*self.soil_force_last[2], self.scaling*self.fill])
        # xi = self.dfl.g_Koop(np.array([ self.pos_tip[0],  self.pos_tip[2],  self.angle_tip]),
        #                      np.array([ self.vel_tip[0],  self.vel_tip[2],  self.omega[1],
        #                                 self.scaling*self.soil_force_last[0],  self.scaling*self.soil_force_last[2], self.scaling*self.fill]),0)
        

        e_x     = self.pos_tip[0] - self.last_x_opt[0]
        e_z     = self.pos_tip[2] - self.last_x_opt[1]
        e_ang   = self.angle_tip  - self.last_x_opt[2]

        e_v_x   = self.vel_tip[0] - self.last_x_opt[3]
        e_v_z   = self.vel_tip[2] - self.last_x_opt[4]
        e_omega = self.omega[1]   - self.last_x_opt[5]

        self.integ_e_x += e_v_x
        self.integ_e_z += e_v_z
        self.integ_e_ang += e_ang


        # force_pid[0]  = -1000.*(1.*(e_v_x) + .5*self.integ_e_x)
        # force_pid[2]  = -1500.*(1.*(e_v_z) + .5*self.integ_e_z)
        # torque_pid    = -5000.*(1.*np.sign(e_ang)*e_ang**2   + .01*self.integ_e_ang + 0.20*self.omega[1]) 
        force_pid[0]  = -1000.*(self.integ_e_x)
        force_pid[2]  = -1500.*(self.integ_e_z)
        torque_pid    = -5000.*(self.integ_e_ang) 


        sliding_x       =  10.*e_x +  e_v_x 
        sliding_z       =  10.*e_z +  e_v_z 
        sliding_ang     =  10.*e_ang +  e_omega 

        # u_sliding_x     = -4000*np.sign(sliding_x)
        # u_sliding_z     = -4000*np.sign(sliding_z)
        # u_sliding_ang   = -4000*np.sign(sliding_ang)

        u_sliding_x     = -7000*np.clip(sliding_x,-1,1)
        u_sliding_z     = -7000*np.clip(sliding_z,-1,1)
        u_sliding_ang   = -7000*np.clip(sliding_ang,-1,1)


        # print("e_ang: ",e_ang)
        # print("e_omega: ",e_omega)
        # print("u_sliding: ",u_sliding_ang)

        print('---------------------------------------------')
        ####################### MPCC CONTROL #########################
        U, x_opt = self.mpcc.control_function(xi, t)
        
        self.last_x_opt = x_opt
        
        print('xi: ', xi)
        print("x_min constraints: ", xi<self.mpcc.x_min[:-1])
        print("x_max constraints: ", xi>self.mpcc.x_max[:-1])
        print("optimal input:", U )
        ####################### LQR CONTROL ##########################
        # U = np.array(-self.K_lqr.dot( xi_error ))
        # xi_error = xi - np.array([xi[0], -0.2, 0.5, xi[3], xi[4] ])
        # print("ERROR: ",xi_error )
        # print("Control: ",U )
 
        pos     = self.bucket.getCmPosition()   
        pos_tip = self.lockSphere.getPosition()   
        r       = pos_tip - pos

        mass = self.bucket.getMassProperties().getMass() - 3.28


        force[0] =  u_sliding_x   + U[0]/self.scaling
        force[2] =  u_sliding_z   + U[1]/self.scaling + 10.0*mass
        torque   =  u_sliding_ang + U[2]/self.scaling + 10.0*mass*r[0]

        print(u_sliding_ang)
        print(torque)
        # print("--------PID/TOTAL-----------")
        # a1 = force_pid[0] / (U[0]/self.scaling)
        # a2 = force_pid[2] / (U[1]/self.scaling)
        # a3 =  torque_pid / (U[2]/self.scaling)

        self.pid_array.append(np.array([ sliding_x, sliding_z, sliding_ang]))
        # print("in function: ", 10.0*mass*r[0])
        # if t > 4.0:
        #     a = np.array(self.pid_array)
        #     plt.plot(a[:,0])
        #     plt.plot(a[:,1])
        #     plt.plot(a[:,2])
        #     plt.show()

        self.t_last_control = t
        
        self.force  = force
        self.torque = torque

        self.setBodyForce(self.force)
        self.setBodyTorque(self.torque)
        # self.setRotation(self.omega_d)

class DiggingPlant():
    
    def __init__(self):

        # Linear part of states matrices
        self.n_x    = 6
        self.n_eta  = 7
        self.n_u    = 3

        # self.n_x    = 3
        # self.n_eta  = 6
        # self.n_u    = 3

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

        # User defined matrices for DFL
        # self.A_cont_x  = np.array([[ 0., 0., 0.],
        #                            [ 0., 0., 0.],
        #                            [ 0., 0., 0.]])

        # self.A_cont_eta = np.array([[ 1., 0., 0., 0., 0., 0.],
        #                             [ 0., 1., 0., 0., 0., 0.],
        #                             [ 0., 0., 1., 0., 0., 0.]])

        # self.B_cont_x = np.array([[ 0., 0., 0.],
        #                           [ 0., 0., 0.],
        #                           [ 0., 0., 0.]])

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

def plotData2(t, x, u, s, e, t2=None, x2=None, u2=None, s2=None, e2=None, comparison=False):

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

        axs[1,0].plot(t[0,:],e[i,:,0],'r',marker=".")
        axs[1,0].plot(t[0,:],e[i,:,1],'g',marker=".")
        axs[1,0].plot(t[0,:],e[i,:,2],'b',marker=".")
        axs[1,0].set_title("tip velocity")

        axs[2,0].plot(t[0,:],e[i,:,3],'r', marker = ".")
        axs[2,0].plot(t[0,:],e[i,:,4],'g', marker = ".")
        axs[2,0].set_title("soil force")

        axs[3,0].plot(t[0,:],u[i,:,0],'r', marker = ".")
        axs[3,0].plot(t[0,:],u[i,:,1],'g', marker = ".")
        axs[3,0].plot(t[0,:],u[i,:,2],'b', marker = ".")
        axs[3,0].set_title("bucket force")

        axs[4,0].plot(t[0,:],e[i,:,5],'r')
        axs[4,0].set_title("Bucket Fill")

        # soil shape variables
        axs[2,1].plot(t[0,:],x[i,:,1],'k')
        axs[2,1].plot(t[0,:],s[i,:,0],'r', marker=".")
        axs[2,1].set_title("Soil height")

        axs[3,1].plot(t[0,:],s[i,:,1],'r', marker=".")
        axs[3,1].set_title("Soil gradient")

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

        axs[1,0].plot(t[0,:],e2[:,0],'r--')
        axs[1,0].plot(t[0,:],e2[:,1],'g--')
        axs[1,0].plot(t[0,:],e2[:,2],'b--')
        axs[1,0].set_title("tip velocity")

        axs[2,0].plot(t[0,:],e2[:,3],'r--')
        axs[2,0].plot(t[0,:],e2[:,4],'g--')
        axs[2,0].set_title("soil force")

        axs[3,0].plot(t[0,:],u2[:,0],'r--')
        axs[3,0].plot(t[0,:],u2[:,1],'g--')
        axs[3,0].plot(t[0,:],u2[:,2],'b--')
        axs[3,0].set_title("bucket force")

        axs[4,0].plot(t[0,:],e2[:,5],'r--')
        axs[4,0].set_title("Bucket Fill")




    plt.subplots_adjust(left = 0.2, top = 0.89, hspace = 0.4)
    fig.tight_layout()
    plt.show()

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

    np.savez('data.npz',   t = t,
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

def g_Koop_x_eta(self,x,eta,s):

        poly = PolynomialFeatures(self.koop_poly_order,include_bias=False)
        xi = np.array(np.concatenate((x,eta)))
        y = poly.fit_transform(np.expand_dims(xi,axis=0))

        return y[0,:]

def main(args):

    dt_control = 0.01
    dt_data = 0.01
    T_traj_data = 7.5
    N_traj_data = 1
    
    plot_data = False
    save_data = False
    use_saved_data = False
    
    T_traj_test = 5.0
    N_traj_test = 1

    agx_sim = AgxSimulator(dt_data, dt_control)    
    agx_sim.model_has_surface_shape = True 
    plant = DiggingPlant()

    dfl = DFLSoil(plant, dt_data    = dt_data,
                         dt_control = dt_control)
    setattr(agx_sim, "dfl", dfl)
   
    if use_saved_data:
        t, x, u, s, e = loadData('data/data_nick_not_flat.npz')
    else:
        t, x, u, s, e, y = agx_sim.collectData(T = T_traj_data, N_traj = N_traj_data)
   
    fig, axs = plt.subplots(2,1, figsize=(8,10))
    print(y.shape)
    axs[0].plot(t[0,:],y[0,:,0]/y[0,-1,0],'r',marker=".")
    axs[0].plot(t[0,:],y[0,:,1]/y[0,-1,1],'g',marker=".")
    axs[0].plot(t[0,:],y[0,:,2]/y[0,-1,2],'b',marker=".")
    axs[0].set_title("tip position")
    
    axs[1].plot(t[0,:],y[0,:,3],'r',marker=".")
    axs[1].plot(t[0,:],s[0,:,0],'g',marker=".")
    axs[1].plot(t[0,:],x[0,:,1],'b',marker=".")
    axs[1].set_title("surcharge height")
    plt.show()

    exit()

    if save_data:
        saveData(t, x, u, s, e)
    
    if plot_data:
        plotData2(t, x, u, s, e)
        # plotData2(t, x[trial_inid, :, :],u[trial_inid, :, :],s[trial_inid, :, :],e[trial_inid, :, :])

    # fig = plt.figure(figsize=[6.4, 2.8])
    # ax = fig.add_subplot(1, 1, 1)
    # ax.quiver(x[0, ::10, 0],
    #           x[0, ::10, 1],
    #           -0.5*np.cos(x[0, ::10, 2]),
    #            0.5*np.sin(x[0, ::10, 2]), units = 'xy',headwidth = 0.0, 
    #            headlength = 0.0, scale = 2.2,width = 0.005)
    # ax.plot(x[0, :, 0], x[0, :, 1],color = 'black')
    
    # ax.set_ylabel(r'$y$   $(m^2)$')
    # ax.set_xlabel(r'$x$   $(m^2)$')
    # ax.axis('equal')
    # plt.tight_layout()
    # plt.show()
    
    
    dfl.koop_poly_order = 1
    # setattr(dfl, "g_Koop", dfl.g_Koop_x_eta)
    setattr(dfl, "g_Koop", dfl.g_Koop_x_eta_2)

    # agx_sim.dfl.regress_model_Koop_with_surf(x[trial_inid, :, :],e[trial_inid, :, :],u[trial_inid, :, :],s[trial_inid, :, :])
    agx_sim.dfl.regress_model_Koop_with_surf(x,e,u,s)
    # agx_sim.dfl.regress_model_custom(x,e,u,s)

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
    ###########################################################
    
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

    ####################### evaluation ########################
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
    ###########################################################
    
    # y_error_dfl = evaluate_error_dataset_size(t, x, u, s, e)
    # y_error_dfl = evaluate_error_modified(t, x, u, s, e)

    # exit()

    # A,B,K = dfl.linearize_soil_dynamics(np.concatenate((x[0,0,:],e[0,0,:])))
    
    # print(A)
    # print(B)
    
    # agx_sim.dfl.regress_model_new(x,e,u,s)
    # A,B,K = dfl.linearize_soil_dynamics(np.concatenate((x[0,0,:],e[0,0,:])))

    # print(A)
    # print(B)
    
    # exit()
    # t_valid, x_valid, u_valid, s_valid, e_valid = agx_sim.collectData(T = 12, N_traj = 1)

    # y_dfl = np.zeros((x_valid.shape[1],plant.n))
    # y_dfl[0,:] = np.concatenate((x_valid[-1,0,:],e_valid[-1,0,:]))
        
    # for i in range(x_valid.shape[1] - 1):
    #     y_dfl[i+1,:] = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[i,:], u_valid[-1,i,:])
    
    # plotData2(t_valid, x_valid, u_valid, s_valid, e_valid,
    #      t_valid, y_dfl[:,:plant.n_x ], u_valid[-1,:,:], s_valid, y_dfl[:,plant.n_x:], comparison = True)
    
    # exit()

    # agx_sim.control_mode = "trajectory_control"

    # # re-run with
    # t_gt_pid, x_gt_pid, u_gt_pid, s_gt_pid, e_gt_pid = agx_sim.collectData(T = 5.0, N_traj =  N_traj_test)
    
    agx_sim.control_mode = "mpcc"

    # re-run with
    agx_sim.q_theta_mpcc = 5.
    t_gt_mpc_1, x_gt_mpc_1, u_gt_mpc_1, s_gt_mpc_1, e_gt_mpc_1 = agx_sim.collectData(T = 7.5, N_traj =  N_traj_test)
    
    agx_sim.q_theta_mpcc = 5. # 8 
    agx_sim.set_height_from_previous = True
    t_gt_mpc_2, x_gt_mpc_2, u_gt_mpc_2, s_gt_mpc_2, e_gt_mpc_2 = agx_sim.collectData(T = 7.5 , N_traj =  N_traj_test)

    agx_sim.q_theta_mpcc = 5. # 8 
    agx_sim.set_height_from_previous = True
    t_gt_mpc_3, x_gt_mpc_3, u_gt_mpc_3, s_gt_mpc_3, e_gt_mpc_3 = agx_sim.collectData(T = 7.5 , N_traj =  N_traj_test)

    agx_sim.q_theta_mpcc = 5. # 8 
    agx_sim.set_height_from_previous = True
    t_gt_mpc_4, x_gt_mpc_4, u_gt_mpc_4, s_gt_mpc_4, e_gt_mpc_4 = agx_sim.collectData(T = 7.5 , N_traj =  N_traj_test)

    _, _, _, _, _ = agx_sim.collectData(T = 0.1 , N_traj =  N_traj_test)

    # print(t_gt_mpc_1[0, -1])
    # print(t_gt_mpc_2[0, -1])

    # y_dfl = np.zeros((x_gt.shape[1],plant.n))
    # y_dfl[0,:] = np.concatenate((x_gt[0,0,:], e_gt[0,0,:]))
    # y_dfl[0,:] =  dfl.g_Koop(x_gt[0,0,:],e_gt[0,0,:])


    # for i in range(x_gt.shape[1] - 1):
    #     y_dfl[i+1,:] = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[i,:], u_gt[0,i,:])

    # plotData(t_gt, x_gt, u_gt, s_gt, e_gt,
    #          t_gt, y_dfl[:,: plant.n_x], u_gt[0,:,:], s_gt, y_dfl[:,plant.n_x :], comparison = True)
    plotData3(t_gt_mpc_1, x_gt_mpc_1, u_gt_mpc_1, s_gt_mpc_1, e_gt_mpc_1, dfl)
   
    
    # ax.quiver(x_gt_mpc[0, ::10, 0],
    #           x_gt_mpc[0, ::10, 1],
    #           -0.5*np.cos(x_gt_mpc[0, ::10, 2]),
    #            0.5*np.sin(x_gt_mpc[0, ::10, 2]),units = 'xy',headwidth = 0.0,headlength = 0.0, scale = 2.2, width = 0.005,color = 'royalblue')

    # ax.quiver(x_gt_pid[0, ::10, 0],
    #           x_gt_pid[0, ::10, 1],
    #           -0.5*np.cos(x_gt_pid[0, ::10, 2]),
    #            0.5*np.sin(x_gt_pid[0, ::10, 2]),units = 'xy',headwidth = 0.0,headlength = 0.0, scale = 2.2, width = 0.005,color = 'forestgreen')
    
    fig = plt.figure(figsize=[6.4, 2.8])
    ax = fig.add_subplot(1, 1, 1)    
    
    ax.plot(np.array([-8,5]),np.array([0,0]), color = 'black')

    ax.plot(x_gt_mpc_1[0, :, 0], x_gt_mpc_1[0, :, 1], color = 'lightsteelblue')
    ax.plot(x_gt_mpc_2[0, :, 0], x_gt_mpc_2[0, :, 1], color = 'cornflowerblue')
    ax.plot(x_gt_mpc_3[0, :, 0], x_gt_mpc_3[0, :, 1], color = 'royalblue')
    ax.plot(x_gt_mpc_4[0, :, 0], x_gt_mpc_4[0, :, 1], color = 'mediumblue')

    ax = agx_sim.mpcc.draw_soil(ax,-5, 5)
    ax = agx_sim.mpcc.draw_path(ax, -10, -5)

    ax.set_ylabel(r'$\mathit{y}$  $(m)$')
    ax.set_xlabel(r'$\mathit{x}$  $(m)$')
    # ax.legend(['Bucket tip '])
    ax.axis('equal')
    plt.tight_layout()
    ax.set_xlim(-4.5,0.0)  
    ax.set_ylim(np.amin(x_gt_mpc_1[0, :, 1])-0.5,np.amax(x_gt_mpc_1[0, :, 1]) + 0.5)
    pickle.dump(fig, open('Figure_mpcc_multi_scoop.fig.pickle', 'wb')) 

    plt.show()


# Entry point when this script is loaded with python
if agxPython.getContext() is None:
    init = agx.AutoInit()
    main(sys.argv)