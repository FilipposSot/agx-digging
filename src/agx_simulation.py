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

# Plotting settings
import matplotlib
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['mathtext.default'] = 'rm'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["legend.loc"] = 'upper left'
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Import DFl library
from dfl.dfl.dfl_soil_agx import *
from dfl.dfl.mpcc import *
from dfl.dfl.dynamic_system import *

# utility imports
import pickle
import copy
import sys
import os
import argparse
import time
from datetime import datetime
from collections import namedtuple
from agxPythonModules.utils.environment import simulation, root, application

# agx imports
sys.path.append(os.getenv("AGX_DIR") + "/data/python/tutorials")
from tutorial_utils import createHelpText
from agxPythonModules.utils.numpy_utils import BufferWrapper

# Print settings for debugging
np.set_printoptions(precision = 5, suppress = True)
np.set_printoptions(edgeitems=30, linewidth=100000)
np.core.arrayprint._line_width = 200

# import controllers
from bucket_controllers import *

# Defualt shovel settings
default = {
    'length': 0.6,
    'width': 0.6,
    'height': 0.45,
    'thickness': 0.02,
    'topFraction': 0.25
}

# Helper class containing spline soil shape
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
# Simulation class
class AgxSimulator():

    def __init__(self, dt_data, dt_control):
        self.control_mode = "data_collection"
        self.dt_control = dt_control
        self.dt_data = dt_data

        self.scaling = .001
        
        self.set_height_from_previous = False
        self.consecutive_scoop_i = 0

    def setupCamera(self,  app):    
   
        cameraData                   = app.getCameraData()
        cameraData.eye               = agx.Vec3( -3.  , -10. , 2.4)
        cameraData.center            = agx.Vec3( -3.   ,  0.  , 0. )
        cameraData.up                = agx.Vec3( 0.0, 0.0, 0.0)
        cameraData.nearClippingPlane = 10
        cameraData.farClippingPlane  = 5000
        app.applyCameraData( cameraData )

    def setHeightField(self, agx_heightField, np_heightField):

        hf_size = agx_heightField.getSize()

        # Set the height field heights
        for i in range(0, agx_heightField.getResolutionX()):
            for j in range(0, agx_heightField.getResolutionY()):
                agx_heightField.setHeight(i, j, np_heightField[i,j])

        return agx_heightField

    # create the agx simulation bucket
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

    # Create a random heightfield with gaussian heaps
    def createRandomHeightfield(self, n_x, n_y, r, random_heaps = 7):

        np_HeightField = np.zeros((n_x, n_y))
        
        x = np.linspace(0.0, r*(n_x-1), n_x)
        y = np.linspace(0.0, r*(n_y-1), n_y)
        X, Y = np.meshgrid(x, y)

        # add/remove some random heaps
        for i in range(random_heaps):

            heap_height = np.random.uniform(0.3,0.7,1)
            heap_sigma  = np.random.uniform(3.0,4.0,1)

            x_c = np.random.uniform(low = 0.0, high = n_x*r, size = 1)
            y_c = np.random.uniform(low = 0.0, high = n_y*r, size = 1)

            surf_heap_i = heap_height*np.exp(-(np.square(X-x_c) + np.square(Y-y_c))/heap_sigma**2)
            np_HeightField = np_HeightField + -np.sign(np.random.uniform(-3,3,1))*surf_heap_i

        return np_HeightField

    # Create a fixed heightfield
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
        np_HeightField = np_HeightField #+ 0.1*np.sign(Y-3) #- 0.2*Y #0.2*np.sign(Y-3)

        return np_HeightField

    # Extract the heightfield from the agx terrain environment
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

    # create the agx simulation environment
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

    # Set the scene in the agx simulation
    def setScene(self, sim, app):
        
        # Start by cleaning up the simulation from any previous content
        sim.cleanup()

        # Next build a scene with some content
        ter, shov, driver, locksphere = self.buildTheScene(app, sim)

        if app:
            # Initialize the simulation in the application
            app.initSimulation(sim, True)

        return sim, app, ter, shov, driver, locksphere
    
    # Build the scene in the agx simulation
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

    # Build the scene in the agx simulation
    def buildScene1(self, app, sim, root):

        # Create the Terrain
        num_cells_x = 80
        num_cells_y = 80
        cell_size   = 0.15
        max_depth   = 1.0

        agx_heightField = agxCollide.HeightField(num_cells_x, num_cells_y, (num_cells_x-1)*cell_size, (num_cells_y-1)*cell_size)
        
        # Define the initial height field (random or fixed) depending on if if data collection or test
        if self.control_mode == "data_collection":
            np_heightField = self.createRandomHeightfield(num_cells_x, num_cells_y, cell_size)
        elif self.control_mode == "mpcc" or self.control_mode == "trajectory_control":
            np_heightField = self.createRandomHeightfield(num_cells_x, num_cells_y, cell_size)


        if self.set_height_from_previous:
            agx_heightField = self.agx_heightField_previous
        else:
            agx_heightField = self.setHeightField(agx_heightField,np_heightField)


        terrain = agxTerrain.Terrain.createFromHeightField(agx_heightField, 5.0)
        sim.add(terrain)
        
        # Define Gravity
        G = agx.Vec3(0, 0, -10.0)
        sim.setUniformGravity(G)
        
        # define the material
        terrain.loadLibraryMaterial("sand_1")

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
            renderer.setRenderVoxelFluidMass(False)
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

        if app:# and self.consecutive_scoop_i < 4:
            # Create visual representation of the shovel
            node = agxOSG.createVisual(bucket, root)
            agxOSG.setDiffuseColor(node, agxRender.Color.Gold())
            agxOSG.setAlpha(node, 1.0)

        # Set initial bucket rotation
        if self.control_mode == "mpcc":
            angle_bucket_initial = -0.3*np.pi
        else:
            angle_bucket_initial = np.random.uniform(low = -0.25*np.pi, high = -0.35*np.pi)

        bucket.setRotation(agx.EulerAngles(0.0, angle_bucket_initial, agx.PI))

        # Get the offset of the bucket tip from the COM
        tip_offset = shovel.getCuttingEdgeWorld().p2

        #
        inertia_tensor = bucket.getMassProperties().getInertiaTensor()
        mass = bucket.getMassProperties().getMass()
        h_offset_sqrd = tip_offset[0]**2 + tip_offset[2]**2
        inertia_bucket = inertia_tensor.at(1,1) + mass*h_offset_sqrd

        # Set initial bucket position (for consecutive scoops)
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
       
        sim.add(prismatic1)
        sim.add(prismatic2)
        sim.add(hinge2)

        lock = agx.LockJoint( sphere3, bucket )
        sim.add( lock )

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

            x_path = x_initial_tip + np.array([0., 0.5, 1.5, 2.0, 2.5, 3.0,])

            y_soil,_,_,_ = self.soilShapeEvaluator.soil_surf_eval(x_path)
            y_path = y_soil + np.array([ -0.07, -0.25, -0.25, -0.25, -0.25, -0.02])

            # Set the state constraints
            if self.observable_type == "dfl":

                x_min = np.array([ x_initial_tip-0.1 ,  -3. , 0.5 , -0.5 , -2.5 , -2.5 , -80000*self.scaling, -80000*self.scaling, 0.0               , -70000*self.scaling, -70000*self.scaling,-70000*self.scaling, -70000*self.scaling])
                x_max = np.array([ 2.                , 5.0  , 2.5 ,  2.5 ,  2.5 ,  2.5 ,  80000*self.scaling,  80000*self.scaling, 3000.*self.scaling,  70000*self.scaling,  70000*self.scaling, 70000*self.scaling,  70000*self.scaling])
                n_dyn = self.dfl.plant.n
            
            elif self.observable_type == "x":

                x_min = np.array([ x_initial_tip-0.1 ,  -3. , 0.5 , -0.5 , -2.5 , -2.5 ])
                x_max = np.array([ 2.                , 5.0  , 2.5 ,  2.5 ,  2.5 ,  2.5 ])
                n_dyn = self.dfl.plant.n_x

            # Set the input constraints    
            u_min = np.array([ -100.*self.scaling   , -70000*self.scaling ,  -70000*self.scaling ])
            u_max = np.array([  75000.*self.scaling,   70000*self.scaling ,   70000*self.scaling ])

            if self.set_height_from_previous: 
                pass
            else:
                self.spl_path = spline_path(x_path,y_path)


            # instantiate the MPCC object
            
            mpcc = MPCC(np.zeros((n_dyn, n_dyn)),
                        np.zeros((n_dyn, self.dfl.plant.n_u)),
                        x_min, x_max,
                        u_min, u_max,
                        dt = self.dt_data, N = 50)

            #  set the observation function, path object and linearization function
            setattr(mpcc, "path_eval",  self.spl_path .path_eval)
            setattr(mpcc, "get_soil_surface", self.soilShapeEvaluator.soil_surf_eval) 

            if self.model_has_surface_shape:
                print('Linearizing with soil shape')
                setattr(mpcc, "get_linearized_model", self.dfl.linearize_soil_dynamics_koop)
            else:
                setattr(mpcc, "get_linearized_model", self.dfl.linearize_soil_dynamics_no_surface)

            self.mpcc = copy.copy(mpcc)

            pos_tip, vel_tip, acl_tip, ang_tip, omega, alpha = measureBucketState(sphere3, shovel)
            

            x_i = np.array([ pos_tip[0], pos_tip[2], ang_tip,  vel_tip[0], vel_tip[2], omega[1]])
            eta_i = np.array([acl_tip[0], acl_tip[2], alpha[1], 0.,  0., 0., 0.])
             
            # Choose the initial path arcposition based on a close initial x tip position           
            theta_array = np.linspace(-10, 0, num=1000)
            for i in range(len(theta_array)):
                x_path, y_path = self.spl_path.path_eval(theta_array[i], d=0)
                if x_path > pos_tip[0]:
                    path_initial = theta_array[i]
                    break

            # set initial input (since input cost is differential)
            x_0_mpcc =  np.concatenate((self.dfl.g_Koop(x_i,eta_i,_), np.array([path_initial])))
            u_minus_mpcc = np.array([0.0, 0.0, 0.0, 0.0])
            driver.last_x_opt = x_0_mpcc

            # sets up the new mpcc problem _mpcc
            mpcc.setup_new_problem(self.Q_mpcc, self.R_mpcc, self.q_theta_mpcc, x_0_mpcc, u_minus_mpcc)

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
            no_graphics = True

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

            terraintoolcollection = agxTerrain.TerrainToolCollection(ter, shov )
            terraintoolcollection.initialize(sim)

            sim.setTimeStep(self.dt_data)
            if app:
                app.setTimeStep(self.dt_data)                    

            # Extract soil shape along the bucket excavation direction
            x_hf,z_hf = self.extractSoilSurface(ter, sim)          
            soilShapeEvaluator = SoilSurfaceEvaluator(x_hf, z_hf)
    
            # time.sleep(10.0)
            if not no_graphics:
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
               
                t_array.append(sim.getTimeStamp())

                # Compose data in relevant arrays
                x_array.append(np.array([pos_tip[0], pos_tip[2], ang_tip, vel_tip[0], vel_tip[2], omega[1]]))


                ##############################################
                fill_dm  = ter.getDynamicMass(shov)  
                fill_dlf = ter.getLastDeadLoadFraction(shov)*ter.getInnerVolume(shov)
                fill_am  = ter.getSoilAggregateMass(shov, 0)
                
                agx_pbhf        = ter.getSoilParticleBoundedHeightField()
                hf_grid_bucket  = ter.getClosestGridPoint(agx.Vec3( pos_tip[0],  pos_tip[1],  pos_tip[2]))
                x_hf_bucket,  y_hf_bucket = hf_grid_bucket[0], hf_grid_bucket[1]
                z_surcharge_hf            = agx_pbhf.getHeight(x_hf_bucket,  y_hf_bucket)

                terraintoolcollection = ter.getToolCollection( shov )

                r_soil_com = terraintoolcollection.getSoilParticleAggregate().getInnerGeometry().getRigidBody().getCmPosition()
                r_soil_com_mag = np.sqrt(r_soil_com[0]**2 + r_soil_com[2]**2)

                if r_soil_com_mag  == 0:
                    r_com = 0.0
                else:
                    r_com = np.sqrt((pos_tip[0]-r_soil_com[0])**2 + (pos_tip[2]-r_soil_com[2])**2)

                eta_array.append(np.array([ self.scaling*soil_force[0],
                                            self.scaling*soil_force[2],
                                            self.scaling*fill_dm,
                                            z_surcharge_hf,
                                            r_com]))
    
                s_array.append(np.array([surf, surf_d, surf_dd]))

                # app.executeOneStepWithGraphics()
                # sim.stepForward()

                mass = shov.getRigidBody().getMassProperties().getMass() - 3.28

                p1 = shov.getRigidBody().getCmPosition()   
                p2 = locksphere.getPosition()   
                r  = p2 - p1
                
                if no_graphics == True:
                    sim.stepForward()
                else:
                    app.executeOneStepWithGraphics()


                # Measure the used control inputs
                bucket_force = driver.force
                bucket_torque = driver.torque

                u_array.append(np.array([ self.scaling*bucket_force[0],
                                          self.scaling*(bucket_force[2] - 10.0*mass),
                                          self.scaling*(bucket_torque   - 10.0*mass*r[0])]))
           
            if not no_graphics:
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

        return T_data, X_data, U_data, S_data, Eta_data, Y_data