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
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splrep, splev, splint
import control
import scipy
import copy

from dfl.dfl.dfl_soil_agx import *
from dfl.dfl.mpcc import *
from dfl.dfl.dynamic_system import *

import sys
import os
import argparse
import time
from collections import namedtuple
from agxPythonModules.utils.environment import simulation, root, application

sys.path.append(os.getenv("AGX_DIR") + "/data/python/tutorials")
from tutorial_utils import createHelpText
from agxPythonModules.utils.numpy_utils import BufferWrapper

np.set_printoptions(precision = 3, suppress = True)

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

    def setupCamera(self,  app):
        cameraData                   = app.getCameraData()
        cameraData.eye               = agx.Vec3( 1.7190326542940962E+01, -1.4658716770523059E-01, 1.2635000298378865E+01 )
        cameraData.center            = agx.Vec3( 3.4621315146672371E-01, -1.2032941390018395E-01, -4.7443399198018110E-01 )
        cameraData.up                = agx.Vec3( -6.1418716626367531E-01, 2.8103832342862844E-04, 7.8916034227174481E-01 )
        cameraData.nearClippingPlane = 0.1
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

    def createRandomHeightfield(self, n_x, n_y, r, random_heaps = 5):

        np_HeightField = np.zeros((n_x, n_y))
        
        x = np.linspace(0.0, r*(n_x-1), n_x)
        y = np.linspace(0.0, r*(n_y-1), n_y)
        X, Y = np.meshgrid(x, y)

        # add/remove some random heaps
        for i in range(random_heaps):

            heap_height = np.random.uniform(0.3,0.5,1)
            heap_sigma  = np.random.uniform(1.0,2.0,1)

            x_c = np.random.uniform(low = 0.0, high = n_x*r, size = 1)
            y_c = np.random.uniform(low = 0.0, high = n_y*r, size = 1)

            surf_heap_i = heap_height*np.exp(-(np.square(X-x_c) + np.square(Y-y_c))/heap_sigma**2)
            np_HeightField = np_HeightField + -np.sign(np.random.uniform(-1,2,1))*surf_heap_i

        # np_HeightField = np_HeightField + np.random.uniform(low = -noise_level, high = noise_level, size = surface.shape)

        return np_HeightField

    def createFixedHeightfield(self, n_x, n_y, r, random_heaps = 5):

        np_HeightField = np.zeros((n_x, n_y))
        
        x = np.linspace(0.0, r*(n_x-1), n_x)
        y = np.linspace(0.0, r*(n_y-1), n_y)
        X, Y = np.meshgrid(x, y)

        heap_height = 0.50
        heap_sigma = 4.5

        x_c =  0.5*n_x*r
        y_c =  0.5*n_y*r

        surf_heap_i = heap_height*np.exp(-(np.square(X-x_c) + np.square(Y-y_c))/heap_sigma**2)
        np_HeightField = np_HeightField + -1*surf_heap_i

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

    def createSimulation(self, args):
        
        ap = argparse.ArgumentParser()

        # the --noApp will run this without a graphics window
        ap.add_argument('--noApp', action='store_true')
        args1, args2 = ap.parse_known_args()
        args1 = vars(args1)

        if args1['noApp']:
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
            np_heightField = self.createRandomHeightfield(num_cells_x, num_cells_y, cell_size)
        elif self.control_mode == "mpcc":
            np_heightField = self.createFixedHeightfield(num_cells_x, num_cells_y, cell_size)

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

        # Setup a renderer for the terrain
        renderer = agxOSG.TerrainVoxelRenderer( terrain, root )

        renderer.setRenderHeightField( True )
        # We choose to render the compaction of the soil to visually denote excavated
        # soil from compacted ground
        # renderer.setRenderCompaction( True, agx.RangeReal( 1.0, 1.05 ) )

        renderer.setRenderHeights(True, agx.RangeReal(-0.5,0.5))
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
        
        # initial x position of bucket
        x_initial_tip = np.random.uniform(low = -4.5, high = -3.0)

        # find the soil height at the initial penetration location
        hf_grid_initial = terrain.getClosestGridPoint(agx.Vec3(x_initial_tip, 0.0, 0.0))
        height_initial = terrain.getHeight(hf_grid_initial) - 0.02
        
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

        node1 = agxOSG.createVisual( sphere1, root)
        node2 = agxOSG.createVisual( sphere2, root)
        node3 = agxOSG.createVisual( sphere3, root)

        agxOSG.setDiffuseColor( node1, agxRender.Color.Red() )
        agxOSG.setDiffuseColor( node2, agxRender.Color.Green() )
        agxOSG.setDiffuseColor( node3, agxRender.Color.Blue() )

        agxOSG.setAlpha( node1 , 0.5 )
        agxOSG.setAlpha( node2 , 0.5 )
        agxOSG.setAlpha( node3 , 0.5 ) 

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
        
        elif self.control_mode == "testing":
            # create driver and add it to the simulation
            driver = ForceDriverDFL(app,
                                    sphere3,
                                    lock,
                                    hinge2,
                                    terrain,
                                    shovel,
                                    self.dt_control)
            
            # Add the current surface evaluator to the controller
            setattr(driver, "soilShapeEvaluator", self.soilShapeEvaluator)
            setattr(driver, "dfl",self.dfl)

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

            ################ LQR CONTROLLER ############################
            # A,B,k = self.dfl.linearize_soil_dynamics_no_surface(np.concatenate((self.x_offset,self.e_offset)))
            
            # Q = np.array([[10,0,0,0,0],
            #               [0,0,0,0,0],
            #               [0,0,10,0,0],
            #               [0,0,0,0,0],
            #               [0,0,0,0,0]])
            
            # R = np.array([[0.000001,0],
            #               [0      ,0.000001]])
           
            # K,S,E = dlqr(A,B,Q,R)
            # setattr(driver, "K_lqr", K)
            ################ MPCC CONTROLLER ############################

            # Add the current surface evaluator to the controller
            setattr(driver, "soilShapeEvaluator", self.soilShapeEvaluator)
            setattr(driver, "dfl", self.dfl)
            setattr(driver, "scaling",self.scaling)

            # x_array.append(np.array([pos[0], pos[2]]))
            # eta_array.append(np.array([vel[0], vel[2], soil_force[0], soil_force[2], fill_2]))
            print(x_initial_tip)
            
            if self.model_has_surface_shape:
                # define the path to be followed by MPCC
                x_path = x_initial_tip + np.array([0., 0.5, 1.5, 2.0, 2.5, 
                                                  3.0, 3.5, 4.0, 4.5, 5.0,
                                                  5.5, 6.0, 6.5, 7.0,7.5])
                y_soil,_,_,_ = self.soilShapeEvaluator.soil_surf_eval(x_path)
                y_path = y_soil + np.array([ -0.0, -0.15, -0.25, -0.25, -0.25,
                                             -0.25, -0.25, -0.25, -0.25, -0.25,
                                             -0.25, -0.25, -0.25, -0.25, -0.25])

                # x_min = np.array([ x_initial_tip-0.1 ,  -2. , 1.0   ,  -0.5 , -2.5 , -2.5 , -100, -100 , -100, -70000*self.scaling, -10000*self.scaling, 0.0])
                # x_max = np.array([ 2.                , 0.0  , 1.9   ,   2.5 ,  2.5 ,  2.5 ,  100,  100 ,  100,  70000*self.scaling,  70000*self.scaling, 3000.*self.scaling])
                
                x_min = np.array([ x_initial_tip-0.1 ,  -2. , 0.5   ,  -0.5 , -2.5 , -2.5 , -70000*self.scaling, -10000*self.scaling, 0.0])
                x_max = np.array([ 2.                , 0.0  , 1.9   ,   2.5 ,  2.5 ,  2.5 ,  70000*self.scaling,  70000*self.scaling, 3000.*self.scaling])
                
                u_min = np.array([ 0.                , -20000*self.scaling ,  -10000*self.scaling ])
                u_max = np.array([ 40000*self.scaling,  20000*self.scaling ,   10000*self.scaling ])

            else:

                x_min = np.array([ x_initial_tip-0.1,  -1.,  -0.1, -1.5, -70000*self.scaling, -10000*self.scaling,    0.])
                x_max = np.array([ 0.               , 0.1 ,   1.5,  1.3,  70000*self.scaling,  10000*self.scaling, 1000.*self.scaling])
                u_min = np.array([ 0.                , -5000*self.scaling])
                u_max = np.array([ 10000*self.scaling,  5000*self.scaling])

                # define the path to be followed by MPCC
                x_path = x_initial_tip + np.array([0., 0.5, 1.5, 2.0, 4.0])
                y_path = np.array([  0., -0.1, -0.3, -0.2, -0.2])

            spl_path = spline_path(x_path,y_path)


            # instantiate the MPCC object
            mpcc = MPCC(np.zeros((self.dfl.plant.n, self.dfl.plant.n)),
                        np.zeros((self.dfl.plant.n, self.dfl.plant.n_u)),
                        x_min, x_max,
                        u_min, u_max,
                        dt = self.dt_data, N = 15)

            #  set the observation function, path object and linearization function
            setattr(mpcc, "path_eval", spl_path.path_eval)
            setattr(mpcc, "get_soil_surface", self.soilShapeEvaluator.soil_surf_eval) 

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax = mpcc.draw_path(ax, -10, -5)
            ax = mpcc.draw_soil(ax, x_initial_tip -1, x_initial_tip + 5)
            ax.axis('equal')
            plt.show()

            if self.model_has_surface_shape:
                print('Linearizing with soil shape')
                setattr(mpcc, "get_linearized_model", self.dfl.linearize_soil_dynamics)
            else:
                setattr(mpcc, "get_linearized_model", self.dfl.linearize_soil_dynamics_no_surface)

            self.mpcc = copy.copy(mpcc)

            # define the MPCC cost matrices and coef.
            Q = 0.1*sparse.diags([20., 10.])
            R = sparse.diags([1., 1., 1., 1.])
            q_theta = 1.

            pos_tip, vel_tip, acl_tip, ang_tip, omega, alpha = measureBucketState(sphere3, shovel)

            # x_0 = np.concatenate((np.array([pos_tip[0], pos_tip[2], ang_tip ,
            #                                 vel_tip[0], vel_tip[2], omega[1],
            #                                 acl_tip[0], acl_tip[2], alpha[1],
            #                                 0.        ,  0.       ,     0.]), np.array([-10.0])))
           
            x_0 = np.concatenate((np.array([pos_tip[0], pos_tip[2], ang_tip ,
                                            vel_tip[0], vel_tip[2], omega[1],
                                            0.        ,  0.       ,     0.]), np.array([-10.0])))
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

        sim, app = self.createSimulation(None)

        for i in range(N_traj):
            
            t_array = []
            x_array = []
            u_array = []
            s_array = []
            eta_array = []

            # Set the simulation scene
            sim, app, ter, shov, driver, locksphere = self.setScene(sim, app)
           
            sim.setTimeStep(self.dt_data)
            app.setTimeStep(self.dt_data)                    

            # Extract soil shape along the bucket excavation direction
            x_hf,z_hf = self.extractSoilSurface(ter, sim)          
            soilShapeEvaluator = SoilSurfaceEvaluator(x_hf, z_hf)

            # Step the simulation forward
            # sim.stepForward()

            # Perform simulation
            while sim.getTimeStamp() <= T:

                t_array.append(sim.getTimeStamp())

                # Measure all the states
                pos_tip, vel_tip, acl_tip, ang_tip, omega, alpha = measureBucketState(locksphere, shov)
                soil_force, fill = measureSoilQuantities(shov, ter)
                surf, surf_d, surf_dd, surf_ddd = soilShapeEvaluator.soil_surf_eval(pos_tip[0])
               
                # Compose data in relevant arrays
                # x_array.append(np.array([pos_tip[0], pos_tip[2], ang_tip, vel_tip[0], vel_tip[2], omega[1]]))
                x_array.append(np.array([pos_tip[0], pos_tip[2], ang_tip ]))

                # eta_array.append(np.array([acl_tip[0], acl_tip[2], alpha[1],
                #                            self.scaling*soil_force[0],
                #                            self.scaling*soil_force[2],
                #                            self.scaling*fill]))

                eta_array.append(np.array([vel_tip[0], vel_tip[2], omega[1],
                                           self.scaling*soil_force[0],
                                           self.scaling*soil_force[2],
                                           self.scaling*fill]))

                s_array.append(np.array([surf, surf_d, surf_dd]))

                # app.executeOneStepWithGraphics()
                # sim.stepForward()


                # # Step the simulation forward
                if self.control_mode == "mpcc":
                    app.executeOneStepWithGraphics()
                    # sim.stepForward()
                    # time.sleep(2.0)
                else:
                    sim.stepForward()
                    # app.executeOneStepWithGraphics()


                # Measure the used control inputs
                bucket_force = driver.force
                bucket_torque = driver.torque

                mass = shov.getRigidBody().getMassProperties().getMass() - 3.28

                p1 = shov.getRigidBody().getCmPosition()   
                p2 = locksphere.getPosition()   
                r  = p2 - p1

                u_array.append(np.array([ self.scaling*bucket_force[0],
                                          self.scaling*(bucket_force[2] - 10.0*mass),
                                          self.scaling*(bucket_torque - 10.0*mass*r[0])]))
                
            t_array   = np.array(t_array)
            x_array   = np.array(x_array)
            u_array   = np.array(u_array)
            s_array   = np.array(s_array)
            eta_array = np.array(eta_array)

            T_data.append(t_array)
            X_data.append(x_array)
            U_data.append(u_array)
            S_data.append(s_array)
            Eta_data.append(eta_array)

        T_data = np.array(T_data)
        X_data = np.array(X_data)
        U_data = np.array(U_data)
        S_data = np.array(S_data)
        Eta_data = np.array(Eta_data)

        return T_data, X_data, U_data, S_data, Eta_data

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
        self.sd = app.getSceneDecorator()

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

        if (t-self.t_last_setpoint) > .2:
            
            self.t_last_setpoint = t
            # generate pseudo-random velocity set point
            if D < 0.15:
                self.theta_d = np.random.uniform(low = -0.45, high = -0.35)
            elif D >= 0.15 and D < 0.25:
                self.theta_d = np.random.uniform(low = -0.1, high = -0.05)
            elif D >= 0.25:
                self.theta_d = np.random.uniform(low = 0.1 , high = 0.25)
            
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
        torque    = -1000.*(1.*e_ang   + .3*self.integ_e_ang + 0.04*self.omega[1]) + 10.0*mass*r[0]

        self.force  = force 
        self.torque = torque

        self.force[0] += np.random.uniform(low = -5000, high = 5000)
        self.force[2] += np.random.uniform(low = -5000, high = 5000)
        self.torque   += np.random.uniform(low = -5000, high = 5000)
        
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
        self.sd = app.getSceneDecorator()

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
        torque = 0.0
        
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
        
        # xi =  np.array([ self.pos_tip[0], self.pos_tip[2], self.angle_tip,  self.vel_tip[0], self.vel_tip[2], self.omega[1], self.acl_tip[0], self.acl_tip[2], self.alpha[1], self.scaling*self.soil_force_last[0],  self.scaling*self.soil_force_last[2], self.scaling*self.fill])
        xi =  np.array([ self.pos_tip[0], self.pos_tip[2], self.angle_tip,  self.vel_tip[0], self.vel_tip[2], self.omega[1], self.scaling*self.soil_force_last[0],  self.scaling*self.soil_force_last[2], self.scaling*self.fill])

        print('---------------------------------------------')
        ####################### MPCC CONTROL #########################
        U = self.mpcc.control_function(xi, t)
        
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


        force[0] = U[0]/self.scaling
        force[2] = U[1]/self.scaling + 10.0*mass
        torque   = U[2]/self.scaling + 10.0*mass*r[0]

        self.t_last_control = t
        
        self.force  = force
        self.torque = torque

        self.setBodyForce(self.force)
        self.setBodyTorque(self.torque)
        # self.setRotation(self.omega_d)

class DiggingPlant():
    
    def __init__(self):

        # Linear part of states matrices
        # self.n_x    = 6
        # self.n_eta  = 6
        # self.n_u    = 3

        self.n_x    = 3
        self.n_eta  = 6
        self.n_u    = 3

        self.n      = self.n_x + self.n_eta

        # # User defined matrices for DFL
        # self.A_cont_x  = np.array([[ 0., 0., 0., 1., 0., 0.],
        #                            [ 0., 0., 0., 0., 1., 0.],
        #                            [ 0., 0., 0., 0., 0., 1.],
        #                            [ 0., 0., 0., 0., 0., 0.],
        #                            [ 0., 0., 0., 0., 0., 0.],
        #                            [ 0., 0., 0., 0., 0., 0.]])

        # self.A_cont_eta = np.array([[ 0., 0., 0., 0., 0., 0.],
        #                             [ 0., 0., 0., 0., 0., 0.],
        #                             [ 0., 0., 0., 0., 0., 0.],
        #                             [ 1., 0., 0., 0., 0., 0.],
        #                             [ 0., 1., 0., 0., 0., 0.],
        #                             [ 0., 0., 1., 0., 0., 0.]])

        # self.B_cont_x = np.array([[ 0., 0., 0.],
        #                           [ 0., 0., 0.],
        #                           [ 0., 0., 0.],
        #                           [ 0., 0., 0.],
        #                           [ 0., 0., 0.],
        #                           [ 0., 0., 0.]])

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[ 0., 0., 0.],
                                   [ 0., 0., 0.],
                                   [ 0., 0., 0.]])

        self.A_cont_eta = np.array([[ 1., 0., 0., 0., 0., 0.],
                                    [ 0., 1., 0., 0., 0., 0.],
                                    [ 0., 0., 1., 0., 0., 0.]])

        self.B_cont_x = np.array([[ 0., 0., 0.],
                                  [ 0., 0., 0.],
                                  [ 0., 0., 0.]])

    def set_soil_surf(self, x, y):

        self.tck_sigma = splrep(x, y, s = 0)

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

def plotData(t, x, u, s, e, t2=None, x2=None, u2=None, s2=None, e2=None, comparison=False):

    fig, axs = plt.subplots(5,2, figsize=(8,10))
    
    # if len(x.shape)==3:
    #     t = t.reshape(-1,t.shape[-1])
    #     x = x.reshape(-1,x.shape[-1])
    #     u = u.reshape(-1,u.shape[-1])
    #     s = s.reshape(-1,s.shape[-1])
    #     e = e.reshape(-1,e.shape[-1])

    for i in range(x.shape[0]):
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

    for i in range(x.shape[0]):
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

        axs[4,1].plot(u[i,:,0],u[i,:,2],'.')

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

def main(args):

    dt_control = 0.02
    dt_data = 0.02
    T_traj_data = 8.0
    N_traj_data = 5

    plot_data = True
    save_data = False
    use_saved_data = True

    T_traj_test = 10.0
    N_traj_test = 1

    agx_sim = AgxSimulator(dt_data, dt_control)    
    agx_sim.model_has_surface_shape = True 
    plant = DiggingPlant()

    dfl = DFLSoil(plant, dt_data    = dt_data,
                         dt_control = dt_control)
    setattr(agx_sim, "dfl", dfl)
   
    if use_saved_data:
        t, x, u, s, e = loadData('data.npz')
    else:
        t, x, u, s, e = agx_sim.collectData(T = T_traj_data, N_traj = N_traj_data)
    
    if plot_data:
        plotData2(t, x, u, s, e)
    
    if save_data:
        saveData(t, x, u, s, e)

    agx_sim.dfl.regress_model_custom(x,e,u,s)
    # agx_sim.dfl.regress_model_new(x,e,u,s)

    # A,B,K = dfl.linearize_soil_dynamics(np.concatenate((x[0,0,:],e[0,0,:])))

    # print(A)
    # print(B)

    # y_dfl = np.zeros((x.shape[1],plant.n))
    # y_dfl[0,:] = np.concatenate((x[0,0,:],e[0,0,:]))
        
    # for i in range(x.shape[1] - 1):
    #     y_dfl[i+1,:] = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[i,:], u[0,i,:])
    
    # plotData2(t, x, u, s, e,
    #      t, y_dfl[:,:plant.n_x ], u[0,:,:], s, y_dfl[:,plant.n_x:], comparison = True)
    
    # exit()

    agx_sim.control_mode = "mpcc"

    # re-run with
    t_gt, x_gt, u_gt, s_gt, e_gt = agx_sim.collectData(T = T_traj_test, N_traj =  N_traj_test)
    
    # y_dfl = np.zeros((x_gt.shape[1],plant.n))
    # y_dfl[0,:] = np.concatenate((x_gt[0,0,:], e_gt[0,0,:]))
    # # y_dfl[0,:] =  dfl.g_Koop(x_gt[0,0,:],e_gt[0,0,:])


    # for i in range(x_gt.shape[1] - 1):
    #     y_dfl[i+1,:] = agx_sim.dfl.f_disc_dfl_tv(0.0, y_dfl[i,:], u_gt[0,i,:])

    # plotData(t_gt, x_gt, u_gt, s_gt, e_gt,
    #          t_gt, y_dfl[:,: plant.n_x], u_gt[0,:,:], s_gt, y_dfl[:,plant.n_x :], comparison = True)
    plotData2(t_gt, x_gt, u_gt, s_gt, e_gt)
   
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_gt[0,:,0], x_gt[0,:,1],'.', color = 'tab:blue')
    ax = agx_sim.mpcc.draw_path(ax, -10, -5)
    ax = agx_sim.mpcc.draw_soil(ax,x_gt[0,0,0]-1, x_gt[0,-1,0]+5)
    ax.axis('equal')
    plt.show()

# Entry point when this script is loaded with python
if agxPython.getContext() is None:
    init = agx.AutoInit()
    main(sys.argv)