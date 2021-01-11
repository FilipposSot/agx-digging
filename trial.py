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
import sys
import os
import argparse
import time
from collections import namedtuple
from agxPythonModules.utils.environment import simulation, root, application

sys.path.append(os.getenv("AGX_DIR") + "/data/python/tutorials")
from tutorial_utils import createHelpText
from agxPythonModules.utils.numpy_utils import BufferWrapper


# Defualt shovel settings
default = {
    'length': 0.6,
    'width': 0.6,
    'height': 0.45,
    'thickness': 0.02,
    'topFraction': 0.25
}

class AgxSimulator():

    def __init__(self):
        pass

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
        print(hf_size[0])
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

        ter, shov, driver = self.buildScene1(app, sim, root)

        self.setupCamera(app)

        return ter, shov, driver

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
        np_heightField = np.zeros((num_cells_x, num_cells_y))
        agx_heightField = self.setHeightField(agx_heightField,np_heightField)

        terrain = agxTerrain.Terrain.createFromHeightField(agx_heightField, 5.0)
        sim.add(terrain)
        # print(terrain.getHeightField().getSize()[0])
        # print(heightField.getSize()[0])
        # exit()
        
        G = agx.Vec3(0, 0, -10.0)
        sim.setUniformGravity(G)
        

        # Setting so that we get vertical walls in compacted terrain while limiting swell factors
            # Load a material from the material library, this sets
        #   - Bulk material
        #   - Particle material
        #   - Terrain material
        #   - Particle-Particle contact material
        #   - Particle-Terrain contact material
        #   - Aggregate-Terrain contact material
        # WARNING:  Changing ANY material, bulk material or contact material retrieved from Terrain
        #           will invalidate these settings!
        # NOTE: Use the agxTerrain.Terrain.getAvailableLibraryMaterials() method to get the available
        #       material presets.
        
        terrain.loadLibraryMaterial("sand_1")
        # terrain.loadMaterialFile("sand_2.json")
        # a = terrain.getAvailableLibraryMaterials()
        terrainMaterial = terrain.getTerrainMaterial()
        terrainMaterial.getBulkProperties().setSwellFactor( 1.00 )
        compactionProperties = terrainMaterial.getCompactionProperties()

        #
        # This is the most important piece of the tutorial for creating vertical trenching walls in the soil.
        #
        # In order to create vertical walls during excavation in the terrain we need to set an initial compaction
        # level and increase the angle of repose compaction scaling. In this example, we create just a minimal
        # increase in compaction ( to limit material swelling after digging for practical reasons ) and also
        # increase the angle repose scaling significantly.
        #
        # The angle of repose compaction rate scales the tan ( angle of repose ) of the material by the
        # following factor:
        #
        # m  = 2.0 ^ ( angleOfReposeCompactionRate * ( compaction - ( 1.0 / swellFactor ) ) )
        #
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
        # print('material 1')
        shovelMaterial = agx.Material( "shovel_material" )
        # print('material 2')
        terrainMaterial = terrain.getMaterial( agxTerrain.Terrain.MaterialType_TERRAIN )
        shovelTerrainContactMaterial = agx.ContactMaterial( shovelMaterial, terrainMaterial )
        shovelTerrainContactMaterial.setYoungsModulus( 1e8 )
        shovelTerrainContactMaterial.setRestitution( 0.0 )
        shovelTerrainContactMaterial.setFrictionCoefficient( 0.4 )
        sim.add( shovelTerrainContactMaterial )

        #
        # Create the trenching shovel body creation, do setup in the Terrain object and
        # constrain it to a kinematic that will drive the motion
        #

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

        # Initial position of the bucket
        position = agx.Vec3(-4.75, 0, 0.3)
        bucket.setPosition( terrain.getTransform().transformPoint( position ) )

        # Initial rotation
        # bucket.setRotation(agx.Quat(1*agx.PI, agx.Vec3.Z_AXIS()))
        bucket.setRotation(agx.EulerAngles(0.0,-0.35*agx.PI,agx.PI))

        # Add a lockjoint between a kinematic sphere and the shovel
        # in order to have some compliance when moving the shovel
        # through the terrain
        offset = agx.Vec3( 0.0, 0.0, 0.0 )
        # kinematicSphere = agx.RigidBody( agxCollide.Geometry( agxCollide.Sphere( .1 ) ) )
        # kinematicSphere.setMotionControl( agx.RigidBody.DYNAMICS )
        # kinematicSphere.getGeometries()[ 0 ].setEnableCollisions( False )
        # kinematicSphere.setPosition( bucket.getFrame().transformPointToWorld( offset ))
        # kinematicSphere.setRotation( bucket.getRotation() )
        # simulation().add( kinematicSphere)

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

        sphere1.setPosition( bucket.getCmFrame().transformPointToWorld( offset ))
        sphere2.setPosition( bucket.getCmFrame().transformPointToWorld( offset ))
        sphere3.setPosition( bucket.getCmFrame().transformPointToWorld( offset ))

        sim.add(sphere1)
        sim.add(sphere2)
        sim.add(sphere3)

        node1 = agxOSG.createVisual( sphere1, root)
        node2 = agxOSG.createVisual( sphere2, root)
        node3 = agxOSG.createVisual( sphere3, root)

        agxOSG.setDiffuseColor( node1, agxRender.Color.Red())
        agxOSG.setDiffuseColor( node2, agxRender.Color.Green())
        agxOSG.setDiffuseColor( node3, agxRender.Color.Blue())

        agxOSG.setAlpha( node1, 0.5)
        agxOSG.setAlpha( node2, 0.5)
        agxOSG.setAlpha( node3, 0.5)

        # Set prismatic joint for x transalation world - sphere 1
        f1 = agx.Frame()
        f1.setLocalRotate(agx.EulerAngles(0, math.radians(90), 0))
        prismatic1 = agx.Prismatic(sphere1, f1)
        sim.add(prismatic1)

        # Set prismatic joint for x transalation world - sphere 2
        f1 = agx.Frame()
        f2 = agx.Frame()
        prismatic2 = agx.Prismatic(sphere1, f1, sphere2, f2)
        sim.add(prismatic2)

        # # Set hinge joint for rotation of the bucket
        f1 = agx.Frame()
        f1.setLocalRotate(agx.EulerAngles(math.radians(90),0, 0))
        f2 = agx.Frame()
        f2.setLocalRotate(agx.EulerAngles(math.radians(90),0, 0))
        f1.setLocalTranslate(0.0,0,0)
        hinge2 = agx.Hinge(sphere2, f1, sphere3, f2)
        sim.add(hinge2)

        lock = agx.LockJoint( sphere3, bucket )
        sim.add( lock )

        # sim.add(agx.LockJoint(sphere2,bucket))
        # matrixx = bucket.getMassProperties().getInertiaTensor()  
        # print(matrixx.at(0,0), matrixx.at(0,1), matrixx.at(0,2))
        # print(matrixx.at(1,0), matrixx.at(1,1), matrixx.at(1,2))
        # print(matrixx.at(2,0), matrixx.at(2,1), matrixx.at(2,2))

        # constant force and torque
        operations = [agx.Vec3( 0.0, 0.0, 0.0), agx.Vec3( 0.0, 0.0, 0.0 )]

        # Create driver of the kinematic sphere that drive shovel motion using the operations
        # list passed as an argument to the driver

        quat = bucket.getRotation()
        rpy = agx.EulerAngles(quat)

        driver = LockForceDriver(app,
                                sphere3,
                                lock,
                                hinge2,
                                terrain,
                                shovel,
                                operations )
        sim.add(driver)

        # Limit core usage to number of physical cores. Assume that HT/SMT is active
        # and divide max threads with 2.
        agx.setNumThreads( 0 )
        n = int(agx.getNumThreads() / 2 - 1)
        agx.setNumThreads( n )

        # Setup initial camera view
        createHelpText(sim, app)

        return terrain, shovel, driver

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
        ter, shov, driver = self.buildTheScene(app, sim)

        if app:
            # Initialize the simulation in the application
            app.initSimulation(sim, True)

        return sim, app, ter, shov, driver

    def runSimulation(self, sim, app, ter, shov, driver, T=5):
            # This is our simulation loop where we will
            # Step simulation, update graphics(if we are using an application window) 
            # We will step the simulation for 5 seconds

            force_array = []
            force2_array = []
            pos_array = []
            vel_array = []
            fill_array = []

            while sim.getTimeStamp() <= T:
                # Step the simulation forward
                sim.stepForward()

                # Print position of the first rigid body
                pos    = sim.getRigidBodies()[0].getPosition()
                vel    = sim.getRigidBodies()[0].getVelocity()
                force  = ter.getSeparationContactForce(shov)
                force2 = driver.force

                fill   = ter.getLastDeadLoadFraction(shov)

                force_array.append(np.array([force[0], force[1], force[2]]))
                force2_array.append(np.array([force2[0], force2[1], force2[2]]))
                pos_array.append(np.array([pos[0], pos[1], pos[2]]))
                vel_array.append(np.array([vel[0], vel[1], vel[2]]))
                fill_array.append(ter.getLastDeadLoadFraction(shov))

                if (not app):
                    print(pos)

                if app:
                    # Update the graphics window/entities
                    app.executeOneStepWithGraphics()

                # When running with a graphics window, we might want to slow it down a bit...
                # Remove this if you want to run as fast as possible
                if app:
                    time.sleep(0.01)

            force_array  = np.array(force_array)
            force2_array = np.array(force2_array)
            pos_array    = np.array(pos_array)
            vel_array    = np.array(vel_array)
            fill_array   = np.array(fill_array)

            # Extract and plot data
            plt.plot(force_array[:,0],'r')
            plt.plot(force_array[:,1],'g')
            plt.plot(force_array[:,2],'b')

            plt.plot(-force2_array[:,0], 'r--')
            plt.plot(-force2_array[:,1], 'g--')
            plt.plot(-force2_array[:,2], 'b--')
            
            plt.show()

            hf = ter.getHeightField()
            print(hf.getResolutionX())
            x = np.arange(0,hf.getResolutionX())*0.15
            z = np.zeros(x.shape)
            for hf_x_index in range(hf.getResolutionX()):
                z[hf_x_index] = hf.getHeight(hf_x_index,40)
            
            plt.plot(vel_array[:,0],'r')
            plt.plot(vel_array[:,1],'g')
            plt.plot(vel_array[:,2],'b')
            plt.show()

            plt.plot(x,z)
            plt.show()

    # def collectData(self, N = 2):



# Class the drive a kinematic body from a list of operations where each operation is defined
# as a Tuple object that contains a time stamp with linear and angular velocities:
# Operation = namedtuple( "Operation", [ 'time', 'velocity', 'angularVelocity' ] )
class LockDriver(agxSDK.StepEventListener):
    def __init__(self, lockSphere, lock, terrain, shovel, operations):
        super(LockDriver, self).__init__()
        self.lockSphere = lockSphere
        self.lock = lock
        self.bucket = shovel.getRigidBody()
        self.shovel = shovel
        self.terrain = terrain
        self.operations = operations
        self.forceLimit = 5e4
        lock.setEnableComputeForces(True)
        self.sd = application().getSceneDecorator()

    def setBodyVelocity(self, velocity):
        self.lockSphere.setVelocity(self.terrain.getTransform().transformVector(velocity))

    def setBodyAngularVelocity(self, angularVelocity):
        newAngVel = self.terrain.getTransform().transformVector(angularVelocity)
        self.lockSphere.setAngularVelocity(newAngVel)

    def pre(self, t):
        if len(self.operations):
            front = self.operations[0]
            while t >= front.time and len(self.operations):
                velocity = front.velocity
                angular_velocity = front.angularVelocity
                self.setBodyVelocity(velocity)
                self.setBodyAngularVelocity(angular_velocity)
                self.operations.pop(0)
                if len(self.operations):
                    front = self.operations[0]

# Class the drive a kinematic body from a list of operations where each operation is defined
# as a Tuple object that contains a time stamp with linear and angular velocities:
# Operation = namedtuple( "Operation", [ 'time', 'velocity', 'angularVelocity' ] )
class LockForceDriver(agxSDK.StepEventListener):
    def __init__(self, app, lockSphere, lock, hinge, terrain, shovel, operations):
        super(LockForceDriver, self).__init__()
        self.lockSphere = lockSphere
        self.lock = lock
        self.hinge = hinge
        self.bucket = shovel.getRigidBody()
        self.shovel = shovel
        self.terrain = terrain
        self.operations = operations
        self.forceLimit = 5e4
        lock.setEnableComputeForces(True)
        self.sd = app.getSceneDecorator()

        self.v_x_low  = 0.1
        self.v_x_high = 2.0
        self.v_z_low  = -0.2
        self.v_z_high = 0.1

        self.v_x_d = 1.0
        self.v_z_d = 0.0

        self.force  = self.operations[0]
        self.torque = self.operations[1]

        self.t_last_control = 0.0
        self.t_last_setpoint = 0.0

        self.integ_e_x = 0.0
        self.integ_e_z = 0.0


    def setBodyForce(self, force):
        self.lockSphere.setForce(self.terrain.getTransform().transformVector(force))

    def setBodyTorque(self, torque):
        new_torque = self.terrain.getTransform().transformVector(torque)
        self.lockSphere.setTorque(new_torque)

    def measureState(self):

        pos = self.bucket.getPosition()
        vel = self.bucket.getVelocity()
       
        angle = self.hinge.getAngle()
        omega = self.hinge.getCurrentSpeed()

        return pos, vel, angle, omega

    def pre(self, t):

        force  = self.operations[0]
        torque = self.operations[1]
        
        pos, vel, angle, omega = self.measureState()
       
        if (t-self.t_last_setpoint) > 0.4:
            
            self.t_last_setpoint = t
            # generate pseudo-random velocity set point
            self.v_x_d = np.random.uniform(low = self.v_x_low , high = self.v_x_high)
            self.v_z_d = np.random.uniform(low = self.v_z_low , high = self.v_z_high)


        if (t-self.t_last_control) > 0.01:

            self.t_last_control = t

            # PID - VELOCITY CONTROL
            # calculate errors and error integral
            e_v_x = self.v_x_d - vel[0]
            e_v_z = self.v_z_d - vel[2]

            self.integ_e_x += e_v_x
            self.integ_e_z += e_v_z

            # calculate actuation forces
            force[0]  = 20000*(e_v_x) + 1000*self.integ_e_x
            force[2]  = 20000*(e_v_z) + 1000*self.integ_e_z + 10.0*self.bucket.getMassProperties().getMass()
            
            torque[1] = 10000*(-0.5 - angle) -1000*omega

            # # PD - POSITION CONTROL
            # # Setting a random "setpoint" 
            # x_d = -4.75 + t*1.5
            # # Joint PD control
            # force[0]  = 50000*(x_d - pos[0]) - 5000*vel[0]
            # force[2]  = 10.0*self.bucket.getMassProperties().getMass() + 50000*(0.5 -0.1*t - pos[2]) - 5000*vel[2]
            # torque[1] = 10000*(-0.5 - angle) -1000*omega

            self.force  = force
            self.torque = torque

        self.setBodyForce(self.force)
        self.setBodyTorque(self.torque)




    # # Helper functions for constructing the height field surface
    # hfIndexToPosition = lambda index: (index / float(resolution) - 0.5) * size
    # heightField = agxCollide.HeightField(resolution, resolution, size, size)

def main(args):
    agx_sim = AgxSimulator()
    sim, app = agx_sim.createSimulation(args)
    
    sim, app, ter, shov, driver = agx_sim.setScene(sim, app)
    agx_sim.runSimulation(sim, app, ter, shov, driver)
    
    # sim, app, ter, shov, driver = agx_sim.setScene(sim, app)
    # agx_sim.runSimulation(sim, app, ter, shov, driver)
# Entry point when this script is loaded with python
if agxPython.getContext() is None:
    init = agx.AutoInit()
    main(sys.argv)


    # '''

    # terrain.getShovels
    # terrain.getSeparationContactForce
    # '''