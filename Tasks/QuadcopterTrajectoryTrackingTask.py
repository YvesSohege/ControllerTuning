# ====================================
# Trajectory tracking for a single PID configuration
# ====================================
import Tasks.Plants.Quadcopter.quadcopter as quadcopter
import Tasks.Controllers.PID.PID_Controller as controller
import numpy as np
import random
#
# controller1 = []
#
# n = 1
# steps1 = []
# total_steps = [ ]
# starttime = 400
# trajectories = []
# fault_mag = 0.3
# stepsToGoal = 0
# steps = 5
# limit = 7
# x_dest = np.random.randint(-limit, limit)
# y_dest = np.random.randint(-limit, limit)
# z_dest = np.random.randint(5, 7)
#
# x_path = [0, 0, x_dest, x_dest]
# y_path = [0, 0, y_dest,y_dest]
# z_path = [0, 5, z_dest, z_dest]
# interval_steps = 50
# yaws = np.zeros(steps)
# goals = []
# safe_region = []
#

# t = np.linspace(0, , 100)
# x_path = np.linspace(-5, 5, steps)
# y_path = np.linspace(-5, 5, steps)
# z_path = np.linspace(1, 10, steps)

class QuadcopterTrajectoryTrackingTask():


    def __init__(self, controllerConfig , QuadcopterConfig, Env):
        self.quad_id = 1

        self.controllerConfig = controllerConfig
        self.P = controllerConfig['Angular_PID'][0]['P']
        self.I = controllerConfig['Angular_PID'][0]['I']
        self.D = controllerConfig['Angular_PID'][0]['D']
        self.Env = Env
        self.Path = QuadcopterConfig['Path']
        self.goals = []
        self.safezone = []
        self.stepsToGoal = 0
        self.maxSteps = self.Path['maxStepsPerRun']
        self.requiredStableAtGoal = self.Path['stablilizationAtGoal']
        CONTROLLER_PARAMETERS = controllerConfig
        #only tune roll and pitch angular controller - yaw controller is fixed
        CONTROLLER_PARAMETERS['Angular_PID']=  [{'P': [self.P, self.P, 1500],
                                                 'I': [self.I, self.I, 1.2],
                                                 'D': [self.D, self.D, 0]}]

        QUADCOPTER_PARAMETERS = {
            str(self.quad_id) : QuadcopterConfig['Config']
        }

        # Make objects for quadcopter
        self.quad = quadcopter.Quadcopter(QUADCOPTER_PARAMETERS)
        # create blended controller and link it to quadcopter object
        self.ctrl = controller.PID_Controller(self.quad.get_state, self.quad.get_time,
                                                 self.quad.set_motor_speeds, self.quad.get_motor_speeds,
                                                 self.quad.stepQuad, self.quad.set_motor_faults, self.quad.setWind ,self.quad.setNormalWind,
                                                 params=CONTROLLER_PARAMETERS, quad_identifier=str(self.quad_id))


        self.setEnv()

        if bool(self.Path['randomPath']) :
            self.setRandomPath()
        else:
            self.setPath()
        self.ctrl.setSafetyMargin(self.Path['safetyRadius'])

        self.currentWaypoint = 0
        self.ctrl.update_target(self.goals[self.currentWaypoint] , self.safe_region[self.currentWaypoint])

        self.done = False
        self.stepcount= 0
        self.stableAtGoal = 0

    def setEnv(self):

        faultModes = []
        if self.Env == []:
            faultModes = ['None']
        # ===============Rotor Fault config====================
        keys = self.Env.keys()
        if "RotorFault" in keys and bool(self.Env['RotorFault']['enabled']):
            faultModes.append("Rotor")
            fault_mag = self.Env['RotorFault']['magnitude']
            faults = [0, 0, 0, 0]
            if bool(self.Env['RotorFault']['randomRotor']):
                rotor = np.random.randint(0, 4)
            else:
                rotor = self.Env['RotorFault']['faultRotorID']
            faults[rotor] = fault_mag
            self.ctrl.setMotorFault(faults)
            if bool(self.Env['RotorFault']['randomTime']):
                stime = random.randint(200, int(self.maxSteps/2) )
                self.ctrl.setFaultTime(stime, self.Env['RotorFault']['endtime'])
            else:
                self.ctrl.setFaultTime(self.Env['RotorFault']['starttime'],self.Env['RotorFault']['endtime'])

        # ===============Wind gust config=================
        if "Wind" in keys and bool(self.Env['Wind']['enabled']):
            if bool(self.Env['Wind']['randomDirection']):
                direction = np.random.randint(0, 4)
            else:
                direction = self.Env['Wind']['direction']

            WindMag = self.Env['Wind']['magnitude']
            if (direction == 0):
                winds = [-WindMag, 0, 0]
            elif (direction == 1):
                winds = [WindMag, 0, 0]
            elif (direction == 2):
                winds = [0, -WindMag, 0]
            else:
                winds = [0, WindMag, 0]
            faultModes.append("Wind")
            self.ctrl.setNormalWind(winds)
        #============= Position Noise config===============
        if "PositionNoise" in keys and bool(self.Env['PositionNoise']['enabled']):
            faultModes.append("PosNoise")
            posNoise = self.Env['PositionNoise']['magnitude']
            self.ctrl.setSensorNoise(posNoise)
        # ============= Attitude Noise config===============
        if "AttitudeNoise" in keys and bool(self.Env['AttitudeNoise']['enabled']):
            faultModes.append("AttNoise")
            attNoise = self.Env['AttitudeNoise']['magnitude']
            self.ctrl.setAttitudeSensorNoise(attNoise)


        self.ctrl.setFaultMode(faultModes)
        return

    def setRandomPath(self):

        limit = self.Path['randomLimit']
        # np.random.seed(self.Path['randomSeed'])
        x_dest = np.random.randint(-limit, limit)
        y_dest = np.random.randint(-limit, limit)
        z_dest = np.random.randint(5, limit)
        x_dest2 = np.random.randint(-limit, limit)
        y_dest2 = np.random.randint(-limit, limit)
        z_dest2 = np.random.randint(5, limit)
        x_path = [0, 0, x_dest, x_dest2, x_dest2]
        y_path = [0, 0, y_dest, y_dest2, y_dest2]
        z_path = [0, 5, z_dest, z_dest2, z_dest2]
        steps = len(x_path)
        interval_steps = 50
        self.goals = []
        self.safe_region = []
        for i in range(steps):
            if (i < steps - 1):
                # create linespace between waypoint i and i+1
                x_lin = np.linspace(x_path[i], x_path[i + 1], interval_steps)
                y_lin = np.linspace(y_path[i], y_path[i + 1], interval_steps)
                z_lin = np.linspace(z_path[i], z_path[i + 1], interval_steps)
            else:
                x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
                y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
                z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

            self.goals.append([x_path[i], y_path[i], z_path[i]])
            # for each pos in linespace append a goal
            self.safe_region.append([])
            for j in range(interval_steps):
                self.safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
                self.stepsToGoal += 1
        return

    def setPath(self):

        x_path = self.Path['waypoints']['x']
        y_path = self.Path['waypoints']['y']
        z_path = self.Path['waypoints']['z']
        steps = len(x_path)
        interval_steps = 50
        self.goals = []
        self.safe_region = []
        for i in range(steps):
            if (i < steps - 1):
                # create linespace between waypoint i and i+1
                x_lin = np.linspace(x_path[i], x_path[i + 1], interval_steps)
                y_lin = np.linspace(y_path[i], y_path[i + 1], interval_steps)
                z_lin = np.linspace(z_path[i], z_path[i + 1], interval_steps)
            else:
                x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
                y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
                z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

            self.goals.append([x_path[i], y_path[i], z_path[i]])
            # for each pos in linespace append a goal
            self.safe_region.append([])
            for j in range(interval_steps):
                self.safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
                self.stepsToGoal += 1
        return

    def run(self):
        numWaypoints = len(self.goals)
        while not self.done:

            self.stepcount +=1
            self.obs =  self.ctrl.step()


            if(self.stepcount > self.maxSteps):
                self.done = True
                return -self.maxSteps

            if(self.ctrl.isAtPos(self.goals[self.currentWaypoint])):
                self.currentWaypoint += 1 if self.currentWaypoint < numWaypoints - 1 else 0
                #if there is another waypoint then set the quadcopters next goal and safety margin
                if (self.currentWaypoint < numWaypoints - 1):
                    self.ctrl.update_target(self.goals[self.currentWaypoint], self.safe_region[self.currentWaypoint-1])

                else:
                    #there is no more waypoints in the trajectory
                    self.stableAtGoal += 1
                    if(self.stableAtGoal > self.requiredStableAtGoal):
                        #wait for quadcopter to stabilize around goal
                        self.done = True
                    else:
                        self.done = False

            else:
                self.stableAtGoal =0

            if self.ctrl.checkOutsideSafezoneTooLong():
                self.done = True
                return -self.maxSteps

            if self.done:
                return -self.ctrl.getTotalTimeOutside()
