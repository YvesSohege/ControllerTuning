# ====================================
# Trajectory tracking for a single PID configuration
# ====================================
import quadcopter,controller
import numpy as np
import gui
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D


controller1 = []

n = 1
steps1 = []
total_steps = [ ]
starttime = 400
trajectories = []
fault_mag = 0.3
stepsToGoal = 0
steps = 5
limit = 7
x_dest = np.random.randint(-limit, limit)
y_dest = np.random.randint(-limit, limit)
z_dest = np.random.randint(5, 7)

x_path = [0, 0, x_dest, x_dest]
y_path = [0, 0, y_dest,y_dest]
z_path = [0, 5, z_dest, z_dest]
interval_steps = 50
yaws = np.zeros(steps)
goals = []
safe_region = []


# t = np.linspace(0, , 100)
# x_path = np.linspace(-5, 5, steps)
# y_path = np.linspace(-5, 5, steps)
# z_path = np.linspace(1, 10, steps)

class ParticleQuad():
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    #metadata = {'render.modes': ['console']}
    # Define constants for clearer code


    def __init__(self, p, i, d):
        self.quad_id = 1

        BLENDED_CONTROLLER_PARAMETERS = {'Motor_limits': [0, 9000],
                                         'Tilt_limits': [-10, 10],
                                         'Yaw_Control_Limits': [-900, 900],
                                         'Z_XY_offset': 500,
                                         'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5],
                                                        'D': [450, 450, 5000]},
                                         'Linear_To_Angular_Scaler': [1, 1, 0],
                                         'Yaw_Rate_Scaler': 0.18,
                                         'Angular_PIDS': [{'P': [p, p, 1500], 'I': [i, i, 1.2],
                                                         'D': [d, d, 0]}]
                                         }


        QUADCOPTER = {
            str(self.quad_id): {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                           'weight': 1.2}}

        # Make objects for quadcopter

        self.quad = quadcopter.Quadcopter(QUADCOPTER)
        Quads = {str(self.quad_id): QUADCOPTER[str(self.quad_id)]}

        self.config = [int(p), int(i), int(d)]


        # create blended controller and link it to quadcopter object
        self.ctrl = controller.PID_Controller(self.quad.get_state, self.quad.get_time,
                                                 self.quad.set_motor_speeds, self.quad.get_motor_speeds,
                                                 self.quad.stepQuad, self.quad.set_motor_faults, self.quad.setWind ,self.quad.setNormalWind,
                                                 params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(self.quad_id))

        #self.gui_object = gui.GUI(quads=QUADCOPTER, ctrl=self.ctrl)
        #

        self.setEasyPath()
        self.current = 0
        self.ctrl.update_target(goals[self.current] , safe_region[self.current])

        self.ctrl.setController("PID")
        self.done = False
        self.stepcount= 0
        self.stableAtGoal = 0


    def setEnv(self,envs,magnitudes):
        #envs = ["Rotor", "Wind"]
        #mag = [0,0,0,0]
        faultModes = []
        #("Setting Env " + str(envs)  + " " + str(magnitudes))
        if envs == []:
            faultModes = ['None']

        # ===============Rotor Fault config====================
        RotorScalar= 0.075
        starttime = 500
        endtime = 31000

        if "Rotor" in envs:
            faultModes.append("Rotor")
            fault_mag = magnitudes[0] * RotorScalar
            faults = [0, 0, 0, 0]
            rotor = np.random.randint(0, 4)
            faults[rotor] = fault_mag
            # print(faults)

            # print(faults)
            self.ctrl.setMotorFault(faults)
            self.ctrl.setFaultTime(starttime, endtime)

        # ===============Wind gust config=================
        WindScalar = 3
        WindMag = magnitudes[1] * WindScalar
        direction = np.random.randint(0, 4)
        # magnitude = 4
        if (direction == 0):
            winds = [-WindMag, 0, 0]
        elif (direction == 1):
            winds = [WindMag, 0, 0]
        elif (direction == 2):
            winds = [0, -WindMag, 0]
        else:
            winds = [0, WindMag, 0]
        if "Wind" in envs:
            faultModes.append("Wind")
            self.ctrl.setNormalWind(winds)


        #============= Noise config===============
        AttScalar  = 0.15
        PosScalar  = 0.9

        if "PosNoise" in envs:
            faultModes.append("PosNoise")
            posNoise = magnitudes[2] * PosScalar
            self.ctrl.setSensorNoise(posNoise)

        if "AttNoise" in envs:
            faultModes.append("AttNoise")
            attNoise = magnitudes[3] * AttScalar

            self.ctrl.setAttitudeSensorNoise(attNoise)


        self.ctrl.setFaultMode(faultModes)






    def run(self):
        global steps, stableAtGoal
        maxSteps = 3000
        performanceScalar = 50
        while not self.done:

            self.stepcount +=1
            self.obs =  self.ctrl.step()

            # # print("C 1:" + str(ctrl.getTotalSteps()))
            # if(self.stepcount%20==0):
            #     self.gui_object.quads[str(self.quad_id)]['position'] = [self.obs[0],self.obs[1],self.obs[2]]
            #     self.gui_object.update()

                #print(" ")
            if(self.stepcount > maxSteps):
                self.done = True
                # self.gui_object.close()
                #return self.current * performanceScalar
                return -3000

            if( self.ctrl.isAtPos(goals[self.current])):

                if self.current < steps - 1:
                    self.current += 1
                else:
                    self.current += 0

                if (self.current < steps - 1):
                    self.ctrl.update_target(goals[self.current], safe_region[self.current-1])
                    if(self.current < steps-1):
                        #gui_object.updatePathToGoal()
                        pass
                    #print("At pos " + str(goals[self.current-1]) + " next goal " + str(goals[self.current]))
                else:

                    self.stableAtGoal += 1
                    if(self.stableAtGoal > 100):

                        self.done = True
                    else:

                        self.done = False


                    if (self.done):
                        #return the number of steps to complete the trajectory.
                        #self.gui_object.close()

                        return -self.ctrl.getTotalTimeOutside()

            else:
                self.stableAtGoal =0

            if self.ctrl.getReward() == -0.1 :

                #self.gui_object.close()
                self.done = True
                #print("out of bounds")
                return -3000


    def setEasyPath(self):
        global stepsToGoal, steps, x_dest, y_dest, x_path, y_path, z_path, goals, safe_region, limit

        stepsToGoal = 0
        seed = self.config[0] + self.config[1] + self.config[2]
        limit = 8
        np.random.seed(seed)
        x_dest = np.random.randint(-limit, limit)
        y_dest = np.random.randint(-limit, limit)
        z_dest = np.random.randint(5, limit)
        x_dest2 = np.random.randint(-limit, limit)
        y_dest2 = np.random.randint(-limit, limit)
        z_dest2 = np.random.randint(5, limit)


        x_path = [0, 0, x_dest, x_dest2, x_dest2]
        y_path = [0, 0, y_dest, y_dest2, y_dest2]
        z_path = [0, 5, z_dest, z_dest2, z_dest2]
        # x_path = [0, 0, x_dest, x_dest]
        # y_path = [0, 0, y_dest, y_dest]
        # z_path = [0, 5, z_dest, z_dest]
        #
        # x_path = [0, 0, 5, 0, -5, 0, 5, 5]
        # y_path = [0, 0, 0, 5, 0, -5, 0, 0]
        # z_path = [0, 5, 5, 5, 5, 5, 5, 5]

        steps = len(x_path)

        interval_steps = 50
        yaws = np.zeros(steps)
        goals = []
        safe_region = []
        for i in range(steps):
            if(i < steps-1 ):
                #create linespace between waypoint i and i+1
                x_lin = np.linspace(x_path[i], x_path[i+1], interval_steps)
                y_lin =  np.linspace(y_path[i], y_path[i+1], interval_steps)
                z_lin =  np.linspace(z_path[i], z_path[i+1], interval_steps)
            else:
                x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
                y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
                z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

            goals.append([x_path[i], y_path[i], z_path[i]])
            #for each pos in linespace append a goal
            safe_region.append([])
            for j in range(interval_steps):
                safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
                stepsToGoal +=1



#
#
# kp = 24000
# ki = 0
# kd = 12000
#
# PQ = ParticleQuad(kp,ki,kd)
# Domain = []
# randRotor = np.random.randint(0, 4)
# randWind = np.random.randint(0, 4)
# randPos = np.random.randint(0, 4)
# randAtt = np.random.randint(0, 4)
#
# magSetting = [randRotor, randWind, randPos, randAtt]
# magSetting = [0, 0, 0, 0]
# PQ.setEnv(Domain, magSetting)
#
# performance = PQ.run()
# print(kp,ki,kd)
# print(performance)