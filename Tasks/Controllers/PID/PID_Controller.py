import numpy as np
import math
import time
import threading
import scipy.stats as stats

class PID_Controller():
    def __init__(self, get_state, get_time, actuate_motors,get_motor_speed,step_quad, set_faults, setWind, setNormWind , params, quad_identifier):
        self.quad_identifier = quad_identifier
        self.actuate_motors = actuate_motors
        self.set_motor_faults = set_faults
        self.get_state = get_state
        self.step_quad = step_quad
        self.get_motor_speed = get_motor_speed
        self.get_time = get_time
        self.setWind = setWind
        self.setNormWind = setNormWind
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0]/180.0)*3.14,(params['Tilt_limits'][1]/180.0)*3.14]
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [self.MOTOR_LIMITS[0]+params['Z_XY_offset'],self.MOTOR_LIMITS[1]-params['Z_XY_offset']]
        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        self.failed = False
        self.goal = True
        #list of controller parameters
        self.ANGULAR_PIDS = params['Angular_PID']

        self.numControllers = len(params['Angular_PID'])

        self.hasLeftBounds = False
        self.PosReward = 0
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.zi_term2 = 0
        self.total_steps = 0
        self.MotorCommands = [0,0,0,0]
        self.thetai_term2 = 0
        self.phii_term2 = 0
        self.gammai_term2 = 0

        self.FaultMode = ["None"]


        self.noiseMag = 0
        self.x_noise = 0
        self.y_noise = 0
        self.z_noise = 0
        self.attNoiseMag = 0
        self.phi_noise = 0
        self.theta_noise = 0
        self.gamma_noise = 0

        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.trajectory = [[0,0,0]]
        self.trackingErrors = { "Pos_err" : 0 , "Att_err" : 0}
        self.startfault = np.random.randint(500,  2000)
        self.endfault = np.random.randint(1500, 3000)
        self.fault_time = [self.startfault,self.endfault]
        self.motor_faults = [0,0,0,0]
        self.current_obs = {}
        self.current_obs["x"] = 0
        self.current_obs["y"] = 0
        self.current_obs["z"] = 0
        self.current_obs["phi"] = 0
        self.current_obs["theta"] = 0
        self.current_obs["gamma"] = 0
        self.current_obs["x_err"] = 0
        self.current_obs["y_err"] = 0
        self.current_obs["z_err"] = 0
        self.current_obs["phi_err"] = 0
        self.current_obs["theta_err"] = 0
        self.current_obs["gamma_err"] = 0


        self.blendDist = (5,5,5)

        self.PosblendDist = (5,5,5)

        self.blends = [0]
        self.current_blend = np.random.dirichlet(self.blendDist, 1).transpose()
        self.current_pos_blend= np.random.dirichlet(self.PosblendDist, 1).transpose()
        self.current_waypoint = -1

        self.safe_bound = []
        self.time_outside_safety = 0
        self.total_time_outside_safety = 0
        self.current_distance_to_opt = 0
        self.safety_margin = 1


        self.trackingAccuracy = 0.5
        self.thread_object = None
        self.target = [0,0,0]
        self.yaw_target = 0.0
        self.run = True


        self.min_distances_points= []
        self.min_distances = []


    def setSafetyMargin(self, newMargin):
        self.safety_margin = newMargin
        return

    def wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def setFaultTime(self,low,high):
        self.startfault = low
        self.endfault = high
        self.fault_time = [self.startfault, self.endfault]

    def setBlendWeight(self, new_weight):
        self.current_blend = new_weight

    def setBlendDist(self,params):

        dirichletParams=()

        for param in params:
            dirichletParams.append(param)

        self.blendDist = dirichletParams


    def setPosBlendDist(self, params):

        dirichletParams = ()

        for param in params:
            dirichletParams.append(param)

        self.PosblendDist = dirichletParams


    def getUniformBlend(self):
        self.current_blend = np.random.dirichlet((5,5,5), 1).transpose()
        return self.current_blend

    def nextBlendWeight(self):
        self.current_blend = np.random.dirichlet(self.blendDist, 1).transpose()

    def nextPosBlendWeight(self):
        self.current__pos_blend = np.random.dirichlet(self.PosblendDist, 1).transpose()


    def getBlendWeight(self):
        return self.current_blend

    def getPosBlendWeight(self):
        return self.current_pos_blend

    def getBlends(self):
        return self.blends

    def setMotorCommands(self , cmds):
        self.MotorCommands = cmds

    def getMotorCommands(self):
        m1 = self.MotorCommands[0]
        m2 = self.MotorCommands[1]
        m3 = self.MotorCommands[2]
        m4 = self.MotorCommands[3]

        return m1, m2, m3, m4

    def setFaultMode(self, mode):
        # [ 'None' , 'Rotor' , 'Wind', 'PosNoise', 'AttNoise']
        self.FaultMode = mode

    def update(self):
        self.total_steps += 1

        self.checkSafetyBound()

        [dest_x,dest_y,dest_z] = self.target
        [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.get_state(self.quad_identifier)



        self.x_noise = np.random.uniform(-self.noiseMag, self.noiseMag)
        self.y_noise = np.random.uniform(-self.noiseMag, self.noiseMag)
        self.z_noise = np.random.uniform(-self.noiseMag, self.noiseMag)

        self.theta_noise = np.random.uniform(-self.attNoiseMag, self.attNoiseMag)
        self.phi_noise = np.random.uniform(-self.attNoiseMag, self.attNoiseMag)
        self.gamma_noise = np.random.uniform(-self.attNoiseMag, self.attNoiseMag)

        self.trajectory.append([x, y, z])

        #print(" X: "+ str(x) + " Y: "+ str(y) +" Z:"+ str(z))
        #print(" Dest X: "+ str(x) + " Y: "+ str(y) +" Z:"+ str(z))
        #print(" Dest: "+ str(dest_x) + " "+ str(dest_y) +" "+ str(dest_z))
        if ( "PosNoise" in self.FaultMode ):
            x = x + self.x_noise
            y = y + self.y_noise
            z = z + self.z_noise
            # print("Ctrl Pos noise ")
        if( "AttNoise" in self.FaultMode):
            theta = theta + self.theta_noise
            phi   = phi + self.phi_noise
            gamma = gamma + self.gamma_noise
            # print("Ctrl Att noise ")


        x_error = dest_x-x
        y_error = dest_y-y
        z_error = dest_z-z


        #print("Pos Errors: X= " + str(x_error) +" Y= " +str(y_error )+ " Z=" + str(z_error))
        self.xi_term += self.LINEAR_I[0]*x_error
        self.yi_term += self.LINEAR_I[1]*y_error
        self.zi_term += self.LINEAR_I[2]*z_error

        dest_x_dot = self.LINEAR_P[0]*(x_error) + self.LINEAR_D[0]*(-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1]*(y_error) + self.LINEAR_D[1]*(-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot) + self.zi_term



        throttle = np.clip(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])


        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))

        # --------------------
        #get required attitude states
        dest_gamma = self.yaw_target
        dest_theta,dest_phi = np.clip(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1]),np.clip(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1])

        theta_error = dest_theta-theta
        phi_error = dest_phi-phi
        gamma_dot_error = (self.YAW_RATE_SCALER*self.wrap_angle(dest_gamma-gamma)) - gamma_dot

        self.trackingErrors["Pos_err"] += (abs(round(x_error, 2)) + abs(round(y_error, 2)) + abs(round(z_error, 2)))
        self.trackingErrors["Att_err"] += (abs(round(phi_error,2)) + abs(round(theta_error,2)) + abs(round(dest_gamma-gamma, 2)))

        p = self.ANGULAR_PIDS[0]['P'][0]
        i = self.ANGULAR_PIDS[0]['I'][0]
        d = self.ANGULAR_PIDS[0]['D'][0]

        x_val = (p * (theta_error)) + (i * (-theta_dot))+ (d * (-theta_dot))
        y_val = (p * (phi_error) + (i * (-phi_error))+ d * (-phi_dot))
        z_val = (p * (gamma_dot_error))
        z_val = np.clip(z_val, self.YAW_CONTROL_LIMITS[0], self.YAW_CONTROL_LIMITS[1])




        #calculate motor commands depending on controller selection

        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val


        #print("X_val 1 =" + str(x_val) + " 2= " +  str(x_val2 ) + " blended = " + str(x_val_blend))
        #print("Y_val 1 =" +  str(y_val) + " 2= " +  str(y_val2 )+ " blended = " + str(y_val_blend))
        #print("Z_val 1 =" +  str(z_val) + " 2= " +  str(z_val2) + " blended = " + str(z_val_blend))


       # [m1, m2, m3, m4] = self.getMotorCommands()
        M = np.clip([m1,m2,m3,m4],self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])


        #check for rotor fault to inject to quad
        if ( "Rotor" in self.FaultMode):
            if (self.fault_time[0] <= self.total_steps and self.fault_time[1] >= self.total_steps):
                # print("Fault at time step " + str(self.total_steps))
                self.setQuadcopterMotorFaults()
            else:
                #print("time step " + str(self.total_steps))
                self.clearQuadcopterMotorFaults()
            # print("Ctrl rotor faults")

        #if( "Wind" in self.FaultMode):
            # if(self.total_steps % 100 == 0):
            #     randWind = np.random.normal(0, 10, size=3)
            #     self.setWind(randWind)
            # print("Ctrl wind faults")
            #print()


        self.actuate_motors(self.quad_identifier,M)



        #step the quad to the next state with the new commands
        self.step_quad(0.01)
        new_obs = self.get_updated_observations()
        #print(new_obs)
        #update the current observations and return


        return new_obs


    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def update_target(self,target,new_safety_bound):
        self.current_waypoint +=1
        self.target = target
        self.safe_bound = new_safety_bound
        self.time_outside_safety = 0
        self.current_distance_to_opt = self.getDistanceToOpt()

    def getCurrentSafeBounds(self):
        return self.safe_bound

    def getTotalTimeOutside(self):
        return self.total_time_outside_safety

    def getLatestMinDistPoint(self):
        return self.min_distances_points[-1]

    def getLatestMinDist(self):
        return self.min_distances[-1]

    def update_yaw_target(self,target):
        self.yaw_target = self.wrap_angle(target)

    def getTrackingErrors(self):
        err_array = []
        print(self.trackingErrors)
        for key, value in self.trackingErrors.items():
            err_array.append((value/self.total_steps))
        return err_array

    def get_updated_observations(self):
        #update the current observation after taking an action and progressing the quad state
        #[dest_x, dest_y, dest_z] = self.target
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state(
            self.quad_identifier)


        #change the states observed by the agent
        [dest_x, dest_y, dest_z] = self.target
        obs = [x, y, z, theta, phi, gamma, theta_dot, phi_dot, gamma_dot, dest_x, dest_y, dest_z ]
        # 12 state variables
        # obs = [x, y, z, theta, phi, gamma,  dest_x, dest_y, dest_z ]

        return obs

    def set_action(self,action):


        # attitudeBlend = [action[0], action[1]]
        # self.setBlendDist(attitudeBlend)

        # roll = action[0]
        # pitch = action[1]
        # actionb = [roll , pitch , 0]
        # self.setBlendWeight(actionb)

        #self.setMotorCommands(action)
        #self.updateAngularPID(action)
        #self.total_steps += 1
        obs = self.update()
        # obs_array = []
        # for key, value in obs.items():
        #     obs_array.append(value)
        # return obs_array
        return obs

    def step(self):
        obs = self.update()
        #print(obs)
        # obs_array = []
        # for key, value in obs.items():
        #     obs_array.append(value)
        # return obs_array
        return obs

    def isAtPos(self,pos):
        [dest_x, dest_y, dest_z] = pos
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state(
            self.quad_identifier)
        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z
        total_distance_to_goal = abs(x_error) + abs(y_error) + abs(z_error)

        isAt = True if total_distance_to_goal < self.trackingAccuracy else False
        if isAt:
            self.PosReward = 500
            #print("Reached goal +500 in mode " + str(self.FaultMode))
        else:
            self.PosReward = 0
        return isAt

    def isDone(self):
        #checks if the agent has ever left the safespace
        print(self.total_time_outside_safety)
        limit =50
        if self.total_time_outside_safety > limit:
            return False
        else:
            return True

    def getDistanceToOpt(self):

        #get closest point on the linspace between waypoints
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state(
            self.quad_identifier)

        p1 =[x,y,z]
        distances = []
        points = []
        for i in range(len(self.safe_bound)):

            p2 = np.array([self.safe_bound[i][0], self.safe_bound[i][1], self.safe_bound[i][2]])
            squared_dist = np.sum((p1 - p2) ** 2, axis=0)
            dist = np.sqrt(squared_dist)
            distances.append(dist)
            points.append(p2)
            #print(str(self.safe_bound[i]) +" "+ str(dist))

        i = np.where(distances == np.amin(distances))
        index = i[0][0]
        #print("min pos index" + str(index) + " dist " + str(distances[index]) + " point " + str(points[index]))
        self.min_distances_points.append(points[index] )
        self.min_distances.append(distances[index])
       # print("min dist point : " +  str(min_dist_point))
        return distances[index]

    def getMinDistances(self):
        return self.min_distances

    def checkSafetyBound(self):
        self.current_distance_to_opt = self.getDistanceToOpt()
       # print(self.current_distance_to_opt)
        if  self.current_distance_to_opt > self.safety_margin :
            #increase total time outside safety bound by 1 ( calculated per step)
            self.time_outside_safety += 1
           # print(self.current_distance_to_opt)
            self.total_time_outside_safety += 1
            self.outsideBounds = True
            self.hasLeftBounds = True
        else:
            self.time_outside_safety = 0
            self.outsideBounds = False

        return

    def getReward(self):
        end_threshold = 3000

        if (self.outsideBounds):
            #left safety region give negative reward
            reward = -1
            #print("outside")
        else:
            reward = 0
        # #print(reward)
        #
        # #reward += self.PosReward
        #
        # #limit = 3
        if self.total_steps > end_threshold:
            reward = -0.1
           # print("Failed epsidoe steps:" + str(self.total_steps))
        # #
        #
        limit = 500
        if(self.hasLeftBounds and self.total_time_outside_safety > limit):
            #print("left flight area - aborting" + str(self.total_time_outside_safety) )
            reward = -0.1

        return reward

    def checkOutsideSafezoneTooLong(self,end_threshold=3000 ):


        if self.total_steps > end_threshold:
            return True

        limit = 500
        if (self.hasLeftBounds and self.total_time_outside_safety > limit):
            # print("left flight area - aborting" + str(self.total_time_outside_safety) )
            return True

        return False

    def thread_run(self,update_rate,time_scaling):
        update_rate = update_rate*time_scaling
        last_update = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self,update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    def updateAngularPID(self, PID):

        self.ANGULAR_P[0] = PID[0] # P roll term
        self.ANGULAR_P[1] = PID[0] # P pitch term (same)
        self.ANGULAR_P[2] = PID[1] # P yaw term (different)

        self.ANGULAR_I[0] = PID[2] # I term roll
        self.ANGULAR_I[1] = PID[2]# I term pitch
        self.ANGULAR_I[2] = PID[3] # I term yaw

        self.ANGULAR_D[0] =PID[4]
        self.ANGULAR_D[1] =PID[4]
        self.ANGULAR_D[2] =PID[5]

        return

    def setMotorFault(self, fault):

        self.motor_faults = fault
        #should be 0-1 value for each motor

    def setQuadcopterMotorFaults(self):
        self.set_motor_faults(self.quad_identifier,self.motor_faults)
        return

    def clearQuadcopterMotorFaults(self):

        self.set_motor_faults(self.quad_identifier, [0,0,0,0])
        return

    def setNormalWind(self,winds):
        self.setNormWind( winds)

    def setSensorNoise(self,noise):
        self.noiseMag = noise

    def setAttitudeSensorNoise(self,noise):
        self.attNoiseMag = noise

    def setWindGust(self,wind):
        self.wind = wind

    def thread_run(self,update_rate,time_scaling):
        update_rate = update_rate*time_scaling
        last_update = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self,update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    def getTrajectory(self):

        return self.trajectory

    def getTotalSteps(self):
        return self.total_steps

