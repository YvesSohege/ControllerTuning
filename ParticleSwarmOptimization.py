from Tasks.QuadcopterTrajectoryTrackingTask import QuadcopterTrajectoryTrackingTask as QTTT
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import os
class ParticleSwarmOptimization():

    def __init__(self, PSOConfig, controllerConfig, plantConfig, databaseConfig, render=False):

        self.PSOConfig = PSOConfig
        self.controllerConfig = controllerConfig
        self.plantConfig = plantConfig
        self.databaseConfig = databaseConfig
        if bool(self.databaseConfig['newFile']):
            dt = str(datetime.datetime.now())[:16].replace(":","-")
            FileName = self.databaseConfig['folder']+"PSO_Results-"+dt+".json"
            print(FileName)
            if not os.path.exists(FileName):
                with open(FileName, 'w') as db:
                    json.dump([], db)

            self.databaseConfig['path'] = FileName
        self.database = []
        self.Environment = self.plantConfig['environment']
        self.Params = []
        #internal swarm representation
        self.particles = [0] * self.PSOConfig['NumberParticles']
        self.velocities = [0] * self.PSOConfig['NumberParticles']
        self.localBests = [0] * self.PSOConfig['NumberParticles']
        self.globalBest = None
        self.render = render
        if self.render:
            self.fig = plt.figure(figsize=(6,6))
            self.ax = plt.subplot(111)
            self.particlePaths = [[]] * self.PSOConfig['NumberParticles']
            self.ax.set_xlim(self.PSOConfig['ParameterRanges']["P"]['min'], self.PSOConfig['ParameterRanges']["P"]['max'])
            self.ax.set_ylim(self.PSOConfig['ParameterRanges']["D"]['min'], self.PSOConfig['ParameterRanges']["D"]['max'])
            # self.ax.set_zlim(self.PSOConfig['ParameterRanges']["D"]['min'], self.PSOConfig['ParameterRanges']["D"]['max'])
        self.initilizeParticles()

        self.finished = False
        return

    def initilizeParticles(self):

        for i in range(self.PSOConfig['NumberParticles']):

            numParameters = 0
            params = []
            for param in self.PSOConfig['ParameterRanges']:
                max = self.PSOConfig['ParameterRanges'][param]['max']
                min = self.PSOConfig['ParameterRanges'][param]['min']
                if max - min > 0:
                    numParameters+=1
                    params.append(param)

            self.Params = params
            particleParams =   [0]* numParameters
            particleVelocity = [0]* numParameters

            paramNumber = 0
            for param in params:
                Max = self.PSOConfig['ParameterRanges'][param]['max']
                Min = self.PSOConfig['ParameterRanges'][param]['min']
                vel =self.PSOConfig['maxVelocity']
                if max - min > 0:
                    particleParams[paramNumber] = np.random.uniform(Min,Max)
                    particleVelocity[paramNumber] = np.random.uniform(-vel,vel)
                    paramNumber+=1


            self.particles[i] = particleParams
            self.particlePaths[i] = [particleParams]
            self.velocities[i] = particleVelocity
            # performance = self.evaluateParticle(particleParams)
            # performance = self.evaluateParticle(particleParams)
            self.localBests[i] = (particleParams, -(self.plantConfig['Path']['maxStepsPerRun']+1))
            # if self.globalBest == None or performance > self.globalBest[1]:
        self.globalBest = (self.particles[i],-(self.plantConfig['Path']['maxStepsPerRun']+1)) #

        if self.render:
            self.render3D()
        return

    def evaluateParticle(self, controller):

        if self.controllerConfig['type'] == 'PID':

            controllerParameters = self.controllerConfig['Config']
            controllerParameters['Angular_PID'][0]['P'] = controller[0]
            controllerParameters['Angular_PID'][0]['I'] = 0
            controllerParameters['Angular_PID'][0]['D'] = controller[1]

            Task = QTTT(controllerParameters,self.plantConfig, self.Environment)

            particlePerformance = Task.run()
            # print(str(controller) + " " + str(particlePerformance))
            self.saveResult(controllerParameters, particlePerformance)
            return  particlePerformance

    def saveResult(self, controllerParameters, particlePerformance):
        datapoint = {'ControllerParameters': controllerParameters,
                     'Environment': self.Environment,
                     'performance': particlePerformance}
        self.database.append(datapoint)
        # print(self.database)
        data = []
        with open(self.databaseConfig['path'], 'r') as db:
            try:
                data = json.load(db)
            except Exception:
                data = []
        data.append(datapoint)
        # print(len(self.database))
        # print(len(data))
        with open(self.databaseConfig['path'], 'w') as db:
            json.dump(data, db)
        return

    def run(self):
        print("PSO running")
        currentWeight = self.PSOConfig['w']
        while not self.finished:
            for i in range(len(self.particles)):

                pos = self.particles[i]

                performance = self.evaluateParticle(pos)

                # print(str(pos) + " fitness: " + str(fitness))
                if performance > self.localBests[i][1]:
                    self.localBests[i] = (pos, performance)

                if performance > self.globalBest[1]:
                    self.globalBest = (pos, performance)

            # update interia
            for i in range(len(self.particles)):
                r1 = 1
                r2 = np.random.uniform(0.0, 1.0) * self.PSOConfig['c1']
                r3 = np.random.uniform(0.0, 1.0) * self.PSOConfig['c2']

                InteriaComp =  [ min(max(r1*v,-self.PSOConfig['maxVelocity']), self.PSOConfig['maxVelocity']) for v in self.velocities[i] ] #  [r1 * self.velocities[i][0], r1 * self.velocities[i][1]]

                CognitiveComp = [0]*len(self.particles[i])
                for j in range(len(self.localBests[i][0])):
                    localBest = float(self.localBests[i][0][j])
                    particleParam = float( self.particles[i][j])
                    cognitiveVelocity = localBest - particleParam
                    cog = min(max(r2*(cognitiveVelocity),
                                  -self.PSOConfig['maxVelocity']), self.PSOConfig['maxVelocity'])
                    CognitiveComp[j] = cog


                SocialComp = [0] * len(self.particles[i])

                for k in range(len(self.globalBest[0])):
                    globalBest = float(self.globalBest[0][k])
                    particleParam = float(self.particles[i][k])
                    socialVelocity = globalBest - particleParam
                    soc = min(max(r3 * (socialVelocity),
                                  -self.PSOConfig['maxVelocity']), self.PSOConfig['maxVelocity'])
                    SocialComp[k] = soc


                newVel = [0] * len(self.particles[i])
                print(self.particles[i])
                for param in range(len(newVel)):
                    newVel[param] = (InteriaComp[param] + CognitiveComp[param] + SocialComp[param])* currentWeight

                self.velocities[i] = newVel

                newPos = [0] * len(self.Params)

                p = 0
                for param in self.Params:
                    minimum = self.PSOConfig['ParameterRanges'][param]['min']
                    maximum = self.PSOConfig['ParameterRanges'][param]['max']
                    if maximum - minimum > 0:
                        parameterPosition = self.particles[i][p]
                        parameterVelocity = self.velocities[i][p]
                        newPosition = parameterPosition + parameterVelocity
                        newPos[p]= min(max(newPosition,minimum),maximum)
                        p+=1

                self.particles[i] = newPos
                self.particlePaths[i].append(self.particles[i])
            print("Global Best = " + str(self.globalBest) + " at " + str(currentWeight))

            if currentWeight > self.PSOConfig['Wmin']:
                currentWeight -= 0.01
            else:
                self.finished = True

            if self.render:
                self.render()
        return self.globalBest

    def render(self):
        if self.render:
            self.ax.cla()

            xdata = []
            ydata = []
            # zdata = []
            sizes = []
            colors = []
            for i in range(self.PSOConfig['NumberParticles']):
                xdata.append(self.particles[i][0])
                ydata.append(self.particles[i][1])
                # zdata.append(self.particles[i][2])
                sizes.append(50)

            if len(self.particlePaths[i]) > 1:
                for particleHistory in self.particlePaths:
                    xPath = []
                    yPath = []
                    # zPath = []
                    for point in particleHistory:
                        xPath.append(point[0])
                        yPath.append(point[1])
                        # zPath.append(point[2])

                    # self.ax.plot(xPath, yPath, zPath)
                    self.ax.plot(xPath, yPath)

            # self.ax.scatter(xdata, ydata, zdata, s=sizes, alpha=0.8, c='k', edgecolor="k", linewidth=0.4)
            self.ax.scatter(xdata, ydata, s=sizes, alpha=0.8, c='k', edgecolor="k", linewidth=0.4)
            self.ax.set_xlim(self.PSOConfig['ParameterRanges']["P"]['min'],
                             self.PSOConfig['ParameterRanges']["P"]['max'])
            self.ax.set_ylim(self.PSOConfig['ParameterRanges']["D"]['min'],
                             self.PSOConfig['ParameterRanges']["D"]['max'])
            # self.ax.set_zlim(self.PSOConfig['ParameterRanges']["D"]['min'],
            #                  self.PSOConfig['ParameterRanges']["D"]['max'])

            plt.draw()

            plt.pause(0.0001)
            # plt.savefig("PSO/" + str(renderCount) + ".png")

        return
