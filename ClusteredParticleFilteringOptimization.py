from Tasks.QuadcopterTrajectoryTrackingTask import QuadcopterTrajectoryTrackingTask as QTTT
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import datetime
import os
import matplotlib.gridspec as gridspec

class ClusteredParticleFilteringOptimization():

    def __init__(self, CPFConfig, controllerConfig, plantConfig, databaseConfig, render=False):

        self.CPFConfig = CPFConfig
        self.controllerConfig = controllerConfig
        self.plantConfig = plantConfig
        self.databaseConfig = databaseConfig
        if bool(self.databaseConfig['newFile']):
            dt = str(datetime.datetime.now())[:16].replace(":","-")
            FileName = self.databaseConfig['folder']+"CPF_Results-"+dt+".json"
            print(FileName)
            if not os.path.exists(FileName):
                with open(FileName, 'w') as db:
                    json.dump([], db)

            self.databaseConfig['path'] = FileName

        self.database = []
        self.Environment = self.plantConfig['environment']
        self.TaskEnv = []
        #internal swarm representation


        self.IntermediateHulls = []
        self.FinalHull = []

        self.numSamples = self.CPFConfig['numberOfParticlesToSample']
        # equal probability for all particles at the start



        # self.initilizeParticles()

        self.render = render
        if self.render:
            self.fig = plt.figure(figsize=(6,6))
            self.ax = plt.subplot(111)
            # self.ax1 = plt.subplot(222)
            self.ax.set_xlim(self.CPFConfig['ParameterRanges']["P"]['min'], self.CPFConfig['ParameterRanges']["P"]['max'])
            # self.ax.set_ylim(self.CPFConfig['ParameterRanges']["I"]['min'], self.CPFConfig['ParameterRanges']["I"]['max'])
            self.ax.set_ylim(self.CPFConfig['ParameterRanges']["D"]['min'], self.CPFConfig['ParameterRanges']["D"]['max'])
            # self.ax1.set_xlim(self.CPFConfig['ParameterRanges']["P"]['min'], self.CPFConfig['ParameterRanges']["P"]['max'])
            # # self.ax1.set_ylim(self.CPFConfig['ParameterRanges']["I"]['min'], self.CPFConfig['ParameterRanges']["I"]['max'])
            # self.ax1.set_ylim(self.CPFConfig['ParameterRanges']["D"]['min'], self.CPFConfig['ParameterRanges']["D"]['max'])

        self.finished = False
        return

    def initilizeParticles(self):

        self.particles = []
        self.performances = []
        self.avgPerformances = []
        self.weights = []
        self.totalPerformance = 0
        self.numberRuns = 0
        self.clusterPoints = []

        startPerf =  self.CPFConfig['StartingPerformance']
        numParticles = self.CPFConfig['NumberParticlesPerDimension']
        P_min =self.CPFConfig['ParameterRanges']["P"]['min']
        P_max =self.CPFConfig['ParameterRanges']["P"]['max']
        P_step = round((P_max-P_min)/numParticles)
        # I_min =self.CPFConfig['ParameterRanges']["I"]['min']
        # I_max =self.CPFConfig['ParameterRanges']["I"]['max']
        # I_step =  round((I_max - I_min) / numParticles)
        D_min =self.CPFConfig['ParameterRanges']["D"]['min']
        D_max =self.CPFConfig['ParameterRanges']["D"]['max']
        D_step = round((D_max - D_min) / numParticles)
        for p in range(P_min, P_max, P_step):
            # for i  in range(I_min, I_max, I_step):
                for d  in range(D_min, D_max, D_step):
                    # self.particles.append((p, i, d))
                    self.particles.append((p, d))
                    self.performances.append([startPerf])
                    self.avgPerformances.append(startPerf)

        for particle in self.particles:
            self.weights.append(1 / len(self.particles))

        return


    def setTaskEnv(self,env):
        self.TaskEnv = self.Environment[env]
        return

    def setRandomEnvironmentMagnitude(self):
        # self.TaskEnv = self.Environment[env]
        print(self.TaskEnv)
        min = self.TaskEnv['min_magnitude']
        max = self.TaskEnv['max_magnitude']
        self.TaskEnv['magnitude'] = random.uniform(min,max)

    def setFixedEnvironment(self, percent):
        min = self.TaskEnv['min_magnitude']
        max = self.TaskEnv['max_magnitude']
        val = min+ ((max - min)*percent)
        self.TaskEnv['magnitude'] = val
        return

    def evaluateParticle(self, controller):

        if self.controllerConfig['type'] == 'PID':

            controllerParameters = self.controllerConfig['Config']
            controllerParameters['Angular_PID'][0]['P'] = controller[0]
            # controllerParameters['Angular_PID'][0]['I'] = controller[1]
            controllerParameters['Angular_PID'][0]['I'] = 0
            # controllerParameters['Angular_PID'][0]['D'] = controller[2]
            controllerParameters['Angular_PID'][0]['D'] = controller[1]

            Task = QTTT(controllerParameters,self.plantConfig, self.TaskEnv )

            particlePerformance = Task.run()
            print(str(controller) + " " + str(particlePerformance))
            self.saveResult(controllerParameters, particlePerformance)
            return  particlePerformance

    def saveResult(self, controllerParameters, particlePerformance):
        datapoint = {'ControllerParameters': controllerParameters,
                     'Environment': self.Environment,
                     'performance': particlePerformance}
        self.database.append(datapoint)

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

    def sampleAndEvaluateParticles(self):
        sampleInd = np.random.choice(len(self.particles), self.numSamples, p=self.weights)

        for ind in sampleInd:
            particle = self.particles[ind]
            # sample from the continoues range around the particle
            # kp = np.random.uniform(max(particle[0] - 500, self.CPFConfig['ParameterRanges']["P"]['min']),
            #                        min(particle[0] + 500, self.CPFConfig['ParameterRanges']["P"]['max']))
            # ki = np.random.uniform(max(particle[1] - 500, self.CPFConfig['ParameterRanges']["I"]['min']),
            #                        min(particle[1] + 500, self.CPFConfig['ParameterRanges']["I"]['max']))
            # kd = np.random.uniform(max(particle[2] - 500, self.CPFConfig['ParameterRanges']["D"]['min']),
            #                        min(particle[2] + 500, self.CPFConfig['ParameterRanges']["D"]['max']))
            kp = particle[0]

            kd = particle[1]
            # kd = particle[1]

            # kd = particle[2]
            # Test the particle on predefined trajectory
            # performance = self.evaluateParticle([kp, ki, kd])
            performance = self.evaluateParticle([kp, kd])

            self.performances[ind].append(performance)


        return

    def calculateAverageParticlePerformance(self):
        # compute average performance of each particle
        ind = 0
        n = self.CPFConfig['averagePerformanceLength']
        for partPerfs in self.performances:
            # print(ind , partPerfs)
            if (len(partPerfs) > n):
                avg = np.average(partPerfs[-n:])
            else:
                avg = np.average(partPerfs)

            self.avgPerformances[ind] = avg

            ind += 1

        self.totalPerformance = sum(self.avgPerformances)
        return

    def updateParticleWeights(self):

        for ind in range(0, len(self.weights)):
            self.weights[ind] = (self.avgPerformances[ind] / self.totalPerformance)

        return

    def checkEndCriteria(self):
        # minEp = 100
        # if (len(volumes) > minEp):
        #     avgClusterDerivitive = np.gradient(self.clusterNumRecord[-self.CPFConfig['averagePerformanceLength']:])
        #     print("volume der. : " + str(avgClusterDerivitive))
        #
        #     Tuned = all(i <= 0.05 for i in avgClusterDerivitive)
        #     print()

        if self.numberRuns < self.CPFConfig['MaxIter']:
            self.numberRuns += 1
        else:
            self.finished = True



        return

    def clusterParticles(self):
        min_sample = 2 * len(self.CPFConfig['ParameterRanges'])  # 2 times the dimensionality of the data (PID)
        # min_sample = 1  # 2 times the dimensionality of the data (PID)

        avgMaxParameterRange = np.average( [ param['max'] for param in self.CPFConfig['ParameterRanges'] ])
        print(avgMaxParameterRange)

        #set epsilon to slightly larger than the distance between particles
        epsilon = (avgMaxParameterRange/self.CPFConfig['NumberParticlesPerDimension'])*1.2
        print(epsilon)
        print(min_sample)
        #remove bad performing particles
        index = 0
        goodParticles = []
        for particle in self.particles:
            print(self.avgPerformances[index])
            if self.avgPerformances[index] > self.CPFConfig['performanceThreshold']:
            # if self.avgPerformances[index] >= self.CPFConfig['StartingPerformance']:
                goodParticles.append(particle)
            index += 1
        print(goodParticles)
        if (len(goodParticles) > min_sample):
            dbClusters = DBSCAN(eps=epsilon, min_samples=min_sample).fit(goodParticles)
            labels = dbClusters.labels_
            print(labels)
            numClusters = max(labels) + 1
            if numClusters == 0:
                print("no clusters found")

                return []
            else:
                largestClusterIndex = None
                largestClusterSize = 0
                for clusterIndex in set(labels):
                    clusterSize = labels.count(clusterIndex)
                    if clusterSize > largestClusterSize:
                        largestClusterIndex = clusterIndex
                        largestClusterSize = clusterSize

                ClusterIndexes = list(filter(lambda x: x == largestClusterIndex, labels))
                ClusterParticles = []
                for ind in ClusterIndexes:
                    ClusterParticles.append(self.particles[ind])

                print("ClusterIndexes of largest cluster")
                print(ClusterIndexes)
                print("Particles of largest cluster")
                print(ClusterParticles)
                self.clusterPoints = ClusterParticles
            return ClusterParticles
        return []

    def calculateClusterAndHull(self):

        cluster = self.clusterParticles()
        clusterPoints = np.array(cluster)
        hull = []
        if not cluster == []:
            hull = ConvexHull(clusterPoints)

        self.IntermediateHulls.append(hull)

        return

    def getCombinedHull(self):
        cluster = self.IntermediateHulls
        clusterPoints = np.array(cluster)
        hull = []
        if not cluster == []:
            hull = ConvexHull(clusterPoints)

    def render3D(self):
        if self.render:
            self.ax.cla()
            # self.ax1.cla()
            clusterPoints = np.array(self.clusterPoints)
            ClusterS = [ 200 for a in clusterPoints ]
            ClusterC = [ 1 for a in clusterPoints ]

            ind = 0
            particles= []
            xdata= []
            ydata= []
            particleS = []
            particleC = []
            for entry in self.particles:
                # if self.avgPerformances[ind] > -:
                particles.append(entry)
                xdata.append(entry[0])
                ydata.append(entry[1])
                # zdata.append(entry[2])
                print(self.avgPerformances[ind])
                particleS.append(500 + max(self.avgPerformances[ind], -480) )
                particleC.append(3000 + max(self.avgPerformances[ind], -3000) )
                ind += 1

            # self.ax1.scatter(particles.T[0], particles.T[1], particles.T[2], alpha=0.01, s=particleS, c=particleC, cmap="Set1")
            self.ax.scatter(xdata,ydata,s=particleS, alpha=0.5 , c=particleC, cmap='PiYG', edgecolor="k")

            if (len(clusterPoints) > 0):
                # self.ax.scatter(clusterPoints.T[0], clusterPoints.T[1], clusterPoints.T[2], alpha=0.01 , s=ClusterS, c=ClusterC, cmap="Set1")
                self.ax.scatter(clusterPoints.T[0], clusterPoints.T[1], alpha=1 , s=ClusterS, c=ClusterC, cmap="Set1")

            for hull in self.IntermediateHulls:
                for s in hull.simplices:
                    print(s)
                    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                    self.ax.plot(clusterPoints[s, 0], clusterPoints[s, 1], "g-")

            self.ax.set_xlim(self.CPFConfig['ParameterRanges']["P"]['min'],
                             self.CPFConfig['ParameterRanges']["P"]['max'])
            # self.ax.set_ylim(self.CPFConfig['ParameterRanges']["I"]['min'], self.CPFConfig['ParameterRanges']["I"]['max'])
            self.ax.set_ylim(self.CPFConfig['ParameterRanges']["D"]['min'],
                             self.CPFConfig['ParameterRanges']["D"]['max'])
            # self.ax1.set_xlim(self.CPFConfig['ParameterRanges']["P"]['min'],
            #                   self.CPFConfig['ParameterRanges']["P"]['max'])
            # # self.ax1.set_ylim(self.CPFConfig['ParameterRanges']["I"]['min'], self.CPFConfig['ParameterRanges']["I"]['max'])
            # self.ax1.set_ylim(self.CPFConfig['ParameterRanges']["D"]['min'],
            #                   self.CPFConfig['ParameterRanges']["D"]['max'])

            plt.draw()

            plt.pause(0.0001)

        return

    def runParticleFiltering(self):
        while not self.finished:

            self.setRandomEnvironmentMagnitude()

            self.sampleAndEvaluateParticles()

            self.calculateAverageParticlePerformance()

            self.updateParticleWeights()

            self.checkEndCriteria()

            if self.render:
                self.render3D()
        return

    def run(self):
        print("CPF running")

        for env in self.Environment:
            enabled = bool(self.Environment[env]['enabled'])
            if enabled:

                self.initilizeParticles()

                if self.render:
                    self.render3D()

                self.setTaskEnv(env)

                self.runParticleFiltering()

                self.calculateClusterAndHull()

        # ControllerParameterSet = 0
        ControllerParameterSet = self.getCombinedHull()

        return ControllerParameterSet







