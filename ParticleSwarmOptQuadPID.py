import gym
import quadcopter,controller #,gui
from sklearn.neighbors import NearestNeighbors
import time
import matplotlib.pyplot as plt

from ParticleQuad import ParticleQuad
import numpy as np

from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.gridspec as gridspec

# Create 2x2 sub plots
gs = gridspec.GridSpec(1, 1)

fig = plt.figure(figsize=(6,6))


ax = plt.subplot(gs[0, 0]) # row 0, col 1


# ax = plt.subplot(gs[0, 1]) # row 1, span all columns



# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax1 = fig.add_subplot(2, 1, 2, projection='3d')
# ax2 = fig.add_subplot(2, 2, 3)


#ax1 = fig.add_subplot(1, 2, 2, projection='3d')
renderCount = 0
figTitle = "Quadcopter Particles: "
envTitle = ""
envsTested = []
volumes = []


kp = 1
ki = 1
kd = 1
P_min = 0
P_max = 20

D_min = 0
D_max = 20
PIDScalar = 1000

lastEps = []
#plt.ion()


numberRuns = 0
particles = []
performances = []
avgPerformances = []
weights = []
clusterNumRecord = []

def render( mode='human'):
    global  renderCount,ax
    ax.cla()
    renderCount+=1

    xdata = []
    ydata = []
    sizes = []
    colors = []
    for i in range(N):
        xdata.append(particles[i][0])
        ydata.append(particles[i][1])
        sizes.append(200)

        velLineX = particles[i][0]-velocities[i][0]
        velLineY = particles[i][1]-velocities[i][1]

        xVel = [particles[i][0], velLineX]
        yVel = [particles[i][1], velLineY]

        ax.plot(xVel, yVel, c='k')

    ax.scatter(xdata, ydata, s=sizes, alpha=0.8, c='k', edgecolor="k", linewidth=0.4)

    # ax2.scatter(envX, envY, alpha=0.5, c="k" , edgecolor="k", linewidth=1)

    #
    ax.set_xlim(P_min,P_max)
    ax.set_ylim(D_min,D_max)

    scalarText = str(PIDScalar)

    ax.set_xlabel("P-gain * " + scalarText)
    ax.set_ylabel("D-gain * "+ scalarText)


    #magSetting = [0,0,0,0]

    s =""
    rotorScalar = 0.05
    windScalar = 3
    PosScalar = 0.9
    AttScalar = 0.15

    if 'Rotor' in Domain:
        s += ' Rotor LOE = ' + str(round((magSetting[0]*rotorScalar)*100 ,2)) +'%'
    if 'Wind' in Domain:
        s += ' Wind = ' + str(magSetting[1]*windScalar) +'m/s'
    if 'PosNoise' in Domain:
        s += ' Position Noise = ' + str(round(magSetting[2]*PosScalar, 2)) +'m'
    if 'AttNoise' in Domain:
        s += ' Attitude Noise = ' + str(round(magSetting[3]*AttScalar , 2)) +'rad'
    if Domain == []:
        s = " No Faults"

    ax.set_title("Particle Position - " + s)
    plt.draw()

    plt.pause(0.0001)
    plt.savefig("PSO/"+str(renderCount)+".png")
    return



def evaluateFitness(pos):

    kp = pos[0]*PIDScalar
    kd = pos[1]*PIDScalar
    PQ = ParticleQuad(kp, 0, kd)

    PQ.setEnv(Domain, magSetting)
    performance = PQ.run()


    return performance

#initilize paramaters for PSO
N = 16
c1 =2
c2 =1
Wmin= 0.05
Wmax = 1
w = 0.9 #usually starts at 0.9 and decreases to 0.4
Vmax =1
MaxIter = 100

particles = {}
velocities = {}
local_best = {}
global_best = ([12 , 4],-2900)

Domain = [ 'Wind' , 'AttNoise']
magSetting = [2,2,2,2]


#Initilize N Particles with random positions and velocities
for i in range(N):
    print(i)
    randX = np.random.uniform(P_min,P_max)
    randY = np.random.uniform(D_min,D_max)
    randVelX = np.random.uniform(-Vmax*3,Vmax*3)
    randVelY = np.random.uniform(-Vmax*3,Vmax*3)

    particles[i] = [randX,randY]
    velocities[i]= [randVelX,randVelY]
    fit = evaluateFitness([randX,randY])
    local_best[i] = ([randX,randY], fit)
    if fit > global_best[1]:
        global_best = ([randX,randY], fit)

fin = False



while not fin:

    for i in range(N):

        pos = particles[i]
        fitness = evaluateFitness(pos)
        print(str(pos) + " fitness: " + str(fitness))
        if fitness >= local_best[i][1]:
            local_best[i] = (pos,fitness)

        if fitness >= global_best[1]:
            global_best= (pos,fitness)


    #update interia

    for i in range(N):
        r1 = np.random.uniform(0,1)
       # r1 = w
        r2 = np.random.uniform(0,1)*c1
        r3 = np.random.uniform(0,1)*c2

        InteriaComp  = [r1*velocities[i][0] ,r1*velocities[i][1]]
        # InteriaComp[0] = min(max(InteriaComp[0], -Vmax), Vmax)
        # InteriaComp[1] = min(max(InteriaComp[1], -Vmax), Vmax)

        CognitiveComp= [(local_best[i][0][0] - particles[i][0])*r2 , (local_best[i][0][1] - particles[i][1])*r2]
        CognitiveComp[0] = min(max(CognitiveComp[0], -Vmax), Vmax)
        CognitiveComp[1] = min(max(CognitiveComp[1], -Vmax), Vmax)

        SocialComp   = [(global_best[0][0] - particles[i][0])*r3 , (global_best[0][1] - particles[i][1])*r3 ]
        SocialComp[0] = min(max(SocialComp[0], -Vmax), Vmax)
        SocialComp[1] = min(max(SocialComp[1], -Vmax), Vmax)

        # print("Velocities:")
        # print(InteriaComp)
        # print(CognitiveComp)
        # print(SocialComp)
        newVel = [ InteriaComp[0] + CognitiveComp[0] + SocialComp[0] ,
                   InteriaComp[1] + CognitiveComp[1] + SocialComp[1]]

        velocities[i]= [Vel*w for Vel in newVel]

        newPos = [particles[i][0] + velocities[i][0] , particles[i][1] + velocities[i][1] ]

        newPos[0] = min(max(newPos[0],P_min),P_max)
        newPos[1] = min(max(newPos[1],P_min),P_max)
        particles[i]= newPos
    print("Global Best = " + str(global_best) + " at " + str(w))
    if w> Wmin:
        w -= 0.01
    else:
        fin = True
    render()

    #temrinating condition

exit()

#LOOP
    #evaluate fitness of all particles at their positions (configs)

        #if fitness of particles is better than local best , update local best

    #Set the best of the local bests as the global bests

    #update every particles velocity and then position
        #get update rules

#is global best == optimal ?  / need to define optimal( below threshold/ minimum? )












