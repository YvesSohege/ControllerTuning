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
gs = gridspec.GridSpec(1, 2)

fig = plt.figure(figsize=(10,5))


ax1 = plt.subplot(gs[0, 0]) # row 0, col 1


ax = plt.subplot(gs[0, 1]) # row 1, span all columns



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
    global  renderCount,ax,ax1
    ax.cla()
    ax1.cla()

    renderCount+=1
    xdata = []
    ydata = []
    zdata = []
    sizes = []
    colors = []
    ind = 0
    points = []
    ClusterS = []
    ClusterC = []
    for entry in particles:
        if avgPerformances[ind] > 950:
            points.append(entry)

        xdata.append(entry[0])
        ydata.append(entry[1])
        # zdata.append(entry[2])
        sizes.append(avgPerformances[ind] / 3)
        colors.append(avgPerformances[ind])
        ind += 1

    pts = np.array(points)

    # envX = []
    # envY = []
    # envZ = []
    # for e in envsTested:
    #     envX.append(e[0])
    #     envY.append(e[1])
    #
    ax.set_title("Particle Probability/Performance")
    ax1.set_title("Optimal Controller \n Parameter Space")

    ax.scatter(xdata, ydata, s=sizes, c=colors, alpha=0.8, cmap='PiYG', edgecolor="k", linewidth=0.4,vmax=1000, vmin=0)

    # ax2.scatter(envX, envY, alpha=0.5, c="k" , edgecolor="k", linewidth=1)

    #
    ax.set_xlim(P_min,P_max)
    ax.set_ylim(D_min,D_max)

    scalarText = str(PIDScalar)

    ax.set_xlabel("P-gain * " + scalarText)
    ax.set_ylabel("D-gain * "+ scalarText)

    ax1.set_xlim(P_min, P_max)
    ax1.set_ylim(D_min, D_max)

    ax1.set_xlabel("P-gain * " + scalarText)
    ax1.set_ylabel("D-gain * " + scalarText)

    #magSetting = [0,0,0,0]

    s =""
    rotorScalar = 0.05
    windScalar = 3
    PosScalar = 0.9
    AttScalar = 0.15

    if 'Rotor' in Domain:
        s += 'Rotor LOE = ' + str(round((magSetting[0]*rotorScalar)*100 ,2)) +'%'
    if 'Wind' in Domain:
        s += 'Wind = ' + str(magSetting[1]*windScalar) +'m/s'
    if 'PosNoise' in Domain:
        s += 'Position Noise = ' + str(magSetting[2]*PosScalar) +'m'
    if 'AttNoise' in Domain:
        s += 'Attitude Noise = ' + str(magSetting[3]*AttScalar) +'rad'
    if Domain == []:
        s = "No Faults"

    ax1.text(P_min+1, D_max-1, s, fontsize=12, bbox=dict(edgecolor='k', facecolor=(0,0,0,0)))
    # print(clusterPoints)


    min_sample = 2 * 2  # 2 times the dimensionality of the data (PID)
    epsilon = 1.2  # k n n
    if (len(pts) > min_sample):
        # Empirically determined

        db = DBSCAN(eps=epsilon, min_samples=min_sample).fit(pts)
        labels = db.labels_
        print(labels)
        particleCluster = []
        clusterPoints = list([])
        ind = 0
        numClusters = max(labels) + 1
        clusterNumRecord.append(numClusters)
        for val in labels:

            if val != -1:
                particleCluster.append(points[ind])
                ClusterS.append(200)
                ClusterC.append(val + 1)
                clusterPoints.append(points[ind])
            # else:
            #     ClusterS.append(50)
            #     ClusterC.append(0)
            ind += 1

        clusterPoints = np.array(clusterPoints)

        if (len(clusterPoints) > 0):
            ax1.scatter(clusterPoints.T[0], clusterPoints.T[1], s=ClusterS, c=ClusterC, cmap="Set1")
        # ax1.scatter(pts.T[0], pts.T[1] , cmap="Set1")

        if (numClusters >= 1):

            try:
                hull = ConvexHull(clusterPoints)
                hullVolume = hull.volume
                #print("Hull Volume : " + str(hullVolume))
                # volumes.append(hullVolume)

                # ax1.plot(pts.T[0], pts.T[1])
                for s in hull.simplices:
                    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                    ax1.plot(clusterPoints[s, 0], clusterPoints[s, 1], "g-")
            except Exception as e:
                print(e)

            # Make axis label
        for i in ["x", "y"]:
            eval("ax1.set_{:s}label('{:s}')".format(i, i))

    plt.draw()

    plt.pause(0.0001)
    plt.savefig("Particles/"+str(renderCount)+".png")
    return

startPerf = 300
for p in range(P_min,P_max):
    # for i in range(I_min,I_max):
        for d in range(D_min,D_max):
            particles.append((p,d))
            performances.append([startPerf])
            avgPerformances.append(startPerf)


for particle in particles:
    weights.append(1/len(particles))


print(weights)
# ============================================
# Set the environment conditons for the run
Domain = []
DomainToTest =['PosNoise']
randRotor= 0 #np.random.randint(1,4)
randWind = 0# np.random.randint(0,5)
randPos  = 0#np.random.randint(0,5)
randAtt  = 0 #np.random.randint(0,5)

magSetting = [randRotor,randWind,randPos, randAtt]


done = False

Tuned = False

threshold = 200
thresholdIncrease = 10
numSamples = 20
render()

maxMag = 1
numberOfRunsPerMag = 200

for i in range(maxMag):

    if i>0:
        Domain = DomainToTest

    if 'Rotor' in Domain:
        magSetting = [i,0,0,0]
    if 'Wind' in Domain:
        magSetting = [0,i,0,0]
    if 'PosNoise' in Domain:
        magSetting = [0,0,i,0]
    if 'AttNoise' in Domain:
        magSetting = [0,0,0,i]
    if Domain == []:
        magSetting=[0,0,0,0]


    for j in range(numberOfRunsPerMag):

        #Step 1 : sample N new particles to test
        particleSet = []
        sampleInd = np.random.choice(len(particles), numSamples, p=weights)

        for ind in sampleInd:
            particle = particles[ind]


            kp = np.random.uniform(max(particle[0]-0.5 , P_min), min(particle[0]+0.5, P_max))   * PIDScalar
            # ki = np.random.uniform(max(particle[1]-0.5 , I_min), min(particle[1]+0.5, I_max))   * PIDScalar
            kd = np.random.uniform(max(particle[1]-0.5 , D_min), min(particle[1]+0.5, D_max))   * PIDScalar
            #Test the particle on predefined trajectory
            PQ = ParticleQuad(kp,0,kd)

            # magSetting = [0,0,0,0]
            PQ.setEnv(Domain, magSetting)

            performance = PQ.run()

            performances[ind].append(performance)


        ind = 0
        n = 10
        for partPerfs in performances:
            #print(ind , partPerfs)
            if(len(partPerfs) > n ):
                avg =  np.average(partPerfs[-n:])
            else:
                avg = np.average(partPerfs)

            avgPerformances[ind] = avg


            ind += 1


        totalPerf = sum(avgPerformances)

        ind = 0
        for partPerfs in performances:
            weights[ind] = avgPerformances[ind]/totalPerf
            ind += 1


        numberRuns += 1
        n = 10
        minEp = 100
        if(len(volumes)  > minEp ):
            avgClusterDerivitive = np.gradient(clusterNumRecord[-n:])
            print("volume der. : " + str(avgClusterDerivitive))

            Tuned = all(i <= 0.05 for i in avgClusterDerivitive)
            print()

        render()





#
# ax.cla()
# ax1.cla()
#
# renderCount+=1
# xdata = []
# ydata = []
# zdata = []
# sizes = []
# colors = []
# ind = 0
# points = []
# ClusterS = []
# ClusterC = []
# for entry in particles:
#     if avgPerformances[ind] > 900:
#         points.append(entry)
#
#     xdata.append(entry[0])
#     ydata.append(entry[1])
#     # zdata.append(entry[2])
#     sizes.append(avgPerformances[ind] / 5)
#     colors.append(avgPerformances[ind])
#     ind += 1
#
# pts = np.array(points)
#
# # envX = []
# # envY = []
# # envZ = []
# # for e in envsTested:
# #     envX.append(e[0])
# #     envY.append(e[1])
# #
# ax.set_title("Particle Probability")
# ax1.set_title("Optimal Controller \n Parameter Space")
#
# ax.scatter(xdata, ydata, s=sizes, c=colors, alpha=0.8, cmap='PiYG', edgecolor="k", linewidth=0.4,vmax=1000, vmin=0)
#
# # ax2.scatter(envX, envY, alpha=0.5, c="k" , edgecolor="k", linewidth=1)
#
# #
# ax.set_xlim(P_min,P_max)
# ax.set_ylim(D_min,D_max)
#
# scalarText = str(PIDScalar)
#
# ax.set_xlabel("P-gain * " + scalarText)
# ax.set_ylabel("D-gain * "+ scalarText)
#
# ax1.set_xlim(P_min, P_max)
# ax1.set_ylim(D_min, D_max)
#
# ax1.set_xlabel("P-gain * " + scalarText)
# ax1.set_ylabel("D-gain * " + scalarText)
# min_sample = 2 * 2  # 2 times the dimensionality of the data (PID)
# epsilon = 1.2  # k n n
# if (len(pts) > min_sample):
#     # Empirically determined
#
#     db = DBSCAN(eps=epsilon, min_samples=min_sample).fit(pts)
#     labels = db.labels_
#     print(labels)
#     particleCluster = []
#     clusterPoints = list([])
#     ind = 0
#     numClusters = max(labels) + 1
#     clusterNumRecord.append(numClusters)
#     for val in labels:
#
#         if val != -1:
#             particleCluster.append(points[ind])
#             ClusterS.append(200)
#             ClusterC.append(val + 1)
#             clusterPoints.append(points[ind])
#         # else:
#         #     ClusterS.append(50)
#         #     ClusterC.append(0)
#         ind += 1
#
#     clusterPoints = np.array(clusterPoints)
#     print("Final cluster points and labels for " + str(Domain))
#     print(clusterPoints)
#     print(labels)
#     if (len(clusterPoints) > 0):
#         ax1.scatter(clusterPoints.T[0], clusterPoints.T[1], s=ClusterS, c=ClusterC, cmap="Set1")
#     # ax1.scatter(pts.T[0], pts.T[1] , cmap="Set1")
#
#     # print(clusterPoints)
#     if (numClusters >= 1):
#
#         try:
#             hull = ConvexHull(clusterPoints)
#             hullVolume = hull.volume
#             #print("Hull Volume : " + str(hullVolume))
#             # volumes.append(hullVolume)
#
#             # ax1.plot(pts.T[0], pts.T[1])
#             for s in hull.simplices:
#                 s = np.append(s, s[0])  # Here we cycle back to the first coordinate
#                 ax1.plot(clusterPoints[s, 0], clusterPoints[s, 1], "g-")
#         except Exception as e:
#             print(e)
#
#         # Make axis label
#     for i in ["x", "y"]:
#         eval("ax1.set_{:s}label('{:s}')".format(i, i))
#
# plt.draw()
#
# plt.show()
# plt.savefig("Particles/"+str(renderCount)+".png")


