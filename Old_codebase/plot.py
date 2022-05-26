import numpy as np
import matplotlib.pyplot as plt
import json


# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')


fp = open('PSO_output.json')
data = json.load(fp)

x = []
y = []
fit = []
points = []
for d in data:
    # print(d)
    P = round(d['ParticleParameters']['P'], 1)
    D = round(d['ParticleParameters']['D'], 1)
    x.append(P)
    y.append(D)
    fit.append(float(d['total_loss']))
    points[P][D] = float(d['total_loss'])
# print(data)
#
# print(x)
# print(y)
# print(fit)

# # fake data
_x = np.arange(0,20,0.1)
_y = np.arange(0,20,0.1)



# _xx, _yy = np.meshgrid(_x, _y)
# # print(_xx)
# x, y = _xx.ravel(), _yy.ravel()
#
# # X, Y = np.meshgrid(x, y)
# zs = np.array([ [10]*len(y) for _ in range(len(x)) ])
# Z=zs.reshape(len(y),len(x))


X, Y = np.meshgrid(_x, _y)
R = points[X][Y]
Z = R
#
# zs = [
#     [0,0,0,0,0],
#     [0,0,0,0,0]
# ]

ax1.plot_surface(X,Y,Z)

plt.draw()
plt.pause(0.0001)

plt.show()