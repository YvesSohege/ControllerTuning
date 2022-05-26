import json
import matplotlib.pyplot as plt
import os
Alldbfiles = [f for f in os.listdir() if not f == "dbAnalysis.py" ]
PSOdbfiles = [f for f in os.listdir() if "PSO" in f ]
CPFdbfiles = [f for f in os.listdir() if "CPF" in f ]

dbfiles = ["PSO_Results-2022-05-12 16-22.json" ]
# dbfiles = ["CPF_Results-2022-05-13 09-56.json" ]
# dbfiles =PSOdbfiles

print(CPFdbfiles)

xs = []
ys = []
zs = []
data = []
for file in dbfiles:
    db= file
    fp = open(db)
    db_data = json.load(fp)
    fp.close()
    # print(db_data)
    for d in db_data:

        zs.append(d['performance'])

        xs.append(d['ControllerParameters']['Angular_PID'][0]['P'][0])

        ys.append(d['ControllerParameters']['Angular_PID'][0]['D'][0])

        data.append(d)

print(len(data))
# print(xs)
#
#
# print(xs)
# print(ys)
# print(zs)

fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection="3d")
ax.set_xlim(min(xs), max(xs))
ax.set_ylim(min(ys), max(ys))
ax.set_zlim(min(zs), max(zs))


# ax1.scatter(particles.T[0], particles.T[1], particles.T[2], alpha=0.01, s=particleS, c=particleC, cmap="Set1")
ax.scatter(xs, ys, zs, alpha=0.3, cmap='PiYG', edgecolor="k")


plt.draw()

plt.pause(0.0001)

plt.show()