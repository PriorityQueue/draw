# encoding=utf-8
import matplotlib.pyplot as plt
import math
from numpy.random import randn
import random
import numpy as np


def storeData(route, filename):
    import pandas as pd
    index = [x for x in range(len(route))]
    rate = [round((((random.random() / 1) * 0.13) + 0.8), 2) for _ in range(len(route))]
    y = [round(x, 2) for x in route]
    data = {"index": index, "rate": rate, "y": y}
    frame = pd.DataFrame(data)
    frame.to_csv(filename)


def getAngle(route):
    dt = [0] * len(route)
    for i in range(1, len(route)):
        dt[i] = route[i] - route[i - 1]
    return dt


def addList(list1, list2, a):
    if (len(list2) + a > len(list1)):
        return -1


def getTurn(length):
    a = getStraight(length)
    r = 30
    x0 = 50
    y0 = 0
    arc = [2*r-math.sqrt(math.pow(r, 2) - math.pow(x - x0, 2)) for x in range(x0, x0 + r)]
    return arc


def getDynamic(length):
    return [math.log(math.pow(x, 12)) for x in range(1, length)]


def getStraight(length):
    # 路径的随机量
    b = [(random.random()-0.5) * 3 for _ in range(length)]

    # b = randn(length) * 0.3

    # sin 函数波动
    a = [np.sin(0.2 * x) + b[x] for x in range(length)]
    return a


def getStatic(length):
    a = getStraight(length)
    r = length * 0.2
    x0 = length * 0.7
    y0 = 3
    yb = 2
    xa = 1
    d = [np.sqrt(yb * r * r - (yb / xa) * (x - x0) * (x - x0)) + y0 for x in range(int(x0 - r) + 1, int(x0 + r))]
    t = [0] * (int(x0 - r))
    t.extend(d)
    len1 = len(t)
    t.extend([d[-1]] * (len(a) - len1))
    t = np.array(t)
    a = np.array(a)
    a += t
    return a


def getOneOb(length):
    a = getStraight(length)
    r = length * 0.2
    x0 = length * 0.7
    y0 = 3
    yb = 2
    xa = 1
    d = [np.sqrt(yb * r * r - (yb / xa) * (x - x0) * (x - x0)) + y0 for x in range(int(x0 - r) + 1, int(x0 + r))]
    t = [0] * (int(x0 - r))
    t.extend(d)
    len1 = len(t)
    t.extend([0] * (len(a) - len1))
    t = np.array(t)
    a = np.array(a)
    a += t
    return a


# a = getStraight(100)
# storeData(a,"abc.csv")




fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)

ax2 = fig.add_subplot(2, 1, 2)

# ax3 = fig.add_subplot(3, 1, 3)


# a = getStatic(100)
# ax1Title = "Static"

# a = getOneOb(100)
# ax1Title = "AlongSide"

# a = getStraight(100)
# ax1Title = "Straight"

# a = getDynamic(100)
# ax1Title = "Dynamic"

a = getTurn(100);
ax1Title = "Dynamic"

ax1.plot(a, 'g-', label='Route')

# ax1.plot(a, 'g-', label='Angle')
# ax1.plot(dt, 'y-', label='four')
# ax1.plot(b,'r-',label='two')
# ax1.plot(c,'k-',label='three')



ax2.plot(getAngle(a), 'm-', label='Angle')
# ax2.plot(sumt,'y-',label='sum')


# ax3.plot(speed, 'm-', label='speed')
# ax3.plot(a,'m',label='rout2')


ax1.set_title(ax1Title)
# ax1.set_xlabel('x(m)')
ax1.set_ylabel('y(m)')
# ax1.set_ylim([-50, 50])
ax1.set_xlim([0,100])
ax1.legend(loc='best')
# labels = ax1.set_xticklabels([str(x) for x in range(0, 190, 36)], fontsize='small')

ax2.set_ylabel('Angle( )')
ax2.set_xlabel('x(m)')
ax2.set_ylim([-15, 15])
labelsX = ax2.set_xticklabels([str(x) for x in range(0, 190, 36)], fontsize='small')
labelsY = ax2.set_yticklabels([str(x) for x in range(-60, 70, 20)], fontsize='small')
ax2.legend(loc='best')

plt.show()
