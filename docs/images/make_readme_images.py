#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pyspeckle

target_length = 50000
target_std = 100

target_ave = 1200
y = pyspeckle.create_exp_1D(target_length, target_ave, target_std, 20)
ave = np.mean(y)
std = np.sqrt(np.var(y))
plt.text(target_length, ave, "  %.0f ± %.0f" % (ave, std), fontsize=14)
plt.plot(y)
plt.plot([0, target_length], [target_ave, target_ave], ":k")

target_ave = 800
y = pyspeckle.create_exp_1D(target_length, target_ave, target_std, 50)
ave = np.mean(y)
std = np.sqrt(np.var(y))
plt.text(target_length, ave, "  %.0f ± %.0f" % (ave, std), fontsize=14)
plt.plot(y)
plt.plot([0, target_length], [target_ave, target_ave], ":k")

target_ave = 400
y = pyspeckle.create_exp_1D(target_length, target_ave, target_std, 100)
ave = np.mean(y)
std = np.sqrt(np.var(y))
plt.text(target_length, ave, "  %.0f ± %.0f" % (ave, std), fontsize=14)
plt.plot(y)
plt.plot([0, target_length], [target_ave, target_ave], ":k")

target_ave = 0
y = pyspeckle.create_exp_1D(target_length, target_ave, target_std, 500)
ave = np.mean(y)
std = np.sqrt(np.var(y))
plt.text(target_length, ave, "  %.0f ± %.0f" % (ave, std), fontsize=14)
plt.plot(y)
plt.plot([0, target_length], [target_ave, target_ave], ":k")

plt.xlim(0, target_length * 1.25)
plt.title("Expected means are 1200, 800, 400, and 0 with std dev=%.0f" % target_std)
plt.xlabel("Sample number")
plt.ylabel("Exp Random Correlates")
plt.savefig('oneD_example.png', dpi=300)
plt.show()

y = pyspeckle.create_Exponential(201, 2)
pyspeckle.statistics_plot(y)
plt.savefig('twoD_speckle.png', dpi=300)
plt.show()
