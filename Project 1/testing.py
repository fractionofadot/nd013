import matplotlib.pyplot as plt
import numpy as np
import lines2

x = []
y = []
m = []

for line in lines2.lines:
	x1,y1,x2,y2 = line
	plt.plot((x1,x2),(y1,y2))
	slope = (y2-y1)/(x2-x1)
	if 0.2 < slope < 0.8:
		pass
	elif -0.8 < slope < -0.2:
		x += [x1,x2]
		y += [y1,y2]
		m += [slope]

z = np.polyfit(x, y, 1)
f = np.poly1d(z)

x_max = max(x)
x_min = min(x)


x_new = np.linspace(x_min, x_max, 20).astype(int)
y_new = f(x_new).astype(int)
points_new = list(zip(x_new, y_new))

# plt.plot((0, x_max), (f(0), f(x_max)), 'r')

# px, py = points_new[0]
# cx, cy = points_new[-1]
px = x_min
cx = x_max
py = int(f(x_min))
cy = int(f(x_max))

plt.plot((px, cx), (py, cy), 'r')

plt.axis([0, 960, 540, 0])
plt.show()

