import matplotlib.pyplot as plt
import numpy as np
im = plt.imread('1.png')
h, w, _ = im.shape
print(h, w)
figsize = (w/100, h/100)
fig = plt.figure(figsize=figsize)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(im)
plt.savefig('create.png')