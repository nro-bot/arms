#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

np.import_array()

def render(points, height, width, vmin, vmax):
    return np.array(crender(points, height, width, vmin, vmax))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef np.float32_t[:,::1] crender(np.float32_t[:,::1] points, int height, int width, float vmin, float vmax):

    cdef np.float32_t[:,::1] image = np.zeros([height, width], dtype=np.float32) - 100.
    cdef int limit = points.shape[0]
    cdef int x, y

    for i in range(limit):
        x = int(points[i, 0])
        y = int(points[i, 1])
        image[x, y] = max(image[x, y], points[i, 2])
    image = np.clip(image, vmin, vmax)

    return image
