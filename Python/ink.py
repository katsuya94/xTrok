import sys
import threading

from PIL import Image
from images2gif import writeGif
import numpy as np

FLOATMIN = 0.0
FLOATMAX = np.nextafter(np.float32(1.0), np.float32(0.0))

RAW = False

FPS = 30
DURATION = 2

HEIGHT = 100
WIDTH = 100

COMPONENTS = 2

'''
Components
---
0 - ink
1 - moisture
'''

BLOT_RADIUS = 15

state = np.zeros((HEIGHT, WIDTH, COMPONENTS,), dtype=np.float32)
prime = np.zeros((HEIGHT, WIDTH, COMPONENTS,), dtype=np.float32)

frames = []


for frame in xrange(FPS * DURATION):
    sys.stdout.write('%d%%' % (100 * frame / (FPS * DURATION)))
    sys.stdout.flush()

    # Events
    if np.random.random() < 6.0 / FPS:
        point = (np.random.randint(HEIGHT), np.random.randint(WIDTH),)
        i,j = np.ogrid[0:HEIGHT,0:WIDTH]
        mask = (i-point[0])*(i-point[0]) + (j-point[1])*(j-point[1]) <= BLOT_RADIUS*BLOT_RADIUS
        state[mask] = np.maximum(state[mask], (0.9, 0.5,))

    # Transformation function

    def transform(it, offseti, offsetj):
        while not it.finished:
            idx = (it.multi_index[0] + offseti, it.multi_index[1] + offsetj, it.multi_index[2],)
            delta = 0.0

            up    = idx[0]+1 < HEIGHT
            down  = idx[0]-1 >= 0
            right = idx[1]+1 < WIDTH
            left  = idx[1]-1 >= 0

            ink   = state[idx[0],idx[1],0]
            moist = state[idx[0],idx[1],1]

            if it.multi_index[2] == 0:
                if up:
                    delta += 5.0 * state[idx[0]+1,idx[1],1] * moist * (state[idx[0]+1,idx[1],0] - ink) / FPS
                if down:
                    delta += 5.0 * state[idx[0]-1,idx[1],1] * moist * (state[idx[0]-1,idx[1],0] - ink) / FPS
                if right:
                    delta += 5.0 * state[idx[0],idx[1]+1,1] * moist * (state[idx[0],idx[1]+1,0] - ink) / FPS
                if left:
                    delta += 5.0 * state[idx[0],idx[1]-1,1] * moist * (state[idx[0],idx[1]-1,0] - ink) / FPS
            elif it.multi_index[2] == 1:
                delta += -0.25 * state[idx] / FPS
                if up:
                    delta += 5.0 * (state[idx[0]+1,idx[1],1] - moist) / FPS
                if down:
                    delta += 5.0 * (state[idx[0]-1,idx[1],1] - moist) / FPS
                if right:
                    delta += 5.0 * (state[idx[0],idx[1]+1,1] - moist) / FPS
                if left:
                    delta += 5.0 * (state[idx[0],idx[1]-1,1] - moist) / FPS
            it[0] = delta
            it.iternext()

    quads = [
        (np.nditer(prime[:HEIGHT/2,:WIDTH/2,:], flags=['multi_index'], op_flags=['writeonly']), 0, 0,),
        (np.nditer(prime[HEIGHT/2:,:WIDTH/2,:], flags=['multi_index'], op_flags=['writeonly']), HEIGHT/2, 0,),
        (np.nditer(prime[:HEIGHT/2,WIDTH/2:,:], flags=['multi_index'], op_flags=['writeonly']), 0, WIDTH/2,),
        (np.nditer(prime[HEIGHT/2:,WIDTH/2:,:], flags=['multi_index'], op_flags=['writeonly']), HEIGHT/2, WIDTH/2,),
    ]

    threads = [threading.Thread(target=transform, args=quad) for quad in quads]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    state += prime

    # Bound the state
    state = state.clip(FLOATMIN, FLOATMAX)

    # State to display function
    if RAW is True:
        data = np.zeros((HEIGHT, WIDTH, 4,), dtype=np.uint8)
        data[:,:,0:2] = (state[:,:,:] * 256).astype(np.uint8)
        data[:,:,3] = 255
        img = Image.fromarray(data, mode='RGBA')
    else:
        data = np.subtract(255, (state[:,:,0] * 256).astype(np.uint8))
        img = Image.fromarray(data, mode='L')
    frames.append(img)

    sys.stdout.write('\r')
    sys.stdout.flush()

writeGif('out.gif', frames, duration=1.0/FPS)
