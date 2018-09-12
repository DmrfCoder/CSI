import numpy as np




def reshapetest():
    x = np.random.random(size=(20, 200, 360))

    x[0][0][0] = 0
    x[0][0][359] = 1
    x[0][199][0] = 2
    x[0][199][359] = 3

    x[19][0][0] = 4
    x[19][0][359] = 5
    x[19][199][0] = 6
    x[19][199][359] = 7

    x = np.reshape(x, newshape=(-1, 7200))

    x = np.reshape(x, newshape=(-1, 200, 360))

    print (x[0][0][0],
           x[0][0][359],
           x[0][199][0],
           x[0][199][359],

           x[19][0][0],
           x[19][0][359],
           x[19][199][0],
           x[19][199][359])



reshapetest()