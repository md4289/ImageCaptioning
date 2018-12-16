import visdom
import random
import numpy as np
vis = visdom.Visdom()

epoch = 10

epochloss = [1, 9, 3, 6, 2, 5, 0, 12]
print(epochloss)
vis.line(Y=epochloss, opts=dict(showlegened=True))