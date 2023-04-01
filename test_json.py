import json
import tensorflow as tf
import numpy as np
# js = json.loads('/home/user/datasets/P016_balloon1_side.json')
from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

ar = np.load('/home/user/PycharmProjects/DS.ComputerVisionProject/efficientnet/B0/P016_balloon1_top.npy')
print('hi')