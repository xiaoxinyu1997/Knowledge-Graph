# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import collections
# import math
# import random
# import jieba
# import numpy as np
# from six.moves import xrange
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import openpyxl
# from LAC import LAC




# reverse_dictionary = {0:'a', 1:'b', 2:'c'}
# final_embeddings = [0.1, 0.2, 0.3]


# dict_file = {}
# for i in range(len(reverse_dictionary)):
#     dict_file[reverse_dictionary[i]] = final_embeddings[i]
# np.save("dictionary.npy", dict_file)


# print(np.load('dictionary.npy', allow_pickle=True).item())