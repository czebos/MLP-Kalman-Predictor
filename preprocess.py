import numpy as np
import sys
import scipy.io as sio
from random import shuffle
import tensorflow as tf

TEST_FRACTION = .1

def mat_to_dict(file_name):
    # srates <-> neurons
    # srates is the variable that keeps track of bucketed fire rates of shape(86517, 260)
    # 260 is the amount of neurons we have, 86517 is all of the samples

    # 14 (minutes of data) * 60 (seconds in min) * 100 (miliseconds in second[we are doing 10 ms buckets and 100 * 10 = 1000 which is one second])
    # equals around 84000 ~ 86516. This is why this number is significant.

    # markpos <-> seglabel
    # markpos is the pos of the marker of the sample of shape (86517, 87)
    # there are 87 segLables, and remeber that this is only two dimensional because we scale
    # position to be 1 dimensional(e.g. below). variable-value x sample -> (86517, 87)

    #       Ypos
    #       |
    #       |
    #       |
    #       |
    # ----------------- #Xpos
    #       |
    #       |
    #       |
    #       |

    # xpos, and ypos are their own variables in markpos,
    # so they are 1 dimensional

    # segrot <-> marklabel
    # segrot is the rotation i believe of the markerlabel of shape (86517, 84)

    mat_contents = sio.loadmat(file_name)
    marker_pos = np.array(mat_contents['markpos'])
    srates = np.array(mat_contents['srates'])
    segrot = np.array(mat_contents['segrot'])
    return srates, marker_pos, segrot

def seperate_data(srates,mark_pos, segrot):
    # For now this function just uses srates as input, markpos as labels

    indices = [i for i in range(len(srates))]
    shuffled = tf.random.shuffle(indices)
    srates = tf.gather(srates, shuffled)
    marker_pos = tf.gather(mark_pos, shuffled)

    train_data = srates[:int(-len(srates)*TEST_FRACTION)]
    train_labels = mark_pos[:int(-len(mark_pos)*TEST_FRACTION)]
    test_data = srates[int(-len(srates)*TEST_FRACTION):]
    test_labels = mark_pos[int(-len(mark_pos)*TEST_FRACTION):]

    return train_data, train_labels, test_data, test_labels


srates, marker_pos, segrot = mat_to_dict('./../COS071212_mocap_processed.mat')
training_data, training_labels, test_data, test_labels = seperate_data(srates, marker_pos, segrot)
print(training_data)
print(training_labels)
