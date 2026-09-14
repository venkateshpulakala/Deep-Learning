import numpy as np


def max_pooling(feature_map, pool_size=2, stride=1):
    height, width = feature_map.shape

    output_height = (height - pool_size) // stride + 1
    output_width = (width - pool_size) // stride + 1

    pooled_output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):

            start_i = i * stride
            start_j = j * stride

            region = feature_map[
                start_i:start_i + pool_size,
                start_j:start_j + pool_size
            ]

            pooled_output[i, j] = np.max(region)

    return pooled_output