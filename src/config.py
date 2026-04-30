classes_config = {
    0: {"class_name": "person", "class_w": 0.55, "class_h": 1.75, "class_d": 0.30},
    2: {"class_name": "car", "class_w": 1.6, "class_h": 1.80, "class_d": 4.00}
}

classes_config_classic_size = {
            0: {"class_name": "person", "class_w": None, "class_h": 1.8, "class_d": None, "distance_by":"h"},
            2: {"class_name": "car", "class_w": None, "class_h": 1.4, "class_d": None, "distance_by":"h"}
        }

import numpy as np
min_depth = np.finfo(float).eps
max_depth = 80

accurancy_threshold = 1.25
accurancy_threshold_2 = 1.25 ** 2
accurancy_threshold_3 = 1.25 ** 3

distance_range_min = 0
distance_range_max = max_depth
distance_range_step = 5

luminosity_middle_min = 85
luminosity_middle_max = 170
