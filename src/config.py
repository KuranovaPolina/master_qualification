classes_config = {
    0: {"class_name": "person", "class_w_sm": 55, "class_h_sm": 175, "class_d_sm": 30},
    # 1: {"class_name": "bicycle", "class_w_sm": 30, "class_h_sm": 30, "class_d_sm": 30},
    2: {"class_name": "car", "class_w_sm": 160, "class_h_sm": 180, "class_d_sm": 400}
}

classes_config_classic_size = {
            0: {"class_name": "person", "class_w": None, "class_h": 1.8, "class_d": None, "distance_by":"h"},
            2: {"class_name": "car", "class_w": None, "class_h": 1.4, "class_d": None, "distance_by":"h"}
        }


accurancy_threshold = 1.25
accurancy_threshold_2 = 1.25 ** 2
accurancy_threshold_3 = 1.25 ** 3

distance_range_min = 0
distance_range_max = 90
distance_range_step = 5

luminosity_middle_min = 85
luminosity_middle_max = 170
