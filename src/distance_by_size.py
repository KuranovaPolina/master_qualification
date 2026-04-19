import cv2

def object_config_print(object_config):
    print(f"Class: {object_config["class_name"]}, \
            Distance by: {object_config["distance_by"]}, \
            Width: {object_config["class_w"]}, \
            Height: {object_config["class_h"]}, \
            D...: {object_config["class_d"]}")

class DistanceBySize:    
    def __init__(self, objects_config):
        self.objects_config = objects_config

    def getCalibParams(self, calib):
        K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(calib.P0)

        return K[0][0], K[1][1]
    
    def calculate(self, box, calib):
        object_config = self.objects_config[box.cls.item()]
        object_config_print(object_config)

        fx, fy = self.getCalibParams(calib)

        if object_config["distance_by"] == 'w':
            return fx * object_config["class_w"] / (box.xywh[0][2].item())
        elif object_config["distance_by"] == 'h':
            return fy * object_config["class_h"] / (box.xywh[0][3].item())
        else:
            return None

def distance_by_size(DistanceBySize_model, boxes, calib):
    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        d = DistanceBySize_model.calculate(box, calib)
        # print(d)
        distances.append(d)

    return distances
