from ultralytics import YOLO

from detect import detect_and_save
from distance_by_size import DistanceBySize
from distance_by_classic_stereo import DistanceByClassicStereo
from distance_by_zoe_depth import DistanceByZoeDepth

def detect_boxes_0():
    model = YOLO('model/yolo26n.pt')
    
    left_boxes = detect_and_save('test_data/left/000000.png', model, 'left_image.jpg', {0: 'person', 2: 'car'})

    return left_boxes

def distance_by_size(boxes):
    distanceByHeight = DistanceBySize('test_data/calib/000000.txt')

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        d = distanceByHeight.calculate(box)
        print(d)

def distance_by_classic_stereo(boxes):
    distance_by_classic_stereo = DistanceByClassicStereo()
    depth_map = distance_by_classic_stereo.calculate_depth_map('test_data/left/000000.png', 'test_data/right/000000.png', 'test_data/calib/000000.txt')

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh.round().int().tolist()}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(d)

def distance_by_zoe_depth(boxes):
    distance_by_zoe_depth = DistanceByZoeDepth(model_type = "ZoeD_NK")
    depth_map = distance_by_zoe_depth.calculate_depth_map('test_data/left/000000.png')

    print(depth_map.shape)

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh.round().int().tolist()}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(x, y)

        print(d)

if __name__ == "__main__":
    boxes = detect_boxes_0()

    distance_by_zoe_depth(boxes)

    # distance_by_size(boxes)
    # distance_by_classic_stereo(boxes)



# def distance_by_YOLO_with_depth():
#     WEIGHTS_PATH = "../external/dist_yolo_core/model_data/yolo3_xception_dist_final.h5"
#     TEST_IMAGE = "../test_data/left/000000.png" 

#     detector = DistYOLODetector(
#         weights_path=WEIGHTS_PATH,
#         config={'score': 0.3, 'iou': 0.45},
#         verbose=True
#     )

#     results = detector.detect(TEST_IMAGE)
        
#     for i, obj in enumerate(results, 1):
#         print(f"[{i}] {obj['class']:12s} | "
#                 f"conf: {obj['confidence']:.3f} | "
#                 f"dist: {obj['distance_m']:5.2f} м | "
#                 f"bbox: {obj['bbox']}")
        
#     img = cv2.imread(TEST_IMAGE)
#     vis = detector.draw_results(img, results)
#     cv2.imwrite("dist_yolo_output.jpg", vis)
