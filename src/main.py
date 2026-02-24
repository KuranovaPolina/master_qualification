from ultralytics import YOLO

TARGET_CLASSES = {0: 'person', 2: 'car'}

model = YOLO('/home/polina/Documents/master_qualification/model/yolo26n.pt')

results_left = model('/home/polina/Documents/kitti_dataset/object_detection_dataset/data_object_image_2/training/image_2/000000.png', 
        classes=list(TARGET_CLASSES.keys()))
results_right = model('/home/polina/Documents/kitti_dataset/object_detection_dataset/data_object_image_3/training/image_3/000000.png', 
        classes=list(TARGET_CLASSES.keys()))

results_left[0].save('left_image.jpg')
results_right[0].save('right_image.jpg')

for box in results_left[0].boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xyxy}")

for box in results_right[0].boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xyxy}")
