def detect_and_save(image_path, model, save_path, TARGET_CLASSES = {0: 'person', 2: 'car'}):
    result = model(image_path, classes=list(TARGET_CLASSES.keys()))
    result[0].save(save_path)
    return result[0].boxes
