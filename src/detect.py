def detect_and_save(image_path, model, save_path, target_classes):
    result = model(image_path, classes=list(target_classes.keys()))
    result[0].save(save_path)
    return result[0].boxes
