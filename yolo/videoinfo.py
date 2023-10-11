from ultralytics import YOLO
import supervision as sv
from supervision import VideoInfo
import cv2
import numpy as np

BULEVAR = r"C:\Users\anja.kovacevic\kod\yolo\bulevar.mp4"
TARGET = r"C:\Users\anja.kovacevic\kod\yolo\bulevar_target3.mp4"

model = YOLO('yolov8s.pt')

video_info=VideoInfo.from_video_path(video_path=BULEVAR)
# print("Video info: ", video_info)
# VideoInfo(width=1920, height=1080, fps=30, total_frames=1841)
polygon = np.array([
    [380, 170],
    [380, 750],
    [1920, 750],
    [1920, 170],
])
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# initiate annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)
'''
# extract video frame
# every frame is a numpy array
generator = sv.get_video_frames_generator(BULEVAR)
iterator = iter(generator)
frame = next(iterator)

# detect
results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.class_id == 2]  # car
zone.trigger(detections=detections)

# annotate
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
labels=[]

for confidence, class_id in zip(detections.confidence, detections.class_id):
    # Retrieve the class name from YOLO model names
    class_name = model.model.names[class_id]
    label = f"{class_name} {confidence:0.2f}"
    labels.append(label)

frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
frame = zone_annotator.annotate(scene=frame)

import matplotlib.pyplot as plt 

sv.plot_image(frame, (12, 6))
'''

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 2]
    zone.trigger(detections=detections)

    # annotate
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels=[]

    for confidence, class_id in zip(detections.confidence, detections.class_id):
        # Retrieve the class name from YOLO model names
        class_name = model.model.names[class_id]
        label = f"{class_name} {confidence:0.2f}"
        labels.append(label)

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    return frame

sv.process_video(source_path=BULEVAR, target_path=TARGET, callback=process_frame)

# from IPython import display
# display.clear_output()