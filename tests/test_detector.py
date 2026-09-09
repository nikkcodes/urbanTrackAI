"""Run the vehicle detector against the sample traffic image."""

import cv2

from perception.vehicle_detector import VehicleDetector


frame = cv2.imread("data/input/traffic.jpg")
if frame is None:
    raise FileNotFoundError("Could not load data/input/traffic.jpg")

detector = VehicleDetector()
vehicles = detector.detect(frame)

print(f"Number of detected vehicles: {len(vehicles)}")
for vehicle in vehicles:
    print(f"Vehicle type: {vehicle['vehicle_type']}")
    print(f"Confidence: {vehicle['confidence']:.2f}")
    print(f"Bounding box: {vehicle['bbox']}")
