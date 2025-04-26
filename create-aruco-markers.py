import os
from pathlib import Path
import numpy as np
import cv2

# Ensure the output directory exists
os.makedirs("aruco_markers", exist_ok=True)

# Generate ArUco markers
marker = np.zeros((300, 300, 1), dtype="uint8")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
print("Available ArUco dictionaries:", aruco_dict.markerSize)

for i in range(aruco_dict.markerSize):
    marker = cv2.aruco.generateImageMarker(aruco_dict, i, 300, marker, 1)
    output_path = Path("aruco_markers") / f"{i}.png"
    cv2.imwrite(output_path, marker)
    print(f"Marker {i} saved as {output_path}")
    marker.fill(0)  # Reset marker for the next iteration
