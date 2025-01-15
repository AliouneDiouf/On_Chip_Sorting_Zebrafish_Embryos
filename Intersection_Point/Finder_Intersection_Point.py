# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:38:04 2024

@author: SEYDINA
"""

from ultralytics import YOLO
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import torch
import os

# Load the trained model
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(r'your link for the best.pt', task='detect')  # Adjust the path to your trained model
model.to(device=device)

def detect_edges_yolov8(frame):
    img_height, img_width = frame.shape[:2]
   
    # Run inference
    results = model(frame, stream=True)
   
    # Create an empty edge map
    edge_map = np.zeros((img_height, img_width), dtype=np.uint8)
   
    # Iterate over each result in the list
    for r in results:
        # Check if segmentation information is available
        if hasattr(r, 'masks') and r.masks is not None:
            masks = r.masks.data
           
            # Iterate over each mask
            for mask in masks:
                mask = mask.cpu().numpy().astype(np.uint8)
               
                # Resize the mask to match the image dimensions
                mask_resized = cv.resize(mask, (img_width, img_height))
               
                # Apply threshold to create binary mask
                edge_map[mask_resized > 0.5] = 255
   
    return edge_map

def detect_lines_with_hough(edge_map, min_length=100):
    # Detect lines using the Hough Line Transform
    lines = cv.HoughLinesP(edge_map, 1, np.pi / 180, threshold=100, minLineLength=min_length, maxLineGap=10)
   
    if lines is None:
        return []
   
    lines = lines[:, 0, :]  # Reshape the lines array
   
    return lines

def cluster_lines_by_angle_and_distance(lines, eps_angle=0.1, eps_distance=50):
    angles = []
    mid_points = []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1)
        mid_point = ((x1 + x2) / 2, (y1 + y2) / 2)
        angles.append([angle])
        mid_points.append(mid_point)
   
    angle_clustering = DBSCAN(eps=eps_angle, min_samples=4).fit(angles)
    distance_clustering = DBSCAN(eps=eps_distance, min_samples=4).fit(mid_points)
   
    clusters = defaultdict(list)
    for i, label in enumerate(angle_clustering.labels_):
        if label != -1:
            clusters[label].append(lines[i])
   
    final_clusters = defaultdict(list)
    for label, cluster in clusters.items():
        points = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in cluster]
        distance_clustering = DBSCAN(eps=eps_distance, min_samples=2).fit(points)
        for i, sublabel in enumerate(distance_clustering.labels_):
            if sublabel != -1:
                final_clusters[(label, sublabel)].append(cluster[i])
   
    return final_clusters

def compute_average_lines(clusters):
    average_lines = []
    for cluster, lines in clusters.items():
        x1_avg = np.mean([line[0] for line in lines])
        y1_avg = np.mean([line[1] for line in lines])
        x2_avg = np.mean([line[2] for line in lines])
        y2_avg = np.mean([line[3] for line in lines])
        average_lines.append([x1_avg, y1_avg, x2_avg, y2_avg])
   
    return average_lines

def filter_lines_by_length(lines, min_length):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
        if length >= min_length:
            filtered_lines.append(line)
    return filtered_lines

def compute_intersections(lines):
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (int(px), int(py))

    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            point = line_intersection(line1, line2)
            if point:
                intersections.append(point)
    return intersections

def compute_center_of_mass(points, img_width, img_height, center_threshold=0.5):
    # Convert points to numpy array for easier manipulation
    points = np.array(points)

    # Find the center of the image
    center_x = img_width / 2
    center_y = img_height / 2

    # Filter points close to the image center
    filtered_points = []
    for point in points:
        px, py = point
        if abs(px - center_x) <= center_threshold * img_width and abs(py - center_y) <= center_threshold * img_height:
            filtered_points.append(point)

    filtered_points = np.array(filtered_points)

    # Calculate weighted sum of coordinates of filtered points
    total_weight = len(filtered_points)
    sum_x = np.sum(filtered_points[:, 0])
    sum_y = np.sum(filtered_points[:, 1])

    # Calculate center of mass
    if total_weight > 0:
        center_x = sum_x / total_weight
        center_y = sum_y / total_weight
        return (int(center_x), int(center_y))
    else:
        return None

def save_image(image, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    cv.imwrite(filepath, image)

def visualize_lines_and_intersections(frame, lines, intersections, highest_intersection, mask, directory, filename):
    lines_image = frame.copy()
   
    # Overlay the mask in red
    red_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    red_mask[:, :, 1] = 0
    red_mask[:, :, 2] = 0
    overlay = cv.addWeighted(lines_image, 1, red_mask, 0.5, 0)
   
    # Draw the detected lines on the image
    for line in lines:
        x1, y1, x2, y2 = map(int, line)
        cv.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing in green (0, 255, 0)
   
    # Draw the intersections on the image
    for point in intersections:
        px, py = point
        cv.circle(overlay, (px, py), 5, (0, 0, 255), -1)  # Drawing in red (0, 0, 255)
   
    # Highlight the highest intersection point
    if highest_intersection:
        px, py = highest_intersection
        cv.circle(overlay, (px, py), 8, (255, 0, 0), -1)  # Drawing in blue (255, 0, 0)
        cv.putText(overlay, f'Intersection Point ({px}, {py})', (px + 10, py), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
   
    save_image(overlay, directory, filename)

def main(image_path, save_directory):
    # Read the image
    frame = cv.imread(image_path)
    if frame is None:
        print(f"Error: Unable to load image {image_path}")
        return
   
    # Detect edges using YOLOv8
    edge_map = detect_edges_yolov8(frame)
    save_image(edge_map, save_directory, 'edge_map.jpg')
   
    # Detect lines using the Hough Transform
    lines = detect_lines_with_hough(edge_map)
    lines_image = frame.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing in green (0, 255, 0)
    save_image(lines_image, save_directory, 'detected_lines.jpg')
   
    if lines is not None and len(lines) > 0:  # Check if there are any lines
        # Cluster lines by angle and distance
        clusters = cluster_lines_by_angle_and_distance(lines)
       
        # Compute average lines for each cluster
        average_lines = compute_average_lines(clusters)
        average_lines_image = frame.copy()
        for line in average_lines:
            x1, y1, x2, y2 = map(int, line)
            cv.line(average_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing in green (0, 255, 0)
        save_image(average_lines_image, save_directory, 'average_lines.jpg')
       
        # Filter average lines by length
        filtered_lines = filter_lines_by_length(average_lines, min_length=350)
        filtered_lines_image = frame.copy()
        for line in filtered_lines:
            x1, y1, x2, y2 = map(int, line)
            cv.line(filtered_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing in green (0, 255, 0)
        save_image(filtered_lines_image, save_directory, 'filtered_lines.jpg')
       
        # Compute intersections of filtered lines
        intersections = compute_intersections(filtered_lines)
       
        # Calculate center of mass of intersections near the image center
        img_height, img_width = frame.shape[:2]
        center_of_mass = compute_center_of_mass(intersections, img_width, img_height)
       
        # Visualize filtered average lines and intersections
        visualize_lines_and_intersections(frame, filtered_lines, intersections, center_of_mass, edge_map, save_directory, 'lines_and_intersections.jpg')
    else:
        print("No lines detected")

if __name__ == "__main__":
    image_path = r"D:\ALIOUNE\THESEPARIS\SOFTWARE\ZebChipSort\intersectionPoint\test.jpg"
    save_directory = r"D:\ALIOUNE\THESEPARIS\SOFTWARE\ZebChipSort\intersectionPoint\results"
    main(image_path, save_directory)
