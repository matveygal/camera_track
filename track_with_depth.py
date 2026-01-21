#!/usr/bin/env python3
"""
Track a point of interest relative to a stable reference frame (heart assembly)
WITH depth measurement, data export, and live plotting
No black dot needed - works on any surface!
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from collections import deque

print("=== RELATIVE POSITION TRACKING WITH DEPTH MEASUREMENT ===")
print("Instructions:")
print("1. Press 'c' to select heart assembly and point of interest")
print("2. Tracking will start with depth measurement and live plotting")
print("3. Press 's' to save data to CSV")
print("4. Press 'p' to show/hide live plot")
print("5. Press 'r' to reset tracking")
print("6. Press 'q' to quit and save")
print()

# Feature detection parameters
feature_params = dict(
    maxCorners=50,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7
)

# Optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Configure streams  
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create alignment object
align_to = rs.stream.color
align = rs.align(align_to)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale}")

# Tracking state
reference_features = None
relative_position = None
old_gray = None
tracking = False

# Data storage
timestamps = []
distances = []
positions_x = []
positions_y = []
start_time = None

# Rolling median filter settings
FILTER_WINDOW_SIZE = 10
distance_buffer = deque(maxlen=FILTER_WINDOW_SIZE)

# Plot settings
show_plot = True
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
fig.canvas.manager.window.wm_geometry("+700+50")
line, = ax.plot([], [], 'b-', linewidth=2)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Distance (mm)')
ax.set_title('Point Distance from Camera')
ax.grid(True)

def select_reference_frame_and_point(frame):
    """Let user select reference region (heart assembly) and point of interest"""
    print("\n=== STEP 1: Select the heart assembly region ===")
    print("Draw a box around the entire heart assembly, then press ENTER")
    
    roi = cv2.selectROI("Select Heart Assembly", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Heart Assembly")
    
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No region selected!")
        return None, None, None
    
    # Find good features in the assembly region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h, x:x+w]
    
    corners = cv2.goodFeaturesToTrack(roi_gray, **feature_params)
    
    if corners is None or len(corners) < 4:
        print("Not enough features found in assembly region!")
        return None, None, None
    
    # Convert corners to full frame coordinates
    corners[:, 0, 0] += x
    corners[:, 0, 1] += y
    
    print(f"Found {len(corners)} tracking features in assembly")
    
    # Draw features on frame
    vis_frame = frame.copy()
    for corner in corners:
        px, py = corner[0]
        cv2.circle(vis_frame, (int(px), int(py)), 3, (0, 255, 0), -1)
    
    print("\n=== STEP 2: Click on the point you want to track ===")
    print("Click on the specific point of interest (e.g., surface point)")
    
    clicked_point = [None]
    
    def mouse_callback(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = np.array([mx, my], dtype=np.float32)
            cv2.circle(vis_frame, (mx, my), 8, (0, 0, 255), -1)
            cv2.imshow("Click Point of Interest", vis_frame)
    
    cv2.namedWindow("Click Point of Interest")
    cv2.setMouseCallback("Click Point of Interest", mouse_callback)
    cv2.imshow("Click Point of Interest", vis_frame)
    
    print("Waiting for click...")
    while clicked_point[0] is None:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyWindow("Click Point of Interest")
            return None, None, None
    
    cv2.waitKey(500)
    cv2.destroyWindow("Click Point of Interest")
    
    point_of_interest = clicked_point[0]
    print(f"Point of interest: ({point_of_interest[0]:.1f}, {point_of_interest[1]:.1f})")
    
    # Calculate relative position of point to reference features
    centroid = np.mean(corners[:, 0], axis=0)
    relative_pos = point_of_interest - centroid
    
    print(f"Relative position from feature centroid: ({relative_pos[0]:.1f}, {relative_pos[1]:.1f})")
    
    return corners, relative_pos, (x, y, w, h)

def save_data():
    """Save tracking data to CSV"""
    if len(timestamps) == 0:
        print("No data to save!")
        return
    
    filename = f"tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time (s)', 'Distance (mm)', 'X (px)', 'Y (px)'])
        for i in range(len(timestamps)):
            writer.writerow([
                f"{timestamps[i]:.3f}",
                f"{distances[i]:.2f}",
                positions_x[i],
                positions_y[i]
            ])
    
    print(f"✓ Data saved to {filename}")
    print(f"  Total samples: {len(timestamps)}")
    print(f"  Duration: {timestamps[-1]:.2f} seconds")
    print(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} mm")

def update_plot():
    """Update the live plot"""
    if len(timestamps) > 1 and show_plot:
        line.set_data(timestamps, distances)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

try:
    # Create camera window and position it on left side
    cv2.namedWindow('Relative Position Tracking with Depth', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Relative Position Tracking with Depth', 50, 50)
    cv2.resizeWindow('Relative Position Tracking with Depth', 640, 480)
    
    print("Camera started. Press 'c' to start tracking...")
    
    while True:
        # Get aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue
        
        # Convert to numpy
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        if tracking and reference_features is not None and old_gray is not None:
            # Track reference features
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, reference_features, None, **lk_params
            )
            
            # Filter out lost features
            good_new = new_features[status == 1]
            
            if len(good_new) < 4:
                cv2.putText(color_image, "LOST TRACKING - Press 'c' to recapture", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 255), 2)
                tracking = False
            else:
                # Update reference features
                reference_features = good_new.reshape(-1, 1, 2)
                
                # Calculate new centroid
                centroid = np.mean(reference_features[:, 0], axis=0)
                
                # Calculate point of interest position
                point_of_interest = centroid + relative_position
                px, py = int(point_of_interest[0]), int(point_of_interest[1])
                
                # Get depth at point location
                region_size = 5
                x1 = max(0, px - region_size)
                x2 = min(depth_image.shape[1], px + region_size)
                y1 = max(0, py - region_size)
                y2 = min(depth_image.shape[0], py + region_size)
                
                depth_region = depth_image[y1:y2, x1:x2]
                depth_value = np.median(depth_region[depth_region > 0])
                
                if depth_value > 0:
                    distance_mm = depth_value * depth_scale * 1000
                    
                    # Add to rolling buffer and apply median filter
                    distance_buffer.append(distance_mm)
                    filtered_distance = np.median(list(distance_buffer))
                    
                    # Record data
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    timestamps.append(elapsed_time)
                    distances.append(filtered_distance)
                    positions_x.append(px)
                    positions_y.append(py)
                    
                    # Update plot
                    update_plot()
                    
                    # Draw tracked features (green)
                    for feature in reference_features:
                        fx, fy = feature[0]
                        cv2.circle(color_image, (int(fx), int(fy)), 3, (0, 255, 0), -1)
                    
                    # Draw centroid (blue)
                    cv2.circle(color_image, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)
                    
                    # Draw point of interest (red)
                    cv2.circle(color_image, (px, py), 8, (0, 0, 255), -1)
                    cv2.circle(color_image, (px, py), 25, (0, 0, 255), 2)
                    cv2.drawMarker(color_image, (px, py), (0, 255, 255), 
                                  cv2.MARKER_CROSS, 30, 2)
                    
                    # Display info
                    cv2.putText(color_image, f"Distance: {filtered_distance:.1f} mm", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(color_image, f"Point: ({px}, {py})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(color_image, f"Features: {len(reference_features)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Samples: {len(timestamps)}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Time: {elapsed_time:.1f}s", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(color_image, "No depth data", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(color_image, "Press 'c' to start tracking", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Update old frame
        old_gray = gray.copy()
        
        # Display controls
        cv2.putText(color_image, "c=capture | s=save | p=plot | r=reset | q=quit", 
                   (10, color_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Relative Position Tracking with Depth', color_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture reference and point
            result = select_reference_frame_and_point(color_image)
            if result[0] is not None:
                reference_features, relative_position, roi = result
                old_gray = gray.copy()
                tracking = True
                start_time = datetime.now()
                print(f"✓ Tracking started at {start_time.strftime('%H:%M:%S')}")
        elif key == ord('s'):
            save_data()
        elif key == ord('p'):
            show_plot = not show_plot
            if show_plot:
                plt.show()
            else:
                plt.close()
        elif key == ord('r'):
            # Reset tracking
            reference_features = None
            relative_position = None
            old_gray = None
            tracking = False
            print("Tracking reset")
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Save data automatically on exit if we have data
    if len(timestamps) > 0:
        save_data()
    
    # Show final plot
    if len(timestamps) > 1:
        plt.ioff()
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, distances, 'b-', linewidth=2)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Distance (mm)', fontsize=12)
        plt.title('Dot Distance Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_filename = f"dot_tracking_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"✓ Plot saved to {plot_filename}")
        plt.show()
    
    print("\nTracking complete!")
