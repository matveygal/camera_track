import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from collections import deque

print("=== DOT TRACKER WITH OPTICAL FLOW + DEPTH MEASUREMENT ===")
print("Instructions:")
print("1. Press 'c' to capture the dot point")
print("2. Tracking will start automatically with depth measurement")
print("3. Press 's' to save data to CSV")
print("4. Press 'p' to show/hide live plot")
print("5. Press 'r' to reset tracking")
print("6. Press 'q' to quit and save")
print()

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

# Optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Tracking state
point = None
old_gray = None
initial_point = None  # Where the dot was first captured
last_good_point = None
lost_frames = 0
max_lost_frames = 10
max_drift = 20  # Maximum allowed drift from initial position in pixels
tracking = False

# Data storage
timestamps = []
distances = []
positions_x = []
positions_y = []
start_time = None

# Rolling median filter settings
FILTER_WINDOW_SIZE = 10  # Number of frames to use for median filtering
distance_buffer = deque(maxlen=FILTER_WINDOW_SIZE)

# Plot settings
show_plot = True
plt.ion()  # Interactive mode
fig, ax = plt.subplots(figsize=(10, 4))
fig.canvas.manager.window.wm_geometry("+700+50")  # Position plot on right side
line, = ax.plot([], [], 'b-', linewidth=2)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Distance (mm)')
ax.set_title('Dot Distance from Camera')
ax.grid(True)

def select_dot_point(frame):
    """Let user select ROI around dot, then find the best feature point"""
    print("\nSelect region around the black dot, then press ENTER or SPACE")
    roi = cv2.selectROI("Select Dot Region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Dot Region")
    
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    
    # Extract ROI
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    
    # Find the best corner/feature point in the ROI
    feature_params = dict(
        maxCorners=1,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    
    corners = cv2.goodFeaturesToTrack(roi_gray, **feature_params)
    
    if corners is None:
        print("Warning: No feature found, using ROI center")
        pt = np.array([[[x + w/2, y + h/2]]], dtype=np.float32)
    else:
        # Convert corner coordinates from ROI to full frame
        pt = corners.copy()
        pt[0][0][0] += x
        pt[0][0][1] += y
    
    print(f"Initial tracking point: ({pt[0][0][0]:.1f}, {pt[0][0][1]:.1f})")
    return pt

def try_reacquire_point(gray, last_pt, search_radius=30):
    """Try to find a good feature point near the last known position"""
    x, y = last_pt[0][0]
    x, y = int(x), int(y)
    
    # Define search region around last known position
    x1 = max(0, x - search_radius)
    y1 = max(0, y - search_radius)
    x2 = min(gray.shape[1], x + search_radius)
    y2 = min(gray.shape[0], y + search_radius)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = gray[y1:y2, x1:x2]
    
    feature_params = dict(
        maxCorners=1,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7
    )
    
    corners = cv2.goodFeaturesToTrack(roi, **feature_params)
    
    if corners is not None:
        # Convert to full frame coordinates
        pt = corners.copy()
        pt[0][0][0] += x1
        pt[0][0][1] += y1
        return pt
    
    return None

def save_data():
    """Save tracking data to CSV"""
    if len(timestamps) == 0:
        print("No data to save!")
        return
    
    filename = f"dot_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
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
    cv2.namedWindow('Optical Flow Tracking with Depth', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Optical Flow Tracking with Depth', 50, 50)
    cv2.resizeWindow('Optical Flow Tracking with Depth', 640, 480)
    
    print("Camera started. Press 'c' to capture dot...")
    
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
        
        # Optical flow tracking
        if point is not None and old_gray is not None:
            # Calculate optical flow
            new_point, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, point, None, **lk_params
            )
            
            if status[0][0] == 1:  # Successfully tracked
                # Check if point has drifted too far from initial position
                x, y = new_point[0][0]
                init_x, init_y = initial_point[0][0]
                drift = np.sqrt((x - init_x)**2 + (y - init_y)**2)
                
                if drift > max_drift:
                    # Reject - drifted too far, probably jumped to ring
                    lost_frames += 1
                    cv2.putText(color_image, f"DRIFT DETECTED ({drift:.1f}px) - Rejecting", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 140, 255), 2)
                    # Try to recover from initial position
                    point = initial_point.copy()
                else:
                    # Accept the new point
                    point = new_point
                    last_good_point = point.copy()
                    lost_frames = 0
                    cx, cy = int(x), int(y)
                    
                    # Get depth at dot location
                    region_size = 5
                    x1 = max(0, cx - region_size)
                    x2 = min(depth_image.shape[1], cx + region_size)
                    y1 = max(0, cy - region_size)
                    y2 = min(depth_image.shape[0], cy + region_size)
                    
                    depth_region = depth_image[y1:y2, x1:x2]
                    depth_value = np.median(depth_region[depth_region > 0])
                    
                    if depth_value > 0:
                        distance_mm = depth_value * depth_scale * 1000
                        
                        # Add to rolling buffer
                        distance_buffer.append(distance_mm)
                        
                        # Apply median filter
                        filtered_distance = np.median(list(distance_buffer))
                        
                        # Record data
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        timestamps.append(elapsed_time)
                        distances.append(filtered_distance)
                        positions_x.append(cx)
                        positions_y.append(cy)
                        
                        # Update plot
                        update_plot()
                        
                        # Draw tracking point
                        cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.circle(color_image, (cx, cy), 20, (0, 255, 0), 2)
                        
                        # Display info
                        cv2.putText(color_image, f"Distance: {filtered_distance:.1f} mm", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(color_image, f"Position: ({cx}, {cy})", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(color_image, f"Error: {error[0][0]:.2f}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(color_image, f"Samples: {len(timestamps)}", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(color_image, f"Time: {elapsed_time:.1f}s", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(color_image, "No depth data", (10, 180),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # Lost tracking - try to recover
                lost_frames += 1
                
                if last_good_point is not None and lost_frames <= max_lost_frames:
                    # Try to reacquire near last known position
                    recovered_point = try_reacquire_point(gray, last_good_point)
                    
                    if recovered_point is not None:
                        # Validate recovered point isn't too far from initial
                        rec_x, rec_y = recovered_point[0][0]
                        init_x, init_y = initial_point[0][0]
                        drift = np.sqrt((rec_x - init_x)**2 + (rec_y - init_y)**2)
                        
                        if drift <= max_drift:
                            point = recovered_point
                            lost_frames = 0
                            x, y = point[0][0]
                            
                            # Draw in yellow to indicate recovery
                            cv2.circle(color_image, (int(x), int(y)), 5, (0, 255, 255), -1)
                            cv2.circle(color_image, (int(x), int(y)), 20, (0, 255, 255), 2)
                            cv2.putText(color_image, f"RECOVERED - X: {x:.1f}, Y: {y:.1f}", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, (0, 255, 255), 2)
                        else:
                            # Recovery found something too far away
                            x, y = initial_point[0][0]
                            cv2.circle(color_image, (int(x), int(y)), 5, (0, 165, 255), -1)
                            cv2.circle(color_image, (int(x), int(y)), 20, (0, 165, 255), 2)
                            cv2.putText(color_image, f"SEARCHING... ({lost_frames}/{max_lost_frames})", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, (0, 165, 255), 2)
                    else:
                        # Keep last good point visible
                        x, y = last_good_point[0][0]
                        cv2.circle(color_image, (int(x), int(y)), 5, (0, 165, 255), -1)
                        cv2.circle(color_image, (int(x), int(y)), 20, (0, 165, 255), 2)
                        cv2.putText(color_image, f"SEARCHING... ({lost_frames}/{max_lost_frames})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 165, 255), 2)
                else:
                    # Give up after max_lost_frames
                    cv2.putText(color_image, "TRACKING LOST - Press 'c' to recapture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 255), 2)
                    point = None
                    last_good_point = None
                    lost_frames = 0
                    tracking = False
        else:
            # No tracking active
            cv2.putText(color_image, "Press 'c' to capture dot", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Update old frame
        old_gray = gray.copy()
        
        # Display controls
        cv2.putText(color_image, "c=capture | s=save | p=plot | r=reset | q=quit", 
                   (10, color_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Optical Flow Tracking with Depth', color_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture new point
            point = select_dot_point(color_image)
            if point is not None:
                old_gray = gray.copy()
                initial_point = point.copy()
                last_good_point = point.copy()
                lost_frames = 0
                tracking = True
                start_time = datetime.now()
                print(f"✓ Point captured! Tracking started at {start_time.strftime('%H:%M:%S')}")
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
            point = None
            old_gray = None
            initial_point = None
            last_good_point = None
            lost_frames = 0
            tracking = False
            print("Tracking reset.")
            
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
