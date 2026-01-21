import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from collections import deque

print("=== RELATIVE POSITION TRACKER WITH DEPTH MEASUREMENT ===")
print("Instructions:")
print("1. Press 'a' to capture the heart assembly")
print("2. Click on the point you want to track")
print("3. Depth measurement starts automatically")
print("4. Press 's' to save data to CSV")
print("5. Press 'r' to reset tracking")
print("6. Press 'q' to quit and save (plot will be generated)")
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

# Tracking state
assembly_template = None
assembly_w, assembly_h = 0, 0
target_point_relative = None  # (x, y) relative to assembly top-left
tracking = False

# Data storage
timestamps = []
distances = []
positions_x = []
positions_y = []
start_time = None

# Rolling median filter settings
FILTER_WINDOW_SIZE = 20  # Increased from 10 to better smooth 0.5mm jumps
distance_buffer = deque(maxlen=FILTER_WINDOW_SIZE)

def capture_assembly(frame):
    """Let user select the heart assembly region"""
    print("\nSelect the entire heart assembly region, then press ENTER")
    roi = cv2.selectROI("Select Heart Assembly", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Heart Assembly")
    
    x, y, w, h = roi
    
    if w > 50 and h > 50:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = gray[int(y):int(y+h), int(x):int(x+w)]
        return template, w, h, (x, y)
    else:
        print("Assembly region too small or not selected")
        return None, 0, 0, None

def select_target_point(event, x, y, flags, param):
    """Mouse callback to select target point within assembly"""
    global target_point_relative, tracking, start_time, timestamps, distances, positions_x, positions_y, distance_buffer
    
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, assembly_pos = param
        
        if assembly_pos is not None:
            ax, ay = assembly_pos
            
            if ax <= x <= ax + assembly_w and ay <= y <= ay + assembly_h:
                target_point_relative = (x - ax, y - ay)
                tracking = True
                start_time = datetime.now()
                
                # Clear all previous data
                timestamps.clear()
                distances.clear()
                positions_x.clear()
                positions_y.clear()
                distance_buffer.clear()
                
                print(f"Target point set at ({x}, {y}) - relative: {target_point_relative}")
                print(f"Tracking started at {start_time.strftime('%H:%M:%S')} - previous data cleared")
            else:
                print("Click inside the assembly region!")
        else:
            print("Capture assembly first with 'a'!")
def save_data():
    """Save tracking data to CSV"""
    if len(timestamps) == 0:
        print("No data to save!")
        return
    
    filename = f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
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

def draw_graph_on_frame(frame, width=280, height=140):
    """Draw a real-time graph directly on the video frame using OpenCV"""
    if len(distances) < 2:
        return
    
    # Position in bottom-right corner
    x_offset = frame.shape[1] - width - 10
    y_offset = frame.shape[0] - height - 35  # Leave room for controls text
    
    # Create graph background
    cv2.rectangle(frame, (x_offset, y_offset), 
                 (x_offset + width, y_offset + height), 
                 (40, 40, 40), -1)
    cv2.rectangle(frame, (x_offset, y_offset), 
                 (x_offset + width, y_offset + height), 
                 (100, 100, 100), 2)
    
    # Calculate scaling
    min_dist = min(distances)
    max_dist = max(distances)
    dist_range = max_dist - min_dist if max_dist > min_dist else 1
    
    time_range = timestamps[-1] - timestamps[0] if timestamps[-1] > timestamps[0] else 1
    
    # Draw grid lines
    for i in range(5):
        y = y_offset + int((i / 4) * height)
        cv2.line(frame, (x_offset, y), (x_offset + width, y), (60, 60, 60), 1)
    
    # Draw graph line
    points = []
    for i in range(len(distances)):
        x = x_offset + int(((timestamps[i] - timestamps[0]) / time_range) * width)
        y = y_offset + height - int(((distances[i] - min_dist) / dist_range) * height)
        points.append((x, y))
    
    # Draw lines connecting points
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)
    
    # Draw labels
    cv2.putText(frame, f"Min: {min_dist:.1f} mm", 
               (x_offset + 5, y_offset + height - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, f"Max: {max_dist:.1f} mm", 
               (x_offset + 5, y_offset + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, f"Time: {timestamps[-1]:.1f}s", 
               (x_offset + width - 80, y_offset + height - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


try:
    # Create camera window and position it on left side
    cv2.namedWindow('Assembly Tracker with Depth', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Assembly Tracker with Depth', 50, 50)
    cv2.resizeWindow('Assembly Tracker with Depth', 640, 480)
    
    assembly_pos = None
    
    print("Camera started. Press 'a' to capture assembly...")
    
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
        display = color_image.copy()
        
        # Track assembly position using template matching
        if assembly_template is not None:
            result = cv2.matchTemplate(gray, assembly_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.5:  # Good match
                assembly_pos = max_loc
                ax, ay = assembly_pos
                
                # Draw assembly bounding box
                cv2.rectangle(display, (ax, ay), 
                            (ax + assembly_w, ay + assembly_h), 
                            (255, 0, 0), 2)
                
                # If we have a target point, track and measure depth
                if tracking and target_point_relative is not None:
                    # Calculate absolute position from relative
                    tx = int(ax + target_point_relative[0])
                    ty = int(ay + target_point_relative[1])
                    
                    # Get depth at target location
                    region_size = 5
                    x1 = max(0, tx - region_size)
                    x2 = min(depth_image.shape[1], tx + region_size)
                    y1 = max(0, ty - region_size)
                    y2 = min(depth_image.shape[0], ty + region_size)
                    
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
                        positions_x.append(tx)
                        positions_y.append(ty)
                        
                        # Draw target point
                        cv2.circle(display, (tx, ty), 8, (0, 255, 0), -1)
                        cv2.circle(display, (tx, ty), 25, (0, 255, 0), 2)
                        
                        # Draw crosshair
                        cv2.line(display, (tx - 15, ty), (tx + 15, ty), (0, 255, 0), 2)
                        cv2.line(display, (tx, ty - 15), (tx, ty + 15), (0, 255, 0), 2)
                        
                        # Display info
                        cv2.putText(display, f"Distance: {filtered_distance:.1f} mm", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                        cv2.putText(display, f"Position: ({tx}, {ty})", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
                        cv2.putText(display, f"Assembly conf: {max_val:.3f}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
                        cv2.putText(display, f"Samples: {len(timestamps)}", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
                        cv2.putText(display, f"Time: {elapsed_time:.1f}s", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, "No depth data", (10, 180),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(display, "Assembly found - Click to select target point", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 0), 2)
            else:
                assembly_pos = None
                cv2.putText(display, f"Assembly lost (conf: {max_val:.3f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Press 'a' to capture assembly", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)
        
        # Draw real-time graph if we have data
        if len(distances) > 1:
            draw_graph_on_frame(display)
        
        # Display controls
        cv2.putText(display, "a=assembly | click=target | s=save | r=reset | q=quit", 
                   (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Set mouse callback with current params
        cv2.setMouseCallback('Assembly Tracker with Depth', select_target_point, 
                            param=(color_image, assembly_pos))
        
        cv2.imshow('Assembly Tracker with Depth', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Capture assembly
            assembly_template, assembly_w, assembly_h, initial_pos = capture_assembly(color_image)
            if assembly_template is not None:
                assembly_pos = initial_pos
                tracking = False
                target_point_relative = None
                print("Assembly captured! Now click on the point you want to track")
        elif key == ord('s'):
            save_data()
        elif key == ord('r'):
            # Reset everything
            assembly_template = None
            assembly_pos = None
            target_point_relative = None
            tracking = False
            print("Reset - press 'a' to capture assembly again")
    
    while True:
        # Get aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue
        
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Save data automatically on exit if we have data
    if len(timestamps) > 0:
        save_data()
        
        # Generate and save plot (no display to avoid hanging)
        if len(timestamps) > 1:
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, distances, 'b-', linewidth=2)
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Distance (mm)', fontsize=12)
            plt.title('Target Point Distance Over Time', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_filename = f"tracking_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename, dpi=150)
            plt.close('all')  # Close without displaying
            print(f"✓ Plot saved to {plot_filename}")
    
    print("\nTracking complete!")

