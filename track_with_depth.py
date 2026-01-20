import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from collections import deque

print("=== DOT TRACKER WITH DEPTH MEASUREMENT ===")
print("Instructions:")
print("1. Press 'c' to capture the dot template")
print("2. Tracking will start automatically with depth measurement")
print("3. Press 's' to save data to CSV")
print("4. Press 'p' to show/hide live plot")
print("5. Press 'q' to quit and save")
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

# Template and tracking state
template = None
template_gray_normalized = None  # Store normalized template
template_w, template_h = 0, 0
tracking = False

# CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

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
line, = ax.plot([], [], 'b-', linewidth=2)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Distance (mm)')
ax.set_title('Dot Distance from Camera')
ax.grid(True)

def select_template():
    """Capture a single frame and let user select the dot"""
    global template, template_gray_normalized, template_w, template_h, tracking, start_time
    
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    
    if not color_frame:
        return False
    
    frame = np.asanyarray(color_frame.get_data())
    
    print("\nSelect the black dot region, then press ENTER")
    roi = cv2.selectROI("Select Dot Template", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Dot Template")
    
    x, y, w, h = roi
    
    if w > 0 and h > 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to normalize lighting on template
        gray_normalized = clahe.apply(gray)
        
        template = gray[int(y):int(y+h), int(x):int(x+w)]
        template_gray_normalized = gray_normalized[int(y):int(y+h), int(x):int(x+w)]
        template_w, template_h = w, h
        tracking = True
        start_time = datetime.now()
        print(f"✓ Template captured! Tracking started at {start_time.strftime('%H:%M:%S')}")
        print(f"  Using CLAHE lighting normalization for robust tracking")
        return True
    else:
        print("✗ No template selected")
        return False

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
    print("Camera started. Press 'c' to capture dot template...")
    
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
        
        # Apply CLAHE to normalize lighting (same as template)
        gray_normalized = clahe.apply(gray)
        
        # Template matching with lighting-normalized images
        if tracking and template_gray_normalized is not None:
            result = cv2.matchTemplate(gray_normalized, template_gray_normalized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.6:  # Good match
                # Get dot position
                top_left = max_loc
                cx = top_left[0] + template_w // 2
                cy = top_left[1] + template_h // 2
                
                # Get depth at dot location (average small region for stability)
                region_size = 5
                x1 = max(0, cx - region_size)
                x2 = min(depth_image.shape[1], cx + region_size)
                y1 = max(0, cy - region_size)
                y2 = min(depth_image.shape[0], cy + region_size)
                
                depth_region = depth_image[y1:y2, x1:x2]
                depth_value = np.median(depth_region[depth_region > 0])  # Use median, ignore zeros
                
                if depth_value > 0:
                    distance_mm = depth_value * depth_scale * 1000
                    
                    # Add to rolling buffer
                    distance_buffer.append(distance_mm)
                    
                    # Apply median filter to smooth out spikes
                    filtered_distance = np.median(list(distance_buffer))
                    
                    # Record data (using filtered distance)
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    timestamps.append(elapsed_time)
                    distances.append(filtered_distance)
                    positions_x.append(cx)
                    positions_y.append(cy)
                    
                    # Update plot
                    update_plot()
                    
                    # Draw visualization
                    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
                    cv2.rectangle(color_image, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.drawMarker(color_image, (cx, cy), (0, 255, 0), 
                                  cv2.MARKER_CROSS, 20, 2)
                    
                    # Display info
                    cv2.putText(color_image, f"Distance: {filtered_distance:.1f} mm (filtered)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Raw: {distance_mm:.1f} mm", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                    cv2.putText(color_image, f"Position: ({cx}, {cy})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Samples: {len(timestamps)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Time: {elapsed_time:.1f}s", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(color_image, "No depth data at dot", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(color_image, f"Lost tracking (conf: {max_val:.3f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(color_image, "Press 'c' to capture dot template", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display controls
        cv2.putText(color_image, "c=capture | s=save | p=plot | q=quit", 
                   (10, color_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Dot Tracking with Depth', color_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not tracking:
            select_template()
        elif key == ord('s'):
            save_data()
        elif key == ord('p'):
            show_plot = not show_plot
            if show_plot:
                plt.show()
            else:
                plt.close()
            
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
