#!/usr/bin/env python3
"""
Track a point relative to the heart assembly position
More robust than tracking individual features
"""
import pyrealsense2 as rs
import numpy as np
import cv2

print("=== RELATIVE POSITION TRACKER ===")
print("Instructions:")
print("1. Press 'a' to capture the assembly (heart mockup) template")
print("2. Click on the point you want to track")
print("3. Tracking will track that point relative to assembly position")
print("4. Press 'r' to reset")
print("5. Press 'q' to quit")
print()

# Configure RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting camera...")
pipeline.start(config)

# Tracking state
assembly_template = None
assembly_w, assembly_h = 0, 0
target_point_relative = None  # (x, y) relative to assembly top-left
tracking = False

def capture_assembly(frame):
    """Let user select the heart assembly region"""
    print("\nSelect the entire heart assembly region, then press ENTER")
    roi = cv2.selectROI("Select Heart Assembly", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Heart Assembly")
    
    x, y, w, h = roi
    
    if w > 50 and h > 50:  # Ensure reasonable size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = gray[int(y):int(y+h), int(x):int(x+w)]
        return template, w, h, (x, y)
    else:
        print("Assembly region too small or not selected")
        return None, 0, 0, None

def select_target_point(event, x, y, flags, param):
    """Mouse callback to select target point within assembly"""
    global target_point_relative, tracking
    
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, assembly_pos = param
        
        if assembly_pos is not None:
            ax, ay = assembly_pos
            
            # Check if click is within assembly bounds
            if ax <= x <= ax + assembly_w and ay <= y <= ay + assembly_h:
                # Store relative position
                target_point_relative = (x - ax, y - ay)
                tracking = True
                print(f"Target point set at ({x}, {y}) - relative: {target_point_relative}")
            else:
                print("Click inside the assembly region!")
        else:
            print("Capture assembly first with 'a'!")

try:
    cv2.namedWindow('Assembly Tracker')
    
    assembly_pos = None  # Current assembly position
    
    print("\nPress 'a' to capture assembly...")
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        
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
                
                # If we have a target point, draw it
                if tracking and target_point_relative is not None:
                    # Calculate absolute position from relative
                    tx = ax + target_point_relative[0]
                    ty = ay + target_point_relative[1]
                    
                    # Draw target point
                    cv2.circle(display, (tx, ty), 8, (0, 255, 0), -1)
                    cv2.circle(display, (tx, ty), 25, (0, 255, 0), 2)
                    
                    # Draw crosshair
                    cv2.line(display, (tx - 15, ty), (tx + 15, ty), (0, 255, 0), 2)
                    cv2.line(display, (tx, ty - 15), (tx, ty + 15), (0, 255, 0), 2)
                    
                    # Display info
                    cv2.putText(display, f"Tracking: ({tx}, {ty})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    cv2.putText(display, f"Assembly confidence: {max_val:.3f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
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
        
        # Display controls
        cv2.putText(display, "a=assembly | click=target | r=reset | q=quit", 
                   (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Set mouse callback with current params
        cv2.setMouseCallback('Assembly Tracker', select_target_point, 
                            param=(frame, assembly_pos))
        
        cv2.imshow('Assembly Tracker', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Capture assembly
            assembly_template, assembly_w, assembly_h, initial_pos = capture_assembly(frame)
            if assembly_template is not None:
                assembly_pos = initial_pos
                tracking = False
                target_point_relative = None
                print("Assembly captured! Now click on the point you want to track")
        elif key == ord('r'):
            # Reset everything
            assembly_template = None
            assembly_pos = None
            target_point_relative = None
            tracking = False
            print("Reset - press 'a' to capture assembly again")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nTracking stopped.")
