import pyrealsense2 as rs
import numpy as np
import cv2

print("=== DOT TRACKER WITH TEMPLATE MATCHING ===")
print("Instructions:")
print("1. Camera will start - position your heart with the dot visible")
print("2. Press 'c' to capture the dot template (select region with mouse)")
print("3. Template matching will track the dot automatically")
print("4. Press 'q' to quit")
print()

# Configure streams  
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Template and tracking state
template = None
template_w, template_h = 0, 0
tracking = False
roi_selected = False
selecting_roi = False

def select_template():
    """Capture a single frame and let user select the dot"""
    global template, template_w, template_h, tracking
    
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return False
    
    frame = np.asanyarray(color_frame.get_data())
    
    print("\nSelect the black dot region with your mouse, then press ENTER or SPACE")
    print("Press 'c' to cancel")
    
    # Let user select ROI
    roi = cv2.selectROI("Select Dot Template", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Dot Template")
    
    x, y, w, h = roi
    
    if w > 0 and h > 0:
        # Extract template
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = gray[int(y):int(y+h), int(x):int(x+w)]
        template_w, template_h = w, h
        tracking = True
        print(f"✓ Template captured! Size: {w}x{h}")
        return True
    else:
        print("✗ No template selected")
        return False

try:
    print("Camera started. Press 'c' to capture dot template...")
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Template matching if we have a template
        if tracking and template is not None:
            # Apply template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Check if match is good enough (threshold)
            if max_val > 0.6:  # Confidence threshold
                # Get top-left corner of match
                top_left = max_loc
                bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
                
                # Calculate center
                cx = top_left[0] + template_w // 2
                cy = top_left[1] + template_h // 2
                
                # Draw bounding box
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), 
                              cv2.MARKER_CROSS, 20, 2)
                
                # Display info
                cv2.putText(frame, f"Dot: ({cx}, {cy})", 
                           (cx + 20, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {max_val:.3f}", 
                           (cx + 20, cy + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, "TRACKING", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Lost tracking (conf: {max_val:.3f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 'c' to capture dot template", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display
        cv2.imshow('Dot Tracker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not tracking:
            select_template()
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nCamera stopped")
