import cv2
import numpy as np

# Try to find RealSense cameras
print("Scanning for cameras...")
available_cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

print(f"Found cameras at indices: {available_cameras}")
print("\nTesting all cameras to identify them...")

# Open all available cameras to see what they are
caps = []
for idx in available_cameras:
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    caps.append((idx, cap))

print(f"Opening {len(caps)} cameras")
print("Press 'q' to quit, or press 1/2/3 to select which pair to use for stereo")

try:
    while True:
        frames = []
        for idx, cap in caps:
            ret, frame = cap.read()
            if ret:
                # Resize to standard size
                frame_resized = cv2.resize(frame, (640, 480))
                # Add label
                cv2.putText(frame_resized, f'Camera {idx}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(frame_resized)
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Display all cameras
        if len(frames) >= 3:
            row1 = np.hstack(frames[:2])
            row2 = np.hstack([frames[2], np.zeros((480, 640, 3), dtype=np.uint8)])
            combined = np.vstack([row1, row2])
        elif len(frames) == 2:
            combined = np.hstack(frames)
        else:
            combined = frames[0] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.imshow('All Cameras - Identify your RealSense cameras', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
finally:
    for idx, cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    print("Cameras released")