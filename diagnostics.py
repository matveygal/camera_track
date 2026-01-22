"""
Diagnostic and testing utilities for the heart tracking system.
Use this to verify camera setup and test tracking performance.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time


def test_camera_connection():
    """
    Test if RealSense camera is properly connected and accessible.
    """
    print("=" * 60)
    print("Testing Intel RealSense 435i Camera Connection")
    print("=" * 60)
    
    try:
        # Create context to query devices
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("[ERROR] No RealSense devices found!")
            print("Please check:")
            print("  1. Camera is plugged in via USB")
            print("  2. USB cable is functioning properly")
            print("  3. Camera has power (LED should be on)")
            return False
        
        print(f"[SUCCESS] Found {len(devices)} RealSense device(s)")
        
        for i, device in enumerate(devices):
            print(f"\nDevice {i}:")
            print(f"  Name: {device.get_info(rs.camera_info.name)}")
            print(f"  Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"  Firmware: {device.get_info(rs.camera_info.firmware_version)}")
            
            # List available sensors
            sensors = device.query_sensors()
            print(f"  Sensors: {len(sensors)}")
            for sensor in sensors:
                print(f"    - {sensor.get_info(rs.camera_info.name)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to query RealSense devices: {e}")
        return False


def test_camera_stream():
    """
    Test camera streaming and display a live feed.
    """
    print("\n" + "=" * 60)
    print("Testing Camera Stream (RGB Only)")
    print("=" * 60)
    print("Press 'q' to exit")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure RGB stream only
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start pipeline
        print("[INFO] Starting pipeline...")
        pipeline.start(config)
        print("[SUCCESS] Camera stream started")
        
        frame_count = 0
        start_time = time.time()
        fps_history = []
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                current_fps = frame_count / elapsed
                fps_history.append(current_fps)
                
                if len(fps_history) > 30:
                    fps_history.pop(0)
                
                avg_fps = np.mean(fps_history)
            
            # Add info overlay
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: 640x480", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to exit", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('RealSense Camera Test', frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n[INFO] Average FPS: {avg_fps:.2f}")
        print(f"[INFO] Total frames captured: {frame_count}")
        
    except Exception as e:
        print(f"[ERROR] Stream test failed: {e}")
        return False
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped")
    
    return True


def test_sift_features():
    """
    Test SIFT feature detection on camera feed.
    """
    print("\n" + "=" * 60)
    print("Testing SIFT Feature Detection")
    print("=" * 60)
    print("Press 'q' to exit")
    print("This test shows how many SIFT features can be detected")
    print("More features = better tracking potential")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=1000)
    
    try:
        pipeline.start(config)
        print("[SUCCESS] Camera started")
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect SIFT keypoints
            start_time = time.time()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            detection_time = (time.time() - start_time) * 1000  # ms
            
            # Draw keypoints
            frame_with_keypoints = cv2.drawKeypoints(
                frame, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Add info
            cv2.putText(frame_with_keypoints, f"Features detected: {len(keypoints)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_with_keypoints, f"Detection time: {detection_time:.1f}ms",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Color-code feature density
            if len(keypoints) < 50:
                status_color = (0, 0, 255)  # Red - poor
                status_text = "POOR - Need more texture"
            elif len(keypoints) < 150:
                status_color = (0, 165, 255)  # Orange - fair
                status_text = "FAIR - May have tracking issues"
            elif len(keypoints) < 300:
                status_color = (0, 255, 255)  # Yellow - good
                status_text = "GOOD - Should track well"
            else:
                status_color = (0, 255, 0)  # Green - excellent
                status_text = "EXCELLENT - Optimal for tracking"
            
            cv2.putText(frame_with_keypoints, status_text,
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame_with_keypoints, "Press 'q' to exit",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('SIFT Feature Detection Test', frame_with_keypoints)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"[ERROR] SIFT test failed: {e}")
        return False
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] SIFT test completed")
    
    return True


def analyze_heart_motion():
    """
    Analyze motion characteristics in the video feed.
    Useful for determining if motion is periodic and setting EKF parameters.
    """
    print("\n" + "=" * 60)
    print("Heart Motion Analysis")
    print("=" * 60)
    print("This tool analyzes motion to help tune EKF parameters")
    print("Click on a point to track its motion")
    print("Press 'q' to exit")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    selected_point = None
    motion_history = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point, motion_history
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = np.array([x, y], dtype=np.float32)
            motion_history = []
            print(f"[INFO] Tracking point at ({x}, {y})")
    
    cv2.namedWindow('Motion Analysis')
    cv2.setMouseCallback('Motion Analysis', mouse_callback)
    
    try:
        pipeline.start(config)
        
        prev_frame = None
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            display_frame = frame.copy()
            
            if selected_point is not None and prev_frame is not None:
                # Use optical flow to track the point
                new_point, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_frame, gray, 
                    selected_point.reshape(-1, 1, 2),
                    None
                )
                
                if status[0] == 1:
                    new_pt = new_point[0][0]
                    
                    # Calculate displacement
                    displacement = np.linalg.norm(new_pt - selected_point)
                    motion_history.append(displacement)
                    
                    # Keep history limited
                    if len(motion_history) > 300:
                        motion_history.pop(0)
                    
                    # Draw tracking
                    cv2.circle(display_frame, tuple(new_pt.astype(int)), 10, (0, 255, 0), 2)
                    cv2.line(display_frame, 
                            tuple(selected_point.astype(int)),
                            tuple(new_pt.astype(int)),
                            (0, 255, 255), 2)
                    
                    # Update point
                    selected_point = new_pt
                    
                    # Calculate statistics
                    if len(motion_history) > 10:
                        avg_motion = np.mean(motion_history[-30:])
                        max_motion = np.max(motion_history[-30:])
                        std_motion = np.std(motion_history[-30:])
                        
                        # Display analysis
                        cv2.putText(display_frame, f"Avg motion: {avg_motion:.2f} px/frame",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Max motion: {max_motion:.2f} px/frame",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Std dev: {std_motion:.2f} px",
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Recommendations
                        if avg_motion > 5:
                            cv2.putText(display_frame, "High motion - use lower process noise",
                                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        if std_motion > 3:
                            cv2.putText(display_frame, "Noisy motion - use higher measurement noise",
                                       (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(display_frame, "Click to select a point to track",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Motion Analysis', display_frame)
            
            prev_frame = gray.copy()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"[ERROR] Motion analysis failed: {e}")
        return False
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    return True


def run_all_tests():
    """
    Run all diagnostic tests in sequence.
    """
    print("\n" + "=" * 60)
    print("Running Full Diagnostic Suite")
    print("=" * 60)
    
    # Test 1: Camera connection
    if not test_camera_connection():
        print("\n[FAILED] Camera connection test failed!")
        print("Cannot proceed with other tests.")
        return False
    
    input("\nPress Enter to continue to stream test...")
    
    # Test 2: Camera stream
    if not test_camera_stream():
        print("\n[FAILED] Camera stream test failed!")
        return False
    
    input("\nPress Enter to continue to SIFT test...")
    
    # Test 3: SIFT features
    if not test_sift_features():
        print("\n[FAILED] SIFT feature test failed!")
        return False
    
    input("\nPress Enter to continue to motion analysis...")
    
    # Test 4: Motion analysis
    if not analyze_heart_motion():
        print("\n[FAILED] Motion analysis failed!")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All diagnostic tests completed!")
    print("=" * 60)
    print("\nYour system is ready for heart tracking.")
    print("Run 'python heart_tracker.py' to start tracking.")
    
    return True


if __name__ == "__main__":
    print("Heart Tracking System - Diagnostic Tool")
    print("=" * 60)
    print("\nAvailable tests:")
    print("  1. Camera connection test")
    print("  2. Camera stream test")
    print("  3. SIFT feature detection test")
    print("  4. Motion analysis")
    print("  5. Run all tests")
    print("  q. Quit")
    
    choice = input("\nSelect test (1-5, q): ").strip()
    
    if choice == '1':
        test_camera_connection()
    elif choice == '2':
        test_camera_stream()
    elif choice == '3':
        test_sift_features()
    elif choice == '4':
        analyze_heart_motion()
    elif choice == '5':
        run_all_tests()
    elif choice.lower() == 'q':
        print("Exiting...")
    else:
        print("Invalid choice!")
