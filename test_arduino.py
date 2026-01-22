#!/usr/bin/env python3
"""
Test Arduino servo control via serial.
Upload arduino_servo_control.ino to Arduino first.
Servo should be connected to pin 9.
"""

import serial
import serial.tools.list_ports
import time

def find_arduino():
    """Find Arduino USB port automatically."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        device = port.device.lower()
        
        # Check for Arduino/Elegoo in description or common device patterns
        if any(keyword in desc for keyword in ['arduino', 'elegoo', 'ch340', 'ch341', 'ftdi']):
            print(f"Found Arduino on {port.device}")
            return port.device
        # macOS patterns
        if 'usbmodem' in device or 'usbserial' in device:
            print(f"Found Arduino on {port.device}")
            return port.device
    return None

def test_connection():
    """Test Arduino servo control."""
    # Find Arduino port
    port = find_arduino()
    
    if not port:
        print("Arduino not found. Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        return
    
    try:
        # Open serial connection (9600 baud is Arduino default)
        ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        
        print(f"Connected to Arduino on {port}")
        
        # Wait for ready message
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"Arduino says: {response}")
        
        print("\nSpinning servo through full range...")
        
        # Sweep servo from 0 to 180 and back
        for angle in list(range(0, 181, 15)) + list(range(180, -1, -15)):
            ser.write(f"{angle}\n".encode())
            time.sleep(0.3)
            
            # Read response
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(response)
        
        # Return to center
        ser.write(b"90\n")
        time.sleep(0.3)
        print("\nServo returned to center position")
        
        ser.close()
        print("Test complete!")
        
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    print("Arduino Servo Test")
    print("=" * 50)
    print("Make sure you've uploaded arduino_servo_control.ino first!")
    print()
    test_connection()
