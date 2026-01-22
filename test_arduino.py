#!/usr/bin/env python3
"""
Test Arduino stepper motor control via serial.
Upload arduino_servo_control.ino to Arduino first.
Stepper motor should be connected to pins 8, 9, 10, 11 (IN1-IN4).
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
    """Test Arduino stepper motor control."""
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
        time.sleep(0.5)
        while ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"Arduino says: {response}")
        
        print("\nTesting stepper motor...")
        print("One full rotation forward (2048 steps)...")
        
        # Full rotation forward
        ser.write(b"2048\n")
        time.sleep(0.5)
        
        # Read responses
        for _ in range(10):  # Wait up to 5 seconds
            time.sleep(0.5)
            while ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                if response:
                    print(f"  {response}")
                if "Done" in response:
                    break
        
        print("\nOne full rotation backward (-2048 steps)...")
        
        # Full rotation backward
        ser.write(b"-2048\n")
        time.sleep(0.5)
        
        # Read responses
        for _ in range(10):  # Wait up to 5 seconds
            time.sleep(0.5)
            while ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                if response:
                    print(f"  {response}")
                if "Done" in response:
                    break
        
        print("\nHalf rotation forward (1024 steps)...")
        ser.write(b"1024\n")
        time.sleep(0.5)
        
        for _ in range(6):
            time.sleep(0.5)
            while ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                if response:
                    print(f"  {response}")
                if "Done" in response:
                    break
        
        ser.close()
        print("\nTest complete!")
        
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    print("Arduino Stepper Motor Test")
    print("=" * 50)
    print("Make sure you've uploaded arduino_servo_control.ino first!")
    print()
    test_connection()
