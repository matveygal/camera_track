#!/usr/bin/env python3
"""
Test Arduino serial connection by blinking LED.
Connect Arduino Uno via USB and upload the standard Blink sketch.
"""

import serial
import serial.tools.list_ports
import time

def find_arduino():
    """Find Arduino USB port automatically."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Arduino Uno typically shows as usbmodem or usbserial on macOS
        if 'usbmodem' in port.device or 'usbserial' in port.device:
            print(f"Found Arduino on {port.device}")
            return port.device
    return None

def test_connection():
    """Test Arduino serial connection."""
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
        print("Sending test commands...")
        
        # Send some test data
        for i in range(5):
            message = f"Test {i}\n"
            ser.write(message.encode())
            print(f"Sent: {message.strip()}")
            
            # Read response if available
            time.sleep(0.5)
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"Received: {response}")
        
        ser.close()
        print("\nConnection test complete!")
        
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    print("Arduino Serial Test")
    print("=" * 50)
    test_connection()
