import sys
import os

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Python path:", sys.path)
print("\nTrying to import pyrealsense2...")

try:
    import pyrealsense2 as rs
    print("✓ pyrealsense2 found!")
    print("  Version:", rs.__version__ if hasattr(rs, '__version__') else "Unknown")
    print("  Location:", rs.__file__)
except ImportError as e:
    print("✗ pyrealsense2 NOT found!")
    print("  Error:", e)

print("\nInstalled packages:")
os.system(f'"{sys.executable}" -m pip list | findstr realsense')
