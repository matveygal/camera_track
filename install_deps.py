import sys
import subprocess

print(f"Installing packages using: {sys.executable}")
print("-" * 50)

packages = ['pyrealsense2', 'opencv-python', 'numpy']

for package in packages:
    print(f"\nInstalling {package}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

print("\n" + "=" * 50)
print("Installation complete!")
print("Now run: python test.py")
