/*
 * Arduino Stepper Motor Control
 * Upload this sketch to your Arduino Uno/Elegoo
 * Stepper motor connected to pins 8, 9, 10, 11 (IN1-IN4)
 * Serial commands: number of steps (positive or negative)
 */

#include <Stepper.h>

// 2048 steps per revolution for 28BYJ-48 stepper with gear reduction
const int stepsPerRevolution = 2048;

// Initialize stepper on pins IN1=8, IN3=9, IN2=10, IN4=11
// Note: sequence is IN1, IN3, IN2, IN4 for correct direction
Stepper myStepper(stepsPerRevolution, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);
  myStepper.setSpeed(10);  // 10 RPM
  Serial.println("Stepper ready on pins 8,9,10,11");
}

void loop() {
  if (Serial.available() > 0) {
    int steps = Serial.parseInt();
    
    if (steps != 0) {
      Serial.print("Moving ");
      Serial.print(steps);
      Serial.println(" steps");
      
      myStepper.step(steps);
      
      Serial.println("Done");
    }
  }
}
