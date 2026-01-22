/*
 * Arduino Servo Control
 * Upload this sketch to your Arduino Uno/Elegoo
 * Servo connected to pin 9
 * Serial commands: angle value (0-180)
 */

#include <Servo.h>

Servo myServo;
const int servoPin = 9;

void setup() {
  Serial.begin(9600);
  myServo.attach(servoPin);
  myServo.write(90);  // Start at center position
  Serial.println("Servo ready on pin 9");
}

void loop() {
  if (Serial.available() > 0) {
    int angle = Serial.parseInt();
    
    // Validate angle range
    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);
      Serial.print("Servo moved to: ");
      Serial.println(angle);
    }
  }
}
