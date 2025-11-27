#include <Servo.h>

Servo servo_0;
Servo servo_1;

int pos = 0; 

void setup() {
  servo_0.attach(9);
  servo_1.attach(10);
}

void loop() {
  // Sweep back and forth
  for (pos = 0; pos <= 180; pos += 1) { 
    servo_0.write(pos);
    servo_1.write(pos);
    delay(15);
  }
  delay(500);
  for (pos = 180; pos >= 0; pos -= 1) { 
    servo_0.write(pos);
    servo_1.write(pos);
    delay(15);
  }
  
  delay(1000);
}
 