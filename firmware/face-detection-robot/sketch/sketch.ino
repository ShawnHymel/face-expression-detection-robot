#include "Arduino_RouterBridge.h"
#include "Servo.h"

// Settings
#define DEBUG           1
#define PAN_SERVO_PIN   9
#define TILT_SERVO_PIN  10
#define SMOOTHING       0.2

// Servo objects
Servo pan_servo;
Servo tilt_servo;

// Default angles
float current_pan = 90.0;
float current_tilt = 90.0;
float target_pan = 90.0;
float target_tilt = 90.0;

// Debug macro
#if DEBUG
# define debug(fmt, ...) \
    do { \
        char buf[128]; \
        snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
        Serial.print(buf); \
    } while(0)
#else
# define debug(fmt, ...)
#endif

void animate_head(float pan, float tilt, int class_num, float score) {
    target_pan = pan;
    target_tilt = tilt;
    debug("Received: %.2f, %.2f (%i: %.2f)\r\n", pan, tilt, class_num, score);
}

void setup() {
    // Pour a bowl of Serial
#if DEBUG
    Serial.begin(115200);
#endif

    // Register App Bridge function
    Bridge.begin();
    Bridge.provide("animate_head", animate_head);
    
     // Initialize servos
    pan_servo.attach(PAN_SERVO_PIN);
    tilt_servo.attach(TILT_SERVO_PIN);
    pan_servo.write(90);
    tilt_servo.write(90);
}

void loop() {
    // This runs at 50+ FPS regardless of inference speed!
    current_pan += (target_pan - current_pan) * SMOOTHING;
    current_tilt += (target_tilt - current_tilt) * SMOOTHING;
    
    pan_servo.write((int)current_pan);
    tilt_servo.write((int)current_tilt);
    
    delay(20);
}