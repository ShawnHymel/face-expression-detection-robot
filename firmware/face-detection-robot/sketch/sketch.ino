#include "Arduino_RouterBridge.h"
#include "Servo.h"

// Settings
#define DEBUG           1       // 0 to disable serial messages, 1 to enable
#define PAN_SERVO_PIN   9       // Pan servo pin
#define TILT_SERVO_PIN  10      // Tilt servo pin
#define PAN_SERVO_MAX   180     // Max servo angle
#define TILT_SERVO_MAX  180     // Max servo angle
#define HORIZONTAL_VIEWING_ANGLE    60  // Degrees
#define VERTICAL_VIEWING_ANGLE      30  // Degrees
#define X_CENTER        0.5     // Center of x in frame
#define Y_CENTER        0.5     // Center of y in frame
#define X_DEAD_ZONE     0.05    // Area in frame (x) to not move camera
#define Y_DEAD_ZONE     0.05    // Area in frame (y) to not move camera
#define SMOOTHING       0.1     // Lower number = slower movement
#define PAN_DIR         -1       // Pan is direct drive
#define TILT_DIR        -1      // Tilt is geared
#define DELAY_MS        50      // Target ~20 updates/sec

// Servo objects
Servo pan_servo;
Servo tilt_servo;

// Global angles
float current_pan = (float)(PAN_SERVO_MAX / 2);
float current_tilt = (float)(TILT_SERVO_MAX / 2);
volatile float target_pan = (float)(PAN_SERVO_MAX / 2);
volatile float target_tilt = (float)(TILT_SERVO_MAX / 2);

// Debug macro
#if DEBUG
# define debug(fmt, ...) \
    do { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
        Serial.print(buf); \
    } while(0)
#else
# define debug(fmt, ...)
#endif

void animate_head(float x_norm, float y_norm, int class_num, float score) {
    // If bounding box X center is outside dead zone, set new pan target
    if ((x_norm < X_CENTER - X_DEAD_ZONE) || 
        (x_norm > X_CENTER + X_DEAD_ZONE)) {
        debug("%.2f\r\n", (x_norm * (float)HORIZONTAL_VIEWING_ANGLE));
        debug("%.2f\r\n", ((float)HORIZONTAL_VIEWING_ANGLE / 2));
        target_pan += PAN_DIR * ((x_norm * (float)HORIZONTAL_VIEWING_ANGLE) - 
            ((float)HORIZONTAL_VIEWING_ANGLE / 2));
    }

    // TODO: tilt

    debug("Received: %.2f, %.2f (%i: %.2f)\r\n", x_norm, y_norm, class_num, score);
    debug("New target: %.2f, %.2f\r\n", target_pan, target_tilt);
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
    pan_servo.write(current_pan);
    tilt_servo.write(current_tilt);
}

void loop() {
    current_pan += (target_pan - current_pan) * SMOOTHING;
    current_tilt += (target_tilt - current_tilt) * SMOOTHING;

    // debug("Moving: %.2f, %.2f\r\n", current_pan, current_tilt);
    
    pan_servo.write((int)current_pan);
    tilt_servo.write((int)current_tilt);
    
    delay(DELAY_MS);
}