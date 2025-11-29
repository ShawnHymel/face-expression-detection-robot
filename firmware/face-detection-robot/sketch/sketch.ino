#include "Arduino_RouterBridge.h"
#include "Servo.h"

#include "ws2812b-bitbang.h"

// Settings
#define DEBUG           1       // 0 to disable serial messages, 1 to enable
#define PAN_SERVO_PIN   9       // Pan servo pin
#define TILT_SERVO_PIN  10      // Tilt servo pin
#define PAN_MIN         0       // Minimum servo angle
#define PAN_MAX         180     // Max servo angle
#define TILT_MIN        0       // Minimum servo angle
#define TILT_MAX        180     // Max servo angle
#define HORIZONTAL_FOV  60      // Degrees
#define VERTICAL_FOV    30      // Degrees
#define X_CENTER        0.5     // Center of x in frame
#define Y_CENTER        0.5     // Center of y in frame
#define X_DEAD_ZONE     0.08    // Area in frame (x) to not move camera
#define Y_DEAD_ZONE     0.08    // Area in frame (y) to not move camera
#define PAN_DIR         -1      // Pan is direct drive
#define TILT_DIR        1       // Tilt is geared
#define DELAY_MS        20      // Target ~50 updates/sec
#define NUM_LEDS        12

// PID settings
#define PAN_KP          0.15
#define PAN_KI          0.005
#define PAN_KD          0.005
#define PAN_I_MAX       10.0 
#define TILT_KP         0.5
#define TILT_KI         0.005
#define TILT_KD         0.005
#define TILT_I_MAX      10.0

// Derived constants
#define PAN_CENTER      (((PAN_MAX - PAN_MIN) / 2) + PAN_MIN)
#define TILT_CENTER     (((TILT_MAX - TILT_MIN) / 2) + TILT_MIN)

// Servo objects
Servo pan_servo;
Servo tilt_servo;

// Global angles
float current_pan = PAN_CENTER;
float current_tilt = TILT_CENTER;
float current_camera_angle_x = 0.0;  // Where camera is actually pointing (degrees from center)
volatile float target_camera_angle_x = 0.0;   // Where camera should point
float current_camera_angle_y = 0.0;  // Where camera is actually pointing (degrees from center)
volatile float target_camera_angle_y = 0.0;   // Where camera should point

// PID state variables
float pan_integral = 0.0;
float pan_last_error = 0.0;
float tilt_integral = 0.0;
float tilt_last_error = 0.0;

// LED framebuffer
uint8_t framebuffer[NUM_LEDS][4];

/*******************************************************************************
 * LED patterns
 */


 
/*******************************************************************************
 * Functions
 */

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

// Simple PID controller
float pid_update(float error, float *integral, float *last_error, 
                 float kp, float ki, float kd, float i_max, float dt) {
    // Proportional term
    float p_term = kp * error;
    
    // Integral term
    *integral += error * dt;
    if (*integral > i_max) *integral = i_max;
    if (*integral < -i_max) *integral = -i_max;
    float i_term = ki * (*integral);
    
    // Derivative term
    float derivative = (error - *last_error) / dt;
    float d_term = kd * derivative;
    
    *last_error = error;
    
#if DEBUG
    static int debug_counter = 0;
    if (++debug_counter >= 20) {
        debug("PID: P=%.2f, I=%.2f, D=%.2f, Out=%.2f\r\n", 
              p_term, i_term, d_term, p_term + i_term + d_term);
        debug_counter = 0;
    }
#endif
    
    return p_term + i_term + d_term;
}

void animate_head(float x_norm, float y_norm, int class_num, float score) {
    float face_angle_in_camera;
    float target_servo;

    // Print out coordinates received from object detection (on MPU)
    debug("Received: %.2f, %.2f (%i: %.2f)\r\n", x_norm, y_norm, class_num, score);

    // Calculate error from center
    float x_error = x_norm - X_CENTER;
    float y_error = y_norm - Y_CENTER;

    // Set new x target if outside dead zone
    if (fabs(x_error) > X_DEAD_ZONE) {
        // Face position in camera FOV (degrees from camera center)
        face_angle_in_camera = x_error * HORIZONTAL_FOV;

        // To center the face, camera needs to rotate to point at it
        target_camera_angle_x = 
            current_camera_angle_x + (PAN_DIR * face_angle_in_camera);

        // Clamp to servo limits (convert to servo angles)
        target_servo = PAN_CENTER + target_camera_angle_x;
        if (target_servo < PAN_MIN) {
            target_camera_angle_x = PAN_MIN - PAN_CENTER;
        } else if (target_servo > PAN_MAX) {
            target_camera_angle_x = PAN_MAX - PAN_CENTER;
        }
        
        debug("X error: %.3f, Face angle: %.2f deg, Target camera: %.2f deg\r\n", 
              x_error, face_angle_in_camera, target_camera_angle_x);
    }

    // Set new y target if outside dead zone
    if (fabs(y_error) > Y_DEAD_ZONE) {
        // Face position in camera FOV (degrees from camera center)
        face_angle_in_camera = y_error * VERTICAL_FOV;

        // To center the face, camera needs to rotate to point at it
        target_camera_angle_y = 
            current_camera_angle_y + (TILT_DIR * face_angle_in_camera);

        // Clamp to servo limits (convert to servo angles)
        target_servo = TILT_CENTER + target_camera_angle_y;
        if (target_servo < TILT_MIN) {
            target_camera_angle_y = TILT_MIN - TILT_CENTER;
        } else if (target_servo > TILT_MAX) {
            target_camera_angle_y = TILT_MAX - TILT_CENTER;
        }
        
        debug("Y error: %.3f, Face angle: %.2f deg, Target camera: %.2f deg\r\n", 
              y_error, face_angle_in_camera, target_camera_angle_y);
    }
}

/*******************************************************************************
 * Main
 */

void setup() {
    // Pour a bowl of Serial
#if DEBUG
    Serial.begin(115200);
#endif

    // Initialize LED ring
    ws2812b_init();

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
    float error_magnitude;
    float norm_error;
    float speed_scale;

    // Get the delay in seconds
    float dt = DELAY_MS / 1000.0;

    // Calculate error in camera angle (not servo angle)
    float camera_error_x = target_camera_angle_x - current_camera_angle_x;
    float camera_error_y = target_camera_angle_y - current_camera_angle_y;
    
    // Compute X PID output (degrees to move camera angle)
    float camera_delta_x = pid_update(
        camera_error_x, 
        &pan_integral, 
        &pan_last_error,
        PAN_KP, 
        PAN_KI, 
        PAN_KD, 
        PAN_I_MAX, 
        dt
    );

    // Compute Y PID output (degrees to move camera angle)
    float camera_delta_y = pid_update(
        camera_error_y, 
        &tilt_integral, 
        &tilt_last_error,
        TILT_KP, 
        TILT_KI, 
        TILT_KD, 
        TILT_I_MAX, 
        dt
    );

    // Velocity profile for X: set scale based on error
    error_magnitude = fabs(camera_error_x);
    if (error_magnitude > 10.0) {
        // Max scale at errors over 10
        speed_scale = 1.0;
    } else {
        // Exponential decay from 10 deg to 0 deg in error
        norm_error = error_magnitude / 10.0;
        speed_scale = 0.1 + (0.9 * norm_error * norm_error);
    }
    current_camera_angle_x += (speed_scale * camera_delta_x);

    // Velocity profile for Y: set scale based on error
    error_magnitude = fabs(camera_error_y);
    if (error_magnitude > 10.0) {
        // Max scale at errors over 10
        speed_scale = 1.0;
    } else {
        // Exponential decay from 10 deg to 0 deg in error
        norm_error = error_magnitude / 10.0;
        speed_scale = 0.1 + (0.9 * norm_error * norm_error);
    }
    current_camera_angle_y += (speed_scale * camera_delta_y);
    
    // Convert camera angle to servo position
    current_pan = PAN_CENTER + current_camera_angle_x;
    current_tilt = TILT_CENTER + current_camera_angle_y;
    
    // Constrain servos
    current_pan = min(max(current_pan, (float)PAN_MIN), (float)PAN_MAX);
    current_tilt = min(max(current_tilt, (float)TILT_MIN), (float)TILT_MAX);
    
    // Update camera angle to stay in sync
    current_camera_angle_x = current_pan - PAN_CENTER;
    current_camera_angle_y = current_tilt - TILT_CENTER;

    // Move servos
    pan_servo.write((int)current_pan);
    tilt_servo.write((int)current_tilt);
    
    // Wait before next update
    delay(DELAY_MS);
}