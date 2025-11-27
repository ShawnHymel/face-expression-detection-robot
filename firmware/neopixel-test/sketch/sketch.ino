#include "ws2812b-bitbang.h"

#define NUM_LEDS 12

uint8_t framebuffer[NUM_LEDS][4];

void setup() {
  ws2812b_init();
}

void loop() {
  // Set solid green
  for (int i = 0; i < NUM_LEDS; i++) {
    framebuffer[i][0] = 0xFF; // Green
    framebuffer[i][1] = 0;    // Red
    framebuffer[i][2] = 0;    // Blue
    framebuffer[i][3] = 0;    // White
  }
  ws2812b_show(framebuffer, NUM_LEDS);
  delay(500);

  // Clear
  for (int i = 0; i < NUM_LEDS; i++) {
    framebuffer[i][0] = 0;
    framebuffer[i][1] = 0;
    framebuffer[i][2] = 0;
    framebuffer[i][3] = 0;
  }
  ws2812b_show(framebuffer, NUM_LEDS);
  delay(500);
}