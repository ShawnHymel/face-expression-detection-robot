// ws2812b-bitbang.h

// gpioa, pin 12 gives us D4 on the UNO Q board See:
// https://github.com/arduino/ArduinoCore-zephyr/blob/main/variants/arduino_uno_q_stm32u585xx/arduino_uno_q_stm32u585xx.overlay
#define DATA_GPIO_NODE DT_NODELABEL(gpioa)
#define DATA_GPIO_PIN 12

// Get the base address and create register access
#define GPIO_BASE DT_REG_ADDR(DATA_GPIO_NODE)
#define GPIO_BSRR (*(volatile uint32_t*)(GPIO_BASE + 0x18))

// Fast pin write 
#define PIN_HIGH() GPIO_BSRR = (1 << DATA_GPIO_PIN)
#define PIN_LOW()  GPIO_BSRR = (1 << (DATA_GPIO_PIN + 16))

// Get device handle for proper initialization
const struct device *gpio_dev = DEVICE_DT_GET(DATA_GPIO_NODE);

// Naive delay in CPU cycles
inline void delay_cycles(uint32_t cycles) __attribute__((always_inline));
inline void delay_cycles(uint32_t cycles) {
  while(cycles--) {
    __asm__ __volatile__("nop");
  }
}

// Initialize pin
void ws2812b_init() {
  gpio_pin_configure(gpio_dev, DATA_GPIO_PIN, GPIO_OUTPUT_INACTIVE);
  gpio_pin_set_raw(gpio_dev, DATA_GPIO_PIN, 0);
}

// Bitbang WS2812b data from the buffer
void ws2812b_show(uint8_t (*buf)[4], int num_leds) {

  // Disable interrupts while we send out WS2812b data
  noInterrupts();

  // Loop through entire buffer
  for (int led = 0; led < num_leds; led++) {
    for (int channel = 0; channel < 4; channel++) {
      uint8_t data = buf[led][channel];
      
      for (int i = 7; i >= 0; i--) {
        if (data & (1 << i)) {
          PIN_HIGH();
          delay_cycles(30);
          PIN_LOW();
          delay_cycles(15);
        } else {
          PIN_HIGH();
          delay_cycles(12);
          PIN_LOW();
          delay_cycles(33);
        }
      }
    }
  }

  // Re-enable interrupts
  interrupts();

  // Make sure that we send the reset signal
  PIN_LOW();
  delayMicroseconds(60);
}
