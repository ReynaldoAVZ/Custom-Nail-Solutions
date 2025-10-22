#include <FastLED.h>
#include <Servo.h>

// --- Configuration from your working script ---
// LED setup
#define LED_PIN     6
#define NUM_LEDS    100
CRGB leds[NUM_LEDS];

// Servo setup
#define SERVO_PIN   9
Servo myServo;

// --- Setup Function (Combined) ---
void setup() {
  // Start Serial communication (from command-listener script)
  // This is required to talk to the Raspberry Pi
  Serial.begin(9600);

  // Initialize LED strip (from your script)
  FastLED.addLeds<WS2812, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(255); // Set max brightness

  // Initialize Servo (from your script)
  myServo.attach(SERVO_PIN);

  // --- Set a safe starting state ---
  myServo.write(80); // Start at the first angle your Python script uses
  fill_solid(leds, NUM_LEDS, CRGB::Black); // Start with LEDs off
  FastLED.show();

  // Send a confirmation message to the Pi's serial monitor for debugging
  Serial.println("Arduino is ready for commands.");
}

// --- Main Loop (from command-listener script) ---
// This loop does nothing until a command arrives from the Raspberry Pi.
// The automatic servo sweep has been removed.
void loop() {
  // Check if there is data available to read from the Pi
  if (Serial.available() > 0) {
    // Read the first character to determine the command type ('S' for Servo, 'L' for LED)
    char command = Serial.read();

    if (command == 'S') {
      // If the command is 'S', the next part is an integer for the angle
      int angle = Serial.parseInt();
      myServo.write(angle); // Move the servo to the specified angle
    } 
    else if (command == 'L') {
      // If the command is 'L', the next part is an integer for the pattern
      int pattern = Serial.parseInt();
      setLedPattern(pattern); // Call the new FastLED pattern function
    }
  }
  // If there's no data, the loop does nothing, holding the last position.
}

// --- Helper Function for LED Patterns (MODIFIED) ---
// This function now uses FastLED commands instead of digitalWrite.
void setLedPattern(int pattern) {
  switch (pattern) {
    case 0: // Pattern 0: All LEDs Off
      fill_solid(leds, NUM_LEDS, CRGB::Black);
      FastLED.show();
      break;
    case 1: // Pattern 1: All LEDs White
      fill_solid(leds, NUM_LEDS, CRGB::White);
      FastLED.show();
      break;
    case 2: // Pattern 2: Rainbow Mode
      // This will run a rainbow animation. Note that this is a "blocking"
      // function, meaning the Arduino will be busy until the rainbow is done.
      // This is perfect for a cool effect after the scan is complete.
      rainbowCycle(20); 
      break;
    default:
      // Unknown pattern, do nothing
      break;
  }
}

// --- Helper functions for the Rainbow Effect ---
void rainbowCycle(uint8_t wait) {
  uint16_t i, j;
  for(j=0; j<256; j++) { // 1 cycle of all colors on the wheel
    for(i=0; i<NUM_LEDS; i++) {
      leds[i] = Wheel(((i * 256 / NUM_LEDS) + j) & 255);
    }
    FastLED.show();
    delay(wait);
  }
}

uint32_t Wheel(byte WheelPos) {
  WheelPos = 255 - WheelPos;
  if(WheelPos < 85) {
    return CRGB(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if(WheelPos < 170) {
    WheelPos -= 85;
    return CRGB(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return CRGB(WheelPos * 3, 255 - WheelPos * 3, 0);
}