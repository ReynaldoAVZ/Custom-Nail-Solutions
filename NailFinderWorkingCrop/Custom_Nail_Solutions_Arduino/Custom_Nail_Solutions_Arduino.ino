#include <FastLED.h>
#include <Servo.h>

// --- Configuration ---
// LED setup
#define LED_PIN 6
#define NUM_LEDS 100
CRGB leds[NUM_LEDS];

// Servo setup
#define SERVO_PIN 9
Servo myServo;

// --- State Variables (NEW) ---
// 1: White (Static), 2: Rainbow (Continuous)
int currentPattern = 2; // Default to continuous Rainbow
uint8_t gHue = 0; // Variable to track the current position in the rainbow

// --- Setup Function (Combined) ---
void setup() {
  // Start Serial communication
  Serial.begin(9600);

  // Initialize LED strip
  FastLED.addLeds<WS2812, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(255); // Set max brightness

  // Initialize Servo
  myServo.attach(SERVO_PIN);

  // --- Set a safe starting state ---
  myServo.write(80); // Start at the first angle your Python script uses
  Serial.println("Arduino is ready for commands.");

  // Initialize LEDs to black, the first call to loop() will start the rainbow
  fill_solid(leds, NUM_LEDS, CRGB::Black); 
  FastLED.show();
}

// --- Main Loop (Refactored for Non-Blocking State Management) ---
void loop() {
  // --- 1. Check for Serial Commands (Highest Priority) ---
  if (Serial.available() > 0) {
    // Read the first character to determine the command type ('S' for Servo, 'L' for LED)
    char command = Serial.read();

    if (command == 'S') {
      // If 'S', the next part is an integer for the angle
      int angle = Serial.parseInt();
      myServo.write(angle); // Move the servo to the specified angle

      // Clear any leftover characters in the buffer after reading the int
      while (Serial.available()) { Serial.read(); }
    }
    else if (command == 'L') {
      // If 'L', the next part is an integer for the pattern
      int pattern = Serial.parseInt();
      setLedPattern(pattern); // Change the active LED pattern

      // Clear any leftover characters in the buffer after reading the int
      while (Serial.available()) { Serial.read(); }
    }
  }

  // --- 2. Execute Current LED Pattern (Non-blocking default mode) ---
  // If no command was received, the Arduino executes the current pattern.
  if (currentPattern == 1) {
    // Pattern 1: Static White. No action needed here, as it was set in setLedPattern.
    // This ensures the LED state is held constant during the scan.
  } 
  else if (currentPattern == 2) {
    // Pattern 2: Continuous Rainbow.
    updateRainbow(); 
  }
 
  // A small delay is acceptable here to debounce the loop and prevent 
  // the Arduino from running too hot, but FastLED often handles timing internally.
  // delay(1);
}

// --- Helper Function for LED Patterns (MODIFIED to set state) ---
void setLedPattern(int pattern) {
  currentPattern = pattern; // Set the new state/pattern

  switch (pattern) {
    case 0: // Pattern 0: All LEDs Off
      fill_solid(leds, NUM_LEDS, CRGB::Black);
      FastLED.show();
      break;
    case 1: // Pattern 1: All LEDs White (used during scanning)
      fill_solid(leds, NUM_LEDS, CRGB::White);
      FastLED.show();
      break;
    case 2: 
      // Pattern 2: Rainbow Mode. We don't run the animation here, 
      // we simply switch the state. The main loop will handle the animation.
      break;
    default:
      // If an unknown pattern is requested, default back to the continuous rainbow
      currentPattern = 2;
      break;
  }
}

// --- Non-Blocking Rainbow Effect (NEW) ---
// This function must be called continuously in the main loop() 
// but is non-blocking (it doesn't use delay()).
void updateRainbow() {
  // FastLED utility that runs the code inside every 20 milliseconds
  EVERY_N_MILLISECONDS(20) {
  gHue++; // Advance the color position
  }

  // Draw the rainbow effect based on the current hue.
  // `fill_rainbow` is a much cleaner way to draw this effect than the old Wheel function.
  fill_rainbow(leds, NUM_LEDS, gHue, 7); // Start color, total LEDs, hue, delta hue
  FastLED.show();
}
