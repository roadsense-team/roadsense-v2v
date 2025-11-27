#include <unity.h>
#include <Arduino.h>
#include <SPI.h>
#include <SD.h>

// Wiring Config (User Provided)
#define SD_CS_PIN    5
#define SD_SCK_PIN   18
#define SD_MISO_PIN  19
#define SD_MOSI_PIN  23

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

void test_sd_initialization() {
    // Initialize SPI manually to ensure pins are correct
    SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
    
    // Attempt to initialize SD card
    // NOTE: This WILL FAIL if no card is inserted!
    if (!SD.begin(SD_CS_PIN)) {
        TEST_FAIL_MESSAGE("SD Card initialization failed! (No card inserted? Check wiring/voltage)");
    }
    
    // If we get here, SPI is working and a card responded
    uint8_t cardType = SD.cardType();
    TEST_ASSERT_NOT_EQUAL(CARD_NONE, cardType);
    
    // Log card size for confirmation
    uint64_t cardSize = SD.cardSize() / (1024 * 1024);
    String msg = "Card Size: " + String((unsigned long)cardSize) + " MB";
    TEST_MESSAGE(msg.c_str());
}

void test_file_write_read() {
    // Create a test file
    File file = SD.open("/test_rw.txt", FILE_WRITE);
    TEST_ASSERT_TRUE_MESSAGE(file, "Failed to open file for writing");
    
    if (file) {
        const char* testText = "RoadSense V2V SD Test";
        file.print(testText);
        file.close();
        
        // Re-open for reading
        file = SD.open("/test_rw.txt", FILE_READ);
        TEST_ASSERT_TRUE_MESSAGE(file, "Failed to open file for reading");
        
        if (file) {
            String readText = file.readString();
            file.close();
            // Verify content match
            TEST_ASSERT_EQUAL_STRING(testText, readText.c_str());
            
            // Cleanup
            SD.remove("/test_rw.txt");
        }
    }
}

void setup() {
    // Wait for power stability
    // CRITICAL for Lolin D32 running module on 3.3V
    delay(2000);
    
    UNITY_BEGIN();
    
    RUN_TEST(test_sd_initialization);
    RUN_TEST(test_file_write_read);
    
    UNITY_END();
}

void loop() {
}
