/*
 * OFNP Firmware — ESP32 Controller
 * Open Facial Neuroprosthesis Project
 *
 * Phase: 2-3 (simulation → bench testing)
 *
 * ⚠  IMPORTANT SAFETY NOTICE:
 *    This firmware controls a device intended for skin-surface use.
 *    All stimulation parameters MUST be validated by a qualified
 *    biomedical engineer and licensed clinician before ANY use
 *    on human subjects.
 *
 *    Phase 0-2: STIM_ENABLED = false (stimulation output disabled)
 *    Phase 3+:  Only enable after bench electrical safety testing
 *
 * Functions:
 *   - EMG ADC acquisition (2 channels, 1kHz)
 *   - Safety watchdog (hardware timer)
 *   - BLE communication with desktop application
 *   - Emergency stop (any watchdog timeout → all outputs off)
 *   - Stimulation pulse generation (DISABLED until Phase 3)
 *   - Skin impedance measurement (100Hz test pulse)
 *   - Status LED
 *
 * Hardware:
 *   MCU:      ESP32-S3 (dual-core, 240MHz)
 *   ADC:      ADS1115 (16-bit, 860SPS) via I²C
 *   DAC/PWM:  For stimulation control (Phase 3+)
 *   LED:      RGB status indicator
 *   BLE:      ESP32 integrated
 */

#include <Arduino.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ArduinoJson.h>
#include <esp_task_wdt.h>

// ── Safety Configuration ──────────────────────────────────────────────────────

#define STIM_ENABLED          false    // Phase 0-2: ALWAYS false
#define WATCHDOG_TIMEOUT_S    3        // Hardware watchdog timeout
#define MAX_CURRENT_MA        5.0f     // Absolute maximum current limit
#define MAX_PULSE_WIDTH_US    300      // Maximum pulse width (µs)
#define MAX_FREQUENCY_HZ      50       // Maximum stimulation frequency
#define MAX_SESSION_S         3600     // 1 hour maximum session
#define IMPEDANCE_MIN_KOHM    0.5f     // Minimum electrode impedance
#define IMPEDANCE_MAX_KOHM    30.0f    // Maximum electrode impedance

// ── Pin Definitions ───────────────────────────────────────────────────────────

#define PIN_EMG_STIM_PWM      25       // Stimulation DAC/PWM (Phase 3+)
#define PIN_STIM_ENABLE       26       // Hardware stimulation enable (active LOW)
#define PIN_EMERGENCY_STOP    27       // Physical emergency stop button (active LOW)
#define PIN_LED_R             32
#define PIN_LED_G             33
#define PIN_LED_B             34
#define PIN_IMPEDANCE_TEST    35       // Test pulse output for impedance measurement
#define I2C_SDA               21
#define I2C_SCL               22

// ── BLE UUIDs ─────────────────────────────────────────────────────────────────

#define SERVICE_UUID          "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_EMG_UUID         "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHAR_STATUS_UUID      "a7b3a6d2-91c5-4f3e-b8d2-5e2a1b4c6d8e"
#define CHAR_CONTROL_UUID     "c2d3e4f5-a6b7-48c9-d0e1-f2a3b4c5d6e7"
#define CHAR_SAFETY_UUID      "d4e5f6a7-b8c9-40d1-e2f3-a4b5c6d7e8f9"

// ── State ─────────────────────────────────────────────────────────────────────

struct SafetyState {
    bool emergency_stop_active;
    bool watchdog_ok;
    bool impedance_ok;
    bool stimulation_enabled;
    float current_ma;
    float impedance_kohm[2];
    uint32_t session_start_ms;
    uint32_t last_stim_ms;
    uint32_t stim_count;
};

struct EMGState {
    int16_t ch0_raw;           // Right zygomaticus (sensing channel)
    int16_t ch1_raw;           // Right cheek (sensing channel)
    float   ch0_uv;
    float   ch1_uv;
    uint32_t sample_count;
};

struct StimState {
    bool     active;
    float    peak_intensity;   // 0.0 – 1.0
    uint16_t pulse_width_us;   // MUST remain 0 in Phase 0-2
    float    frequency_hz;
    uint32_t ramp_up_ms;
    uint32_t sustain_ms;
    uint32_t ramp_down_ms;
};

SafetyState safety = {
    .emergency_stop_active = false,
    .watchdog_ok           = true,
    .impedance_ok          = false,  // Until first measurement
    .stimulation_enabled   = STIM_ENABLED,
    .current_ma            = 0.0f,
    .impedance_kohm        = {0.0f, 0.0f},
    .session_start_ms      = 0,
    .last_stim_ms          = 0,
    .stim_count            = 0,
};

EMGState emg = {};
StimState stim = {.active = false, .pulse_width_us = 0};

BLEServer*         bleServer     = nullptr;
BLECharacteristic* charEMG       = nullptr;
BLECharacteristic* charStatus    = nullptr;
BLECharacteristic* charControl   = nullptr;
BLECharacteristic* charSafety    = nullptr;
bool               bleConnected  = false;

// ── Emergency Stop ────────────────────────────────────────────────────────────

void IRAM_ATTR emergencyStopISR() {
    /*
     * Called from hardware interrupt when:
     *   - Physical button pressed
     *   - Hardware watchdog fires
     *   - Current limit exceeded
     * MUST be safe to call from ISR context.
     */
    safety.emergency_stop_active = true;
    // Direct GPIO — no FreeRTOS calls in ISR
    digitalWrite(PIN_STIM_ENABLE, HIGH);  // Active LOW — disable immediately
}

void emergencyStopSafe(const char* reason) {
    safety.emergency_stop_active = true;
    safety.stimulation_enabled   = false;

    // Disable stimulation hardware
    digitalWrite(PIN_STIM_ENABLE, HIGH);

    // LED: solid red
    digitalWrite(PIN_LED_R, HIGH);
    digitalWrite(PIN_LED_G, LOW);
    digitalWrite(PIN_LED_B, LOW);

    Serial.printf("[SAFETY] EMERGENCY STOP: %s\n", reason);

    // Send BLE alert
    if (bleConnected && charSafety) {
        StaticJsonDocument<128> doc;
        doc["event"]  = "emergency_stop";
        doc["reason"] = reason;
        char buf[128];
        serializeJson(doc, buf);
        charSafety->setValue((uint8_t*)buf, strlen(buf));
        charSafety->notify();
    }
}

// ── Safety Checks ─────────────────────────────────────────────────────────────

bool runSafetyChecks() {
    /*
     * Called before EVERY stimulation pulse.
     * Must return true for stimulation to proceed.
     * Any failure → immediate abort.
     */
    if (safety.emergency_stop_active)     return false;
    if (!STIM_ENABLED)                    return false;  // Phase 0-2 gate

    // Session timeout
    uint32_t session_elapsed = (millis() - safety.session_start_ms) / 1000;
    if (session_elapsed > MAX_SESSION_S) {
        emergencyStopSafe("session_timeout");
        return false;
    }

    // Current limit check (hardware comparator must mirror this)
    if (safety.current_ma > MAX_CURRENT_MA) {
        emergencyStopSafe("current_limit_exceeded");
        return false;
    }

    // Impedance check
    for (int ch = 0; ch < 2; ch++) {
        if (safety.impedance_kohm[ch] > 0.01f) {  // Not yet measured = skip
            if (safety.impedance_kohm[ch] < IMPEDANCE_MIN_KOHM) {
                emergencyStopSafe("impedance_too_low_possible_short");
                return false;
            }
            if (safety.impedance_kohm[ch] > IMPEDANCE_MAX_KOHM) {
                // Poor contact — block but don't emergency stop
                Serial.printf("[SAFETY] Impedance ch%d = %.1f kΩ — blocking\n",
                              ch, safety.impedance_kohm[ch]);
                return false;
            }
        }
    }

    // Stimulation parameters
    if (stim.pulse_width_us > MAX_PULSE_WIDTH_US) {
        emergencyStopSafe("pulse_width_exceeded");
        return false;
    }
    if (stim.frequency_hz > MAX_FREQUENCY_HZ) {
        emergencyStopSafe("frequency_exceeded");
        return false;
    }

    return true;
}

// ── EMG Acquisition Task (Core 0) ─────────────────────────────────────────────

void taskEMGAcquisition(void* pvParameters) {
    /*
     * 1kHz EMG sampling on dedicated core.
     * Reads ADS1115 via I²C, sends data via BLE at 100Hz.
     */
    const TickType_t period = pdMS_TO_TICKS(1);  // 1ms = 1kHz

    while (true) {
        TickType_t start = xTaskGetTickCount();

        // Read ADC (ADS1115 I²C — simplified; use Adafruit ADS1X15 library in production)
        // int16_t raw0 = ads.readADC_SingleEnded(0);
        // int16_t raw1 = ads.readADC_SingleEnded(1);
        // emg.ch0_raw = raw0;
        // emg.ch1_raw = raw1;
        // emg.ch0_uv  = raw0 * 0.1875f;  // ADS1115 @ ±6.144V gain → 187.5µV/LSB (use ±256mV for EMG)
        // emg.ch1_uv  = raw1 * 0.1875f;

        // Simulation: increment counter for testing
        emg.sample_count++;

        // Send via BLE every 10ms (100Hz) to avoid BLE congestion
        if (emg.sample_count % 10 == 0 && bleConnected && charEMG) {
            StaticJsonDocument<128> doc;
            doc["t"]   = millis();
            doc["c0"]  = emg.ch0_uv;
            doc["c1"]  = emg.ch1_uv;
            doc["n"]   = emg.sample_count;
            char buf[128];
            serializeJson(doc, buf);
            charEMG->setValue((uint8_t*)buf, strlen(buf));
            charEMG->notify();
        }

        // Feed hardware watchdog
        esp_task_wdt_reset();

        vTaskDelayUntil(&start, period);
    }
}

// ── Status Task (Core 1) ──────────────────────────────────────────────────────

void taskStatus(void* pvParameters) {
    while (true) {
        // Status LED
        if (safety.emergency_stop_active) {
            // Solid red
            digitalWrite(PIN_LED_R, HIGH);
            digitalWrite(PIN_LED_G, LOW);
            digitalWrite(PIN_LED_B, LOW);
        } else if (!bleConnected) {
            // Slow blue blink — waiting for connection
            digitalWrite(PIN_LED_B, !digitalRead(PIN_LED_B));
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        } else {
            // Green heartbeat — normal operation
            digitalWrite(PIN_LED_G, HIGH);
            vTaskDelay(pdMS_TO_TICKS(50));
            digitalWrite(PIN_LED_G, LOW);
        }

        // Send status via BLE every 1s
        if (bleConnected && charStatus) {
            StaticJsonDocument<256> doc;
            doc["uptime_s"]     = millis() / 1000;
            doc["stim_enabled"] = STIM_ENABLED;
            doc["estop"]        = safety.emergency_stop_active;
            doc["imp0"]         = safety.impedance_kohm[0];
            doc["imp1"]         = safety.impedance_kohm[1];
            doc["stim_count"]   = safety.stim_count;
            char buf[256];
            serializeJson(doc, buf);
            charStatus->setValue((uint8_t*)buf, strlen(buf));
            charStatus->notify();
        }

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// ── BLE Callbacks ─────────────────────────────────────────────────────────────

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        bleConnected = true;
        Serial.println("[BLE] Client connected");
        safety.session_start_ms = millis();
    }
    void onDisconnect(BLEServer* pServer) override {
        bleConnected = false;
        Serial.println("[BLE] Client disconnected — disabling stimulation");
        emergencyStopSafe("ble_disconnect");
        pServer->startAdvertising();
    }
};

class ControlCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pChar) override {
        /*
         * Receives control commands from desktop application:
         *   {"cmd": "stim", "intensity": 0.5, "pulse_us": 200, "freq": 30}
         *   {"cmd": "stop"}
         *   {"cmd": "emergency_stop"}
         *   {"cmd": "set_limits", "max_current_ma": 3.0}
         */
        std::string value = pChar->getValue();
        if (value.empty()) return;

        StaticJsonDocument<256> doc;
        DeserializationError err = deserializeJson(doc, value.c_str());
        if (err) {
            Serial.println("[BLE] JSON parse error");
            return;
        }

        const char* cmd = doc["cmd"];

        if (strcmp(cmd, "emergency_stop") == 0) {
            emergencyStopSafe("remote_command");

        } else if (strcmp(cmd, "stop") == 0) {
            stim.active = false;
            digitalWrite(PIN_STIM_ENABLE, HIGH);

        } else if (strcmp(cmd, "stim") == 0) {
            if (!runSafetyChecks()) {
                Serial.println("[SAFETY] Stim command blocked by safety checks");
                return;
            }
            // In Phase 3+: set PWM parameters here
            // stim.peak_intensity = doc["intensity"].as<float>();
            // stim.pulse_width_us = doc["pulse_us"].as<uint16_t>();
            // stim.frequency_hz   = doc["freq"].as<float>();
            // stim.active = true;
            Serial.println("[STIM] Command received but STIM_ENABLED=false (Phase 0-2)");

        } else if (strcmp(cmd, "set_limits") == 0) {
            if (doc.containsKey("max_current_ma")) {
                float v = doc["max_current_ma"].as<float>();
                if (v > 0 && v <= MAX_CURRENT_MA) {
                    safety.current_ma = v;
                }
            }
        }
    }
};

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    Serial.println("\n[OFNP] Firmware v0.1 — Phase 0-2 (Simulation)");
    Serial.printf("[OFNP] STIM_ENABLED = %s\n", STIM_ENABLED ? "true" : "false");

    // Safety pins — set outputs BEFORE anything else
    pinMode(PIN_STIM_ENABLE,    OUTPUT);
    digitalWrite(PIN_STIM_ENABLE, HIGH);  // Stimulator disabled (active LOW)
    pinMode(PIN_LED_R, OUTPUT);
    pinMode(PIN_LED_G, OUTPUT);
    pinMode(PIN_LED_B, OUTPUT);
    pinMode(PIN_EMERGENCY_STOP, INPUT_PULLUP);

    // Emergency stop interrupt
    attachInterrupt(digitalPinToInterrupt(PIN_EMERGENCY_STOP),
                    emergencyStopISR, FALLING);

    // Hardware watchdog: triggers emergency stop if main loop stalls
    esp_task_wdt_init(WATCHDOG_TIMEOUT_S, true);
    esp_task_wdt_add(NULL);

    // I²C
    Wire.begin(I2C_SDA, I2C_SCL);

    // BLE
    BLEDevice::init("OFNP-Controller");
    bleServer = BLEDevice::createServer();
    bleServer->setCallbacks(new ServerCallbacks());

    BLEService* svc = bleServer->createService(SERVICE_UUID);

    charEMG = svc->createCharacteristic(CHAR_EMG_UUID,
        BLECharacteristic::PROPERTY_NOTIFY);
    charEMG->addDescriptor(new BLE2902());

    charStatus = svc->createCharacteristic(CHAR_STATUS_UUID,
        BLECharacteristic::PROPERTY_NOTIFY);
    charStatus->addDescriptor(new BLE2902());

    charControl = svc->createCharacteristic(CHAR_CONTROL_UUID,
        BLECharacteristic::PROPERTY_WRITE);
    charControl->setCallbacks(new ControlCallbacks());

    charSafety = svc->createCharacteristic(CHAR_SAFETY_UUID,
        BLECharacteristic::PROPERTY_NOTIFY);
    charSafety->addDescriptor(new BLE2902());

    svc->start();
    bleServer->getAdvertising()->start();
    Serial.println("[BLE] Advertising as 'OFNP-Controller'");

    // Launch tasks on separate cores
    xTaskCreatePinnedToCore(taskEMGAcquisition, "EMG", 4096, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(taskStatus,         "STS", 4096, NULL, 1, NULL, 1);

    Serial.println("[OFNP] Ready");
}

void loop() {
    esp_task_wdt_reset();
    vTaskDelay(pdMS_TO_TICKS(100));
}
