#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "board_config.h"
#include "wifi_secrets.h"

WiFiServer streamServer(80);
bool serverReady = false;
enum class PythonClientState {
  Disconnected,
  Connected,
  Streaming
};

PythonClientState pythonClientState = PythonClientState::Disconnected;
IPAddress pythonClientIP;

const char *wifiStatusName(wl_status_t status) {
  switch (status) {
    case WL_IDLE_STATUS:
      return "idle";
    case WL_NO_SSID_AVAIL:
      return "SSID not found";
    case WL_SCAN_COMPLETED:
      return "scan completed";
    case WL_CONNECTED:
      return "connected";
    case WL_CONNECT_FAILED:
      return "connection/authentication failed";
    case WL_CONNECTION_LOST:
      return "connection lost";
    case WL_DISCONNECTED:
      return "disconnected";
    default:
      return "unknown";
  }
}

void printConnectionStatus() {
  Serial.printf("[STATUS] Wi-Fi: %s",
                WiFi.status() == WL_CONNECTED ? "connected" : "disconnected");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(" (");
    Serial.print(WiFi.localIP());
    Serial.printf(", RSSI %d dBm)", WiFi.RSSI());
  }

  Serial.print(" | Python client: ");
  switch (pythonClientState) {
    case PythonClientState::Disconnected:
      Serial.println("disconnected");
      break;
    case PythonClientState::Connected:
      Serial.print("TCP connected, awaiting START from ");
      Serial.println(pythonClientIP);
      break;
    case PythonClientState::Streaming:
      Serial.print("streaming to ");
      Serial.println(pythonClientIP);
      break;
  }
}

void printConnectionStatusPeriodically() {
  static unsigned long lastStatusTime = 0;
  const unsigned long now = millis();
  if (now - lastStatusTime >= 5000) {
    lastStatusTime = now;
    printConnectionStatus();
  }
}

bool initCamera() {
  const bool hasPsram = psramFound();
  Serial.printf("PSRAM: %s\n", hasPsram ? "available" : "not detected");

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 30;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_sharpness(s, 2);
    s->set_contrast(s, 1);
  }
  Serial.printf("Camera sensor PID: 0x%04x, configured for QVGA streaming\n",
                s->id.PID);
  
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  // Initialize camera first to keep memory allocation predictable
  if (!initCamera()) {
    Serial.println("Camera init failed!");
    return;
  }

  // Connect to Wi-Fi
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  if (WIFI_SSID[0] == '\0' || WIFI_PASSWORD[0] == '\0') {
    Serial.println("Set the phone hotspot SSID and password in wifi_secrets.h.");
    return;
  }

  Serial.printf("Scanning for network '%s'...\n", WIFI_SSID);
  const int networkCount = WiFi.scanNetworks();
  bool networkFound = false;
  for (int i = 0; i < networkCount; i++) {
    if (WiFi.SSID(i) == WIFI_SSID) {
      networkFound = true;
      Serial.printf("Network found: RSSI %d dBm, channel %d\n",
                    WiFi.RSSI(i), WiFi.channel(i));
      break;
    }
  }
  WiFi.scanDelete();

  if (!networkFound) {
    Serial.println("Hotspot is not visible. Check that it is enabled and the SSID is exact.");
  }

  bool connected = false;
  for (int attempt = 1; attempt <= 2 && !connected; attempt++) {
    Serial.printf("Connecting to phone hotspot (attempt %d/2)", attempt);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    const unsigned long wifiTimeout = millis() + 20000;
    while (WiFi.status() != WL_CONNECTED && millis() < wifiTimeout) {
      delay(500);
      Serial.print(".");
    }

    connected = WiFi.status() == WL_CONNECTED;
    if (!connected) {
      Serial.printf("\nWi-Fi status: %s (%d)\n",
                    wifiStatusName(WiFi.status()), WiFi.status());
      WiFi.disconnect();
      delay(1000);
    }
  }

  if (!connected) {
    Serial.println("Could not join the hotspot. Verify its SSID/password and 2.4 GHz compatibility.");
    return;
  }
  Serial.println("\nWi-Fi connected!");
  Serial.printf("Assigned IP: %s, signal: %d dBm\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());

  // Flush stale frames
  for (int i = 0; i < 3; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }

  streamServer.begin();
  streamServer.setNoDelay(true);
  serverReady = true;

  Serial.println("=== SERVER READY ON PORT 80 ===");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  printConnectionStatus();
}

void loop() {
  if (!serverReady) {
    delay(1000);
    return;
  }
  printConnectionStatusPeriodically();

  WiFiClient client = streamServer.accept();
  if (!client) {
    delay(5);
    return;
  }

  client.setNoDelay(true);
  pythonClientIP = client.remoteIP();
  pythonClientState = PythonClientState::Connected;
  Serial.print("[CLIENT] TCP connection from ");
  Serial.println(pythonClientIP);
  printConnectionStatus();

  // 1.5s watchdog handshake to ignore port scanners
  unsigned long timeout = millis() + 1500;
  String handshake = "";
  while (client.connected() && millis() < timeout) {
    if (client.available()) {
      handshake = client.readStringUntil('\n');
      handshake.trim();
      break;
    }
  }

  if (handshake != "START") {
    Serial.printf("[CLIENT] Rejected: expected START handshake, received '%s'.\n",
                  handshake.c_str());
    client.stop();
    pythonClientState = PythonClientState::Disconnected;
    printConnectionStatus();
    return;
  }

  Serial.println("Python AI Client Authenticated & Streaming!");
  pythonClientState = PythonClientState::Streaming;
  printConnectionStatus();
  char command[128];
  size_t commandLength = 0;

  while (client.connected()) {
    printConnectionStatusPeriodically();
    while (client.available()) {
      const int incoming = client.read();
      if (incoming < 0) {
        break;
      }
      if (incoming == '\n') {
        command[commandLength] = '\0';
        if (commandLength > 0) {
          Serial.printf("[DRONE CMD]: %s\n", command);
        }
        commandLength = 0;
      } else if (incoming != '\r') {
        if (commandLength < sizeof(command) - 1) {
          command[commandLength++] = static_cast<char>(incoming);
        } else {
          commandLength = 0;
          Serial.println("Ignoring oversized client command.");
        }
      }
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed.");
      delay(5);
      continue;
    }

    uint32_t len = fb->len;
    uint8_t header[4] = {
      (uint8_t)((len >> 24) & 0xFF),
      (uint8_t)((len >> 16) & 0xFF),
      (uint8_t)((len >> 8) & 0xFF),
      (uint8_t)(len & 0xFF)
    };

    if (client.write(header, 4) != 4 || client.write(fb->buf, fb->len) != fb->len) {
      esp_camera_fb_return(fb);
      break;
    }

    esp_camera_fb_return(fb);
  }
  client.stop();
  pythonClientState = PythonClientState::Disconnected;
  Serial.println("Python Client Disconnected.");
  printConnectionStatus();
}