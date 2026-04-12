#include <WiFi.h>
#include <driver/i2s.h>

const char* ssid = ":D";
const char* password = "18511632005";
const char* host = "192.168.1.104";
const int port = 5000;

#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 32
#define I2S_PORT I2S_NUM_0

WiFiClient client;

void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 32000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,   // INMP441 requires 32-bit clock
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,   // Read BOTH to bypass ESP32 channel bugs
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  setupI2S();
}

void loop() {
  if (!client.connected()) {
    if (!client.connect(host, port)) {
      delay(1000);
      return;
    }
  }

  size_t bytes_read;
  int32_t i2s_data[1024]; // Raw 32-bit stereo buffer
  int16_t out_data[512];  // Clean 16-bit mono buffer

  i2s_read(I2S_PORT, &i2s_data, sizeof(i2s_data), &bytes_read, portMAX_DELAY);

  int samples_read = bytes_read / 4; // Total 32-bit samples (Left + Right)

  if (samples_read > 0) {
    int out_idx = 0;
    
    // Step by 2 to grab ONLY the Left channel (indices 0, 2, 4...)
    // Since L/R is tied to GND, the Left channel contains the actual audio.
    for (int i = 0; i < samples_read; i += 2) {
      
      // Shift 32-bit data down to 16-bit. 
      // >> 14 is the sweet spot for the INMP441 to prevent clipping.
      out_data[out_idx] = i2s_data[i] >> 14; 
      out_idx++;
    }
    
    // Send the clean, mono 16-bit data to the Pi
    client.write((uint8_t*)out_data, out_idx * 2);
  }
}
