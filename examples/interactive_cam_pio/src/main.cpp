#include <WiFi.h>
#include <WebServer.h>
#include <math.h>
#include "esp_camera.h"
#include "img_converters.h"

extern "C" {
#include <layer.h>
#include <matrix.h>
}

// ── Network credentials ──────────────────────────────────────────────
const char *ssid     = "Airtel_Neelam";
const char *password = "Neelam@321";

WebServer server(80);

// ── AI-Thinker ESP32-CAM pinout ──────────────────────────────────────
static constexpr auto PWDN_GPIO_NUM  = 32;
static constexpr auto RESET_GPIO_NUM = -1;
static constexpr auto XCLK_GPIO_NUM  = 0;
static constexpr auto SIOD_GPIO_NUM  = 26;
static constexpr auto SIOC_GPIO_NUM  = 27;
static constexpr auto Y9_GPIO_NUM    = 35;
static constexpr auto Y8_GPIO_NUM    = 34;
static constexpr auto Y7_GPIO_NUM    = 39;
static constexpr auto Y6_GPIO_NUM    = 36;
static constexpr auto Y5_GPIO_NUM    = 21;
static constexpr auto Y4_GPIO_NUM    = 19;
static constexpr auto Y3_GPIO_NUM    = 18;
static constexpr auto Y2_GPIO_NUM    = 5;
static constexpr auto VSYNC_GPIO_NUM = 25;
static constexpr auto HREF_GPIO_NUM  = 23;
static constexpr auto PCLK_GPIO_NUM  = 22;
static constexpr auto FLASH_GPIO_NUM = 4;

// ── Resolution ───────────────────────────────────────────────────────
static constexpr auto CAM_FRAMESIZE = FRAMESIZE_VGA; // 640 × 480
static constexpr int  CAM_W = 640;
static constexpr int  CAM_H = 480;
static constexpr int  KERNEL_SIZE = 3;
static constexpr int  OUT_W = CAM_W - KERNEL_SIZE + 1;
static constexpr int  OUT_H = CAM_H - KERNEL_SIZE + 1;

// ── Vertical Sobel kernel (detects vertical edges) ───────────────────
static const float VERTICAL_KERNEL[KERNEL_SIZE * KERNEL_SIZE] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1,
};

// ── Library objects (pre-allocated once) ─────────────────────────────
static Layer  *g_conv   = nullptr;
static Matrix *g_input  = nullptr;
static Matrix *g_output = nullptr;

// ── Reusable working buffers (PSRAM) ─────────────────────────────────
static uint8_t *g_filtered = nullptr;  // CAM_W × CAM_H

// ── Camera config (static for & operator) ────────────────────────────
static camera_config_t s_camera_config;

static void init_camera_config() {
    camera_config_t c{};
    c.ledc_channel   = LEDC_CHANNEL_0;
    c.ledc_timer     = LEDC_TIMER_0;
    c.pin_d0         = Y2_GPIO_NUM;
    c.pin_d1         = Y3_GPIO_NUM;
    c.pin_d2         = Y4_GPIO_NUM;
    c.pin_d3         = Y5_GPIO_NUM;
    c.pin_d4         = Y6_GPIO_NUM;
    c.pin_d5         = Y7_GPIO_NUM;
    c.pin_d6         = Y8_GPIO_NUM;
    c.pin_d7         = Y9_GPIO_NUM;
    c.pin_xclk       = XCLK_GPIO_NUM;
    c.pin_pclk       = PCLK_GPIO_NUM;
    c.pin_vsync      = VSYNC_GPIO_NUM;
    c.pin_href       = HREF_GPIO_NUM;
    c.pin_sccb_sda   = SIOD_GPIO_NUM;
    c.pin_sccb_scl   = SIOC_GPIO_NUM;
    c.pin_pwdn       = PWDN_GPIO_NUM;
    c.pin_reset      = RESET_GPIO_NUM;
    c.xclk_freq_hz   = 20000000;
    c.pixel_format   = PIXFORMAT_GRAYSCALE;
    c.frame_size     = CAM_FRAMESIZE;
    c.jpeg_quality   = 12;
    c.fb_count       = 1;
    s_camera_config = c;
}

static void init_flash_light() {
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    digitalWrite(FLASH_GPIO_NUM, HIGH);
}

static uint8_t sobel_to_u8(float value) {
    float scaled = fabsf(value) * 0.25f;
    if (scaled > 255.0f) return 255;
    return (uint8_t)scaled;
}

static bool run_filter(camera_fb_t *fb) {
    if (!fb || !g_conv || !g_input || !g_output || !g_filtered) return false;
    if (fb->width != CAM_W || fb->height != CAM_H) return false;

    for (int i = 0; i < CAM_W * CAM_H; i++) {
        g_input->data[i] = (float)fb->buf[i];
    }

    if (layer_forward_conv2d_into(g_conv, g_input, g_output) != 0) {
        return false;
    }

    memset(g_filtered, 0, (size_t)CAM_W * (size_t)CAM_H);
    for (int y = 0; y < g_output->rows; y++) {
        uint8_t *dst = g_filtered + (y + 1) * CAM_W + 1;
        float *src = g_output->data + y * g_output->columns;
        for (int x = 0; x < g_output->columns; x++) {
            dst[x] = sobel_to_u8(src[x]);
        }
    }
    return true;
}

// ── HTTP handlers ────────────────────────────────────────────────────

void handleCapture() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { server.send(500, "text/plain", "Capture failed"); return; }
    server.sendHeader("Content-Type", "image/jpeg");
    server.sendHeader("Content-Length", String(fb->len));
    server.send(200);
    server.sendContent((const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
}

void handleStream() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { server.send(500, "text/plain", "Capture failed"); return; }
    
    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    if (!frame2jpg(fb, 80, &jpg_buf, &jpg_len)) {
        server.send(500, "text/plain", "JPEG encode failed");
        esp_camera_fb_return(fb);
        return;
    }
    
    server.setContentLength(jpg_len);
    server.send(200, "image/jpeg", "");
    server.sendContent((const char*)jpg_buf, jpg_len);
    
    free(jpg_buf);
    esp_camera_fb_return(fb);
}

void handleFilteredStream() {
    // Verify working buffers are available
    if (!g_filtered) {
        server.send(500, "text/plain", "Buffers not allocated");
        return;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Capture failed");
        return;
    }

    if (!run_filter(fb)) {
        esp_camera_fb_return(fb);
        server.send(500, "text/plain", "Filter failed");
        return;
    }
    esp_camera_fb_return(fb);

    uint8_t *jpg_buf = nullptr;
    size_t jpg_len = 0;
    if (fmt2jpg(g_filtered, (size_t)CAM_W * (size_t)CAM_H,
                CAM_W, CAM_H, PIXFORMAT_GRAYSCALE,
                85, &jpg_buf, &jpg_len)) {
        server.setContentLength(jpg_len);
        server.send(200, "image/jpeg", "");
        server.sendContent((const char*)jpg_buf, jpg_len);
        free(jpg_buf);
    } else {
        server.send(500, "text/plain", "JPEG encode failed");
    }
}

void handleRoot() {
    const char html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32-CAM + Conv2D Filter</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;background:#000;color:#fff;font-family:sans-serif}
    h2{margin:8px 10px;font-size:16px}
    img{display:block;width:100%;height:auto;background:#000}
  </style>
</head>
<body>
  <h2>Raw VGA</h2>
  <img id="raw" src="">
  <h2>Vertical Sobel VGA</h2>
  <img id="filtered" src="">
  <script>
    function startStreams() {
        const raw = document.getElementById('raw');
        const filtered = document.getElementById('filtered');

        function schedule(delay) {
            setTimeout(updateRaw, delay);
        }

        function updateRaw() {
            const stamp = Date.now();
            raw.onload = raw.onerror = () => updateFiltered(stamp);
            raw.src = '/stream?t=' + stamp;
        }

        function updateFiltered(stamp) {
            filtered.onload = () => schedule(120);
            filtered.onerror = () => schedule(1000);
            filtered.src = '/filtered?t=' + stamp;
        }

        updateRaw();
    }
    window.onload = () => {
        startStreams();
    }
  </script>
</body>
</html>
)rawliteral";
    server.send(200, "text/html", html);
}

// ── Helpers ──────────────────────────────────────────────────────────

static void *psram_alloc(size_t n) {
    void *p = ps_malloc(n);
    if (!p) p = malloc(n);
    return p;
}

static bool allocate_buffers() {
    g_filtered = (uint8_t *)psram_alloc((size_t)CAM_W * (size_t)CAM_H);

    bool ok = g_filtered;
    if (!ok) {
        free(g_filtered); g_filtered = nullptr;
    }
    return ok;
}

// ── Setup ────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1000);
    init_flash_light();

    init_camera_config();
    esp_err_t err = esp_camera_init(&s_camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return;
    }

    // ── Pre-allocate CNeuralNet objects ──
    g_conv = layer_create_conv2d_shape(CAM_H, CAM_W, KERNEL_SIZE, KERNEL_SIZE);
    if (!g_conv) { Serial.println("layer_create_conv2d_shape failed"); return; }
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++)
        g_conv->weights->data[i] = VERTICAL_KERNEL[i];
    g_conv->bias->data[0] = 0.0f;

    g_input = create_matrix(CAM_H, CAM_W);
    if (!g_input) { Serial.println("create input matrix failed"); return; }

    g_output = create_matrix(OUT_H, OUT_W);
    if (!g_output) { Serial.println("create output matrix failed"); return; }

    // ── Pre-allocate working buffers ──
    if (!allocate_buffers()) {
        Serial.println("Buffer allocation failed (PSRAM?)");
        return;
    }

    // ── WiFi ──
    WiFi.begin(ssid, password);
    Serial.print("Connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print('.');
    }
    Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());

    // ── HTTP routes ──
    server.on("/",         handleRoot);
    server.on("/favicon.ico", []() { server.send(204); });
    server.on("/capture",  handleCapture);
    server.on("/stream",   handleStream);
    server.on("/filtered", handleFilteredStream);
    server.begin();
    Serial.println("Server started");
}

void loop() {
    server.handleClient();
}
