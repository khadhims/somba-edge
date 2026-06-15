# Somba Edge

Edge pipeline untuk inferensi AI, rekaman go2rtc, dan sinkronisasi event ke server via WebSocket.

## Arsitektur

- **go2rtc**: Stream proxy — dikonfigurasi langsung di `go2rtc/go2rtc.yaml`
- **worker**: Fetch daftar kamera dari backend, inferensi AI, sync WebSocket
- **SQLite**: Buffer lokal event (`PENDING` → `SENT` setelah ACK server)

## Setup

1. Buat **Site** di Control Plane dan salin `api_key` dari Site Settings.
2. Buat/edit `go2rtc/go2rtc.yaml` langsung dengan stream RTSP per kamera.
   - Gunakan penamaan `{camera_uuid}_main` / `{camera_uuid}_sub` agar selaras dengan worker dan Web UI (atau nama custom seperti `cam-01`).
3. Buat **Kamera** di master data:
   - `rtsp_url` — referensi RTSP (sama seperti di go2rtc.yaml)
   - `stream_url` — URL HLS `.m3u8` dari go2rtc (untuk Web UI), contoh: `http://<edge-ip>:1988/api/stream.m3u8?src={uuid}_main`
4. Letakkan file model YOLO (`.pt`) di folder **`models/`** — nama file tanpa ekstensi diisi di master data Activity (mis. `memasak.pt` → input `memasak`).
5. Set `.env` dan jalankan:

```env
SERVER_API_URL=http://your-backend:3000
SERVER_WS_URL=http://your-backend:3000
EDGE_API_KEY=<api_key dari Site Settings>
GO2RTC_URL=http://go2rtc:1988
MODELS_DIR=/app/models
```

```bash
docker compose up -d --build
```

### Docker build notes

- Default image uses **CPU-only PyTorch** (`requirements.txt`) for faster, smaller builds on mini PCs without GPU.
- If the mini PC has NVIDIA GPU + `nvidia-container-toolkit`, build with GPU PyTorch:

```bash
REQUIREMENTS_FILE=requirements-gpu.txt docker compose build --no-cache worker
docker compose up -d
```

## Alur Data

### Live view (Web UI)

Browser memuat `stream_url` langsung dari master data kamera — **tidak** melalui backend proxy.

### Events & Alerts

1. Inferensi AI baca `rtsp_url` kamera dari backend (`GET /edge/cameras`); fallback go2rtc `{uuid}_sub` jika `rtsp_url` kosong
2. Deteksi → snapshot/rekaman → S3 → SQLite → sync WebSocket ke server

### Startup worker

1. Connect WebSocket `/edge` dengan `EDGE_API_KEY`
2. `GET /edge/cameras` → daftar kamera untuk inferensi
3. go2rtc **tidak** diatur otomatis oleh worker — edit `go2rtc/go2rtc.yaml` manual, lalu `docker compose restart go2rtc`
