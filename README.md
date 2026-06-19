# Somba Edge

Edge pipeline untuk inferensi AI, rekaman go2rtc, dan sinkronisasi event ke server via WebSocket.

## Arsitektur

- **go2rtc**: Stream proxy — dikonfigurasi langsung di `go2rtc/go2rtc.yaml`
- **worker**: Sync kamera dari backend → SQLite `master_cameras`, inferensi AI, sync WebSocket
- **SQLite**: `master_cameras` (config kamera) + antrean `recording_events` / `alerts` (`PENDING` → `SENT`)

## Setup

1. Buat **Site** di Control Plane dan salin `api_key` dari Site Settings.
2. Buat/edit `go2rtc/go2rtc.yaml` langsung dengan stream RTSP per kamera.
3. Buat **Kamera** di master data:
   - `rtsp_url` — referensi RTSP (sama seperti di go2rtc.yaml)
   - `stream_url` — URL HLS `.m3u8` dari go2rtc (untuk Web UI)
   - `activity` — label aktivitas (mis. `memasak`)
   - `alert` — jika `true`, worker menjalankan `best.pt` + `yolov5s.pt`; jika `false`, hanya `yolov5s.pt` (deteksi `person`)
4. Letakkan model di folder **`models/`**:
   - `yolov5s.pt` — deteksi person untuk trigger rekaman
   - `best.pt` — deteksi pelanggaran (hanya jika `alert=true`)
5. Set `.env` dan jalankan:

```env
SERVER_API_URL=http://your-backend:3000
SERVER_WS_URL=http://your-backend:3000
EDGE_API_KEY=<api_key dari Site Settings>
GO2RTC_URL=http://go2rtc:1988
MODELS_DIR=/app/models
PERSON_MODEL=yolov5s.pt
VIOLATION_MODEL=best.pt
```

```bash
docker compose up -d --build
```

## Alur Data

### Startup (Fase I)

1. Connect WebSocket `/edge` dengan `EDGE_API_KEY`
2. `GET /edge/cameras` → upsert SQLite `master_cameras`
3. Worker baca kamera dari SQLite dan spawn thread per kamera

### Inferensi

| `alert` | Model | Output |
|---------|-------|--------|
| `false` | `yolov5s.pt` | Deteksi `person` → rekaman |
| `true` | `yolov5s.pt` + `best.pt` | Rekaman + alert pelanggaran |

### Sync ke server

- `recording_event` — antrean SQLite `recording_events`
- `detection_event` — antrean SQLite `alerts` (hanya kamera `alert=true`)
