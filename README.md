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
VIOLATION_CLASSES=0,1,2
MIN_CONFIDENCE=0.5
VIOLATION_MIN_CONFIDENCE=0.5
VIOLATION_EPISODE_START_FRAMES=15
VIOLATION_EPISODE_START_SEC=0.5
VIOLATION_EPISODE_END_SEC=5
RECORDING_POST_BUFFER_SEC=90
PERSON_START_FRAMES=25
MIN_RECORDING_DURATION_SEC=180
```

```bash
docker compose up -d --build
```

### Threshold & debounce

| Variabel | Default | Keterangan |
|----------|---------|------------|
| `MIN_CONFIDENCE` | `0.5` | Confidence deteksi `person` (rekaman) |
| `VIOLATION_MIN_CONFIDENCE` | `0.5` | Confidence deteksi pelanggaran (`best.pt`) |
| `VIOLATION_EPISODE_START_FRAMES` | `15` | Frame berturut-turut untuk mulai episode alert |
| `VIOLATION_EPISODE_START_SEC` | `0.5` | Atau mulai episode setelah 0,5 detik deteksi |
| `VIOLATION_EPISODE_END_SEC` | `5` | Episode selesai setelah 5 detik tanpa deteksi |
| `PERSON_START_FRAMES` | `25` | Mulai rekaman setelah 25 frame `person` berturut-turut |
| `RECORDING_POST_BUFFER_SEC` | `90` | Stop rekaman setelah `person` hilang ≥ 90 detik |
| `MIN_RECORDING_DURATION_SEC` | `180` | Skip simpan event jika durasi < 3 menit |

## Alur Data

### Startup (Fase I)

1. Connect WebSocket `/edge` dengan `EDGE_API_KEY`
2. `GET /edge/cameras` → upsert SQLite `master_cameras`
3. Worker baca kamera dari SQLite dan spawn thread per kamera

### Inferensi

| `alert` | Model | Output |
|---------|-------|--------|
| `false` | `yolov5s.pt` | Deteksi `person` → rekaman (debounce 25 frame, stop setelah hilang 90s) |
| `true` | `yolov5s.pt` + `best.pt` (class `0,1,2`) | Rekaman + **satu alert per episode** pelanggaran |

**Rekaman (events):** mulai setelah `person` terdeteksi 25 frame berturut-turut; berhenti setelah `person` tidak terdeteksi selama `RECORDING_POST_BUFFER_SEC` (default 90s); event tidak disimpan jika durasi < `MIN_RECORDING_DURATION_SEC` (default 3 menit).

**Alert pelanggaran:** episode dimulai setelah 15 frame berturut-turut **atau** 0,5 detik deteksi; episode selesai setelah 5 detik tanpa deteksi; hanya satu alert dikirim per episode (snapshot dari deteksi confidence tertinggi).

### Sync ke server

- `recording_event` — antrean SQLite `recording_events`
- `detection_event` — antrean SQLite `alerts` (hanya kamera `alert=true`)
