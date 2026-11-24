# Peta berkas pipeline

Mulai dari README, kemudian buka notebook pengendali. Notebook memanggil skrip; algoritme tidak disalin ke sel notebook.

| Berkas | Tanggung jawab | Masukan dan keluaran utama |
|---|---|---|
| `config.yaml` | Parameter penelitian yang berlaku | Periode, split, fitur, grid, seed, baseline dan evaluasi |
| `01_dataset_sumber/sensor_raw.csv` | Lokasi bawaan CSV yang disiapkan secara lokal | Dibaca praproses |
| `02_pengendali/02_pengendali_training.ipynb` | Antarmuka mulai/pantau melalui Jupyter Windows | Memanggil WSL setelah jalur disesuaikan, lalu menampilkan status |
| `02_pengendali/notebook_support.py` | Fungsi kecil untuk pemantauan dan pemanggilan WSL | Status/CSV/log; tidak melatih model sendiri |
| `02_pengendali/unpack_runtime_pc.py` | Memeriksa lalu mengekstrak paket Mac | ZIP dan folder baru; tidak memulai training |
| `03_praproses/02_prepare.py` | Penyaringan, agregasi, scaler, split, sekuens | Data sumber → prepared arrays, scaler, jumlah sampel |
| `04_training/04_train.py` | Model, fit, checkpoint, tuning, seleksi bersama dan final | Prepared arrays → trial/history/model/prediksi |
| `04_training/pc_training_state.py` | Identitas run, checksum, lock dan pembekuan seleksi | Menolak artefak tertukar atau resume dengan sumber berbeda |
| `04_training/job_control.py` | Menjalankan satu fase sebagai proses terpisah | Fase smoke/benchmark/tune/final → log/status job |
| `04_training/42_run_pipeline.py` | Mengurutkan fase sampai final lengkap | Praproses → smoke → benchmark → tune → final; status pipeline |
| `05_baseline/03_baselines.py` | Naive, Seasonal Naive, MA dan ARIMA | Prepared series → prediksi baseline; pemilihan hanya train/validation |
| `06_evaluasi/05_evaluate.py` | Pasangan target, metrik tiap seed dan DM jika diizinkan | Prediksi → tabel metrik, cakupan dan status pengujian |
| `fungsi_bersama/common.py` | Operasi bersama yang konsisten | Config, fitur, transformasi, penulisan atomik dan metrik |
| `pemeriksaan_kode/test_*.py` | Pemeriksaan perilaku dan regresi | Data sintetis; bukan hasil penelitian |
| `package_manifest.json` dalam paket | Identitas kode/config salinan eksekusi | Dihitung ulang saat membangun paket baru, tanpa hash CSV mentah |
| `hasil_eksperimen/<nama_run>/` | Artefak satu eksperimen | Config beku, data, log, state, model, results |

## Isi hasil eksperimen

- `config/`: konfigurasi yang benar-benar digunakan.
- `data/`: data prepared, scaler dan sekuens.
- `logs/`: lingkungan, status, console, history dan setiap attempt fit.
- `state/`: identitas dan seleksi final yang dibekukan.
- `models/`: model final; cocokkan dengan manifest.
- `results/`: trial, seleksi, prediksi dan tabel evaluasi.

## Batas tanggung jawab

Praproses tidak memilih pemenang dari test. Training tidak memasang model produksi. Evaluasi tidak mengubah bobot atau memilih ulang seed dari test. Pemantauan tidak memulai fit. Penataan kode mengubah checksum sumber, sehingga kode revisi dipakai pada salinan eksekusi/run baru; jangan mengganti sumber atau identitas run lama.
