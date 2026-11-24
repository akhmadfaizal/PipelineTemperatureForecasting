# PipelineTemperatureForecasting

Pipeline pelatihan dan evaluasi prediksi suhu ruang server dengan LSTM, GRU, serta pembanding Naive, Seasonal Naive, Moving Average, dan ARIMA. Repositori berisi kode aktif, data sumber, konfigurasi penelitian, notebook pengendali, dan tes. Model terlatih dan hasil eksperimen disimpan terpisah.

## Pilih versi

- **Modular.** Gunakan susunan folder repositori dan [panduan berkas](PANDUAN_FILE.md) untuk membaca atau mengembangkan kode per tahap.
- **Satu notebook.** Unduh [PipelineTemperatureForecasting_Standalone.ipynb](PipelineTemperatureForecasting_Standalone.ipynb) untuk menjalankan kode, konfigurasi, dan CSV yang tertanam tanpa menyalin folder repositori. Pengaturan awal menjalankan praproses. Pilih `TAHAP = "lengkap"` untuk pelatihan hingga evaluasi pada kernel Linux atau WSL dengan GPU yang terdeteksi TensorFlow.

## Mulai membaca

1. Baca [PANDUAN_FILE.md](PANDUAN_FILE.md) untuk fungsi setiap berkas.
2. Periksa `config.yaml`: dataset 42 hari, split 28/7/7, lookback 144 slot, horizon 12/24/36 slot.
3. Untuk memakai JupyterLab Windows, buka `02_pengendali/02_pengendali_training.ipynb` dan sesuaikan jalur klon, interpreter WSL, dan nama run. Jalur pada notebook adalah contoh.
4. Jalankan sel pemantauan untuk membaca status run.
5. Gunakan sel mulai hanya untuk **folder run baru**. Jangan menjalankannya untuk mengulang run selesai.

Config aktif untuk run baru memakai normalisasi **train-only**, disetujui Afi pada 13 Oktober 2026. Parameter suhu/kelembapan dihitung hanya dari pengamatan sah pada split train, sebelum interpolasi. Parameter diterapkan pada validation/test tanpa fit ulang dan tanpa clipping. Run historis tetap dicatat memakai global scaling; persetujuan pengguna ini tidak dinyatakan sebagai persetujuan pembimbing.

## Urutan proses

Praproses → smoke GPU → benchmark → tuning → training final.

Tuning mencakup 48 konfigurasi × 2 arsitektur × 3 horizon = 288 fit dengan seed 1. Training final menghasilkan 2 arsitektur × 3 horizon × 3 seed = 18 model. Benchmark berhenti untuk ditinjau bila estimasi melebihi 12 jam. Controller tetap berjalan saat browser ditutup selama PC/WSL tetap menyala.

Baseline dan evaluasi adalah tahap terpisah. Training selesai tidak otomatis berarti evaluasi, uji portabilitas atau deployment selesai. Seed tidak dipilih dari skor test.

## Susunan folder

| Folder | Isi |
|---|---|
| `01_dataset_sumber` | Data sumber dan provenance |
| `02_pengendali` | Notebook, pemantauan WSL dan ekstraksi paket |
| `03_praproses` | Penyaringan sampai sekuens |
| `04_training` | Fit, checkpoint, seleksi dan pengendalian proses |
| `05_baseline` | Model pembanding |
| `06_evaluasi` | Metrik, pasangan data dan pengujian statistik |
| `fungsi_bersama` | Fungsi lintas tahap |
| `pemeriksaan_kode` | Tes perilaku/regresi |
| `hasil_eksperimen` | Artefak terpisah untuk setiap run |

## Persiapan

Gunakan Python 3.12 dan instal dependensi dengan `python -m pip install -r requirements.txt`. Konfigurasi pelatihan memerlukan TensorFlow yang dapat memakai GPU pada lingkungan WSL/Linux. Pemeriksaan kode di bawah dapat dijalankan tanpa dataset atau GPU.

Praproses dan pelatihan membaca [`01_dataset_sumber/sensor_raw.csv`](01_dataset_sumber/sensor_raw.csv) secara bawaan. CSV sumber berisi 76.979 pembacaan pada 4 Agustus–6 Oktober 2026. Praproses memilih jendela penelitian 20 Agustus–30 September 2026 (WIB) yang memuat 67.299 pembacaan sebelum penyaringan. Kolom masukan adalah `waktu_wib`, `temperature`, dan `humidity`. Kolom `linkquality` mencatat kualitas sinyal dan tidak menjadi masukan model. Controller training memvalidasi periode dan pembagian 42 hari pada konfigurasi.

## Perintah WSL

Jalankan dari direktori salinan eksekusi dengan interpreter environment training:

```bash
python 04_training/42_run_pipeline.py start --config config.yaml --run-dir hasil_eksperimen/training_baru
```

Pemantauan melalui notebook membaca `logs/pipeline_status.json`, `logs/pipeline_console.log` dan CSV hasil. Fase individual juga dapat dikendalikan melalui `04_training/job_control.py`; lihat bantuan `--help` sebelum menjalankan.

Setelah pelatihan final selesai, jalankan pembanding, lalu evaluasi, menggunakan run yang sama:

```bash
python 05_baseline/03_baselines.py --config config.yaml --run-dir hasil_eksperimen/training_baru
python 06_evaluasi/05_evaluate.py --config config.yaml --run-dir hasil_eksperimen/training_baru
```

Analisis peringatan historis menggunakan protokol tersendiri. Pengaturan `evaluation.warning_final_protocol` pada konfigurasi bawaan masih kosong.

## Identitas dan penggunaan ulang

Perubahan format/docstring pun mengubah checksum kode. Jangan menimpa sumber, config, `training_identity.json` atau manifest run historis untuk melewati pemeriksaan. Fit terputus diulang dari seed awal dengan attempt lama disimpan; resume tidak diklaim melanjutkan state RNG/early stopping secara identik.

Commit Git mengidentifikasi versi kode yang dipublikasikan. Paket ZIP eksekusi memakai manifest kode dan konfigurasi sendiri. Run 42 hari yang telah selesai tetap merujuk snapshot yang dibekukan saat eksekusi. Evaluasi ulang memakai model lama membutuhkan versi analisis terpisah dengan asal model yang jelas.

## Pemeriksaan

```bash
python -m unittest discover -s pemeriksaan_kode -v
```

Tes memakai data sintetis dan tidak menggantikan pemeriksaan hasil penelitian atau model final lintas perangkat.
