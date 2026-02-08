# Forensik AI Estimasi PMI (Post-Mortem Interval)

Prototype Python untuk **estimasi PMI** berbasis:
- **gambar** temuan (proxy livor/rigor/tekstur/dekomposisi),
- **sensor lingkungan** (suhu + kelembapan),
- **input naratif** petugas/ahli.

Skrip juga membandingkan dengan estimasi **manual** (aturan klasik sederhana) untuk konteks casework:
- error absolut (MAE),
- konsistensi AI vs manual,
- waktu penyusunan laporan,
- rate AI lebih akurat dari manual.

## Struktur
- `forensik_pmi_ai.py` — pipeline utama pelatihan + evaluasi.
- `data/generate_sample_data.py` — generator data contoh (gambar sintetis + CSV).
- `data/sample_cases.csv` — dataset contoh (dibuat oleh generator).

## Menjalankan
```bash
python data/generate_sample_data.py
python forensik_pmi_ai.py --cases data/sample_cases.csv --out hasil_evaluasi_pmi.json
```

## Format CSV
Wajib ada kolom:
- `case_id`
- `image_path`
- `suhu_c`
- `kelembapan_pct`
- `rigor_score`
- `livor_score`
- `insect_activity`
- `narrative`

Opsional untuk training/evaluasi:
- `pmi_jam_aktual`

## Catatan penting forensik
Model ini adalah **prototype riset** dan **bukan pengganti** judgement ahli forensik. Aturan manual yang dipakai adalah simplifikasi untuk baseline komparasi.
