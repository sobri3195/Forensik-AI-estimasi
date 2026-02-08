from pathlib import Path
import csv
import random

random.seed(42)

img_dir = Path("data/images")
img_dir.mkdir(parents=True, exist_ok=True)

phrases = [
    "terdapat livor menetap dengan bau busuk mulai menyengat",
    "rigor tampak kaku total pada ekstremitas",
    "ditemukan belatung dan serangga aktif pada area luka",
    "kulit kehijauan awal, bau ringan",
    "hipostasis menetap, dekomposisi mulai jelas",
]

rows = []
for i in range(1, 31):
    # Simulasi file foto sebagai byte acak (untuk demo ekstraksi fitur file-level).
    raw = bytes(random.randint(0, 255) for _ in range(random.randint(6000, 20000)))
    img_path = img_dir / f"case_{i:02d}.binimg"
    img_path.write_bytes(raw)

    suhu = round(random.uniform(22, 34), 1)
    hum = round(random.uniform(45, 90), 1)
    rigor = round(random.uniform(0, 3), 2)
    livor = round(random.uniform(0, 3), 2)
    insect = round(random.uniform(0, 5), 2)
    narrative = random.choice(phrases)

    noise = random.uniform(-1.5, 1.5)
    pmi = max(1.0, (37 - suhu) / 0.9 + rigor * 1.8 + livor * 1.3 + insect * 2.5 + noise)

    rows.append(
        {
            "case_id": f"C{i:03d}",
            "image_path": str(img_path),
            "suhu_c": suhu,
            "kelembapan_pct": hum,
            "rigor_score": rigor,
            "livor_score": livor,
            "insect_activity": insect,
            "narrative": narrative,
            "pmi_jam_aktual": round(pmi, 2),
        }
    )

with open("data/sample_cases.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Generated data/sample_cases.csv with", len(rows), "cases")
