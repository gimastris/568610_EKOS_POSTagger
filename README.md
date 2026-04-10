# POS Tagger Bahasa Indonesia dengan HMM + Viterbi

Repositori ini berisi implementasi **Part-of-Speech (POS) Tagger Bahasa Indonesia** menggunakan **Hidden Markov Model (HMM)** dan **algoritma Viterbi

Eko Sutrisno - 25/568610/PPA/07153

MKA UGM - PBAL

Program membaca dataset bertipe TSV berbasis kalimat, melatih model HMM, mengevaluasi hasil pada test set, lalu menyediakan **mode interaktif** untuk:
- menguji kalimat pada data test,
- melihat detail prediksi token,
- mencoba kalimat bebas,
- dan menampilkan debug tabel Viterbi.

---

## 1. Isi File

Struktur minimum pekerjaan ini:

- `viterbi_pos.py` → script utama POS Tagger
- `Data_POS-Tag-ID.tsv` → dataset POS tagging Bahasa Indonesia
- `README.md` → panduan replikasi

---

## 2. Kebutuhan Sistem

Script ini **tidak memerlukan library eksternal**. Semua modul yang dipakai berasal dari **Python standard library**:
- `math`
- `re`
- `sys`
- `collections`

### Versi Python yang disarankan
- **Python 3.9 atau lebih baru**

Cek versi Python:

```bash
python3 --version
```

---

## 3. Instalasi / Persiapan

Karena tidak ada package tambahan, langkah persiapannya sangat sederhana.

### Opsi A — langsung jalankan
```bash
git clone https://github.com/gimastris/568610_EKOS_POSTagger.git
cd 568610_EKOS_POSTagger
```

### Opsi B — memakai virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4. Format Dataset

Dataset dibaca oleh fungsi `load_data(filepath)` dengan pola seperti berikut:

```text
<kalimat id=1>
Eko	NN
belajar	VB
ai	NN
</kalimat>
```

### Karakteristik format dataset
1. Setiap kalimat dibuka dengan tag seperti `<kalimat id=...>`
2. Setiap token berada pada satu baris
3. Format token adalah:
   - **kata / token**
   - diikuti **tag POS** pada kolom terakhir
4. Penutup kalimat adalah `</kalimat>`
5. Token multi-kata tetap didukung, misalnya:
   - `pesta olahraga	NN`

---

## 5. Cara Menjalankan

Jalankan script dengan memberikan path dataset sebagai argumen:

```bash
python3 viterbi_pos.py Data_POS-Tag-ID.tsv
```

Atau jika file ada di folder lain:

```bash
python3 viterbi_pos.py /path/ke/Data_POS-Tag-ID.tsv
```

---

## 6. Proses yang Dilakukan Script

Setelah dijalankan, program akan melakukan langkah berikut:

1. Membaca dataset TSV
2. Menghitung jumlah kalimat
3. Membagi data menjadi:
   - **80% train set**
   - **20% test set**
4. Melatih model HMM
5. Menghitung:
   - **token-level accuracy**
   - **precision per tag**
   - **recall per tag**
   - **F1-score per tag**
   - **accuracy per tag**
6. Masuk ke **mode interaktif**
