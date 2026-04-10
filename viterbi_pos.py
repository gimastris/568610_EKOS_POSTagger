#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POS Tagger Bahasa Indonesia
Menggunakan Hidden Markov Model (HMM) + Algoritma Viterbi

Fitur utama:
1. load_data(filepath)
2. train_hmm(sentences)
3. viterbi(sentence, tags, pi, trans_prob, emit_prob)
4. evaluate(test_sentences, tags, pi, trans_prob, emit_prob)
5. predict_sentence(raw_sentence, tags, pi, trans_prob, emit_prob)

"""

import math
import re
import sys
from collections import Counter, defaultdict


# =========================================================
# KONSTANTA
# =========================================================
EMIT_SMOOTHING_K = 0.001
EPSILON = 1e-12
CASE_BACKOFF_WEIGHT = 0.35
DEFAULT_BATCH_SIZE = 10

# Global state model
MODEL_VOCAB = set()                 # vocab exact form
MODEL_LOWER_VOCAB = set()           # vocab lowercase form
DEFAULT_EMIT_PROB = {}              # smoothing known exact word tapi belum pernah pada tag tsb
DEFAULT_LOWER_EMIT_PROB = {}        # smoothing known lower-form group tapi belum pernah pada tag tsb
LOWER_EMIT_PROB = {}                # P(lower_form | tag) hasil agregasi semua varian kapitalisasi
TAG_COUNTS = Counter()              # count tag training
WORD_TAG_DISTRIBUTION = defaultdict(Counter)       # distribusi exact word -> tag
LOWER_WORD_TAG_DISTRIBUTION = defaultdict(Counter) # distribusi lower-form -> tag


POS_TAG_NAMES = {
    "CC": "Coordinating Conjunction",
    "CD": "Cardinal Number",
    "DT": "Determiner",
    "FW": "Foreign Word",
    "IN": "Preposition / Subordinating",
    "JJ": "Adjective",
    "MD": "Modal",
    "NEG": "Negation",
    "NN": "Noun",
    "NND": "Derived Noun",
    "NNP": "Proper Noun",
    "OD": "Ordinal Number",
    "PR": "Particle / Pronoun Marker",
    "PRP": "Personal Pronoun",
    "RB": "Adverb",
    "RP": "Particle",
    "SC": "Subordinating Conjunction",
    "SYM": "Symbol",
    "UH": "Interjection",
    "VB": "Verb",
    "WH": "Wh-Word",
    "X": "Unknown / Other",
    "Z": "Punctuation",
}


# =========================================================
# FUNGSI BANTUAN
# =========================================================
def safe_log(value):
    """
    Menghitung log secara aman.
    Jika value <= 0, gunakan EPSILON agar tidak error log(0).
    """
    if value <= 0:
        return math.log(EPSILON)
    return math.log(value)


def normalize_word(word):
    """
    Normalisasi ringan pada word TANPA lowercasing global.
    Hanya membersihkan spasi berlebih agar token multi-kata tetap aman.
    """
    return " ".join(word.strip().split())


def normalize_tag(tag):
    """
    Menyamakan tag jika ada variasi penulisan.
    Pada dataset ini ada 'fw' dan 'FW', kita satukan menjadi 'FW'.
    """
    return tag.strip().upper()


def simple_tokenize(text):
    """
    Tokenizer sederhana untuk input kalimat baru.
    Memisahkan kata dan tanda baca.
    """
    return re.findall(r"\w+(?:-\w+)*|[^\w\s]", text, flags=re.UNICODE)


def normalize_distribution(distribution):
    """
    Menormalkan dictionary probabilitas agar totalnya menjadi 1.
    """
    total = sum(distribution.values())

    if total <= 0:
        item_count = len(distribution)
        if item_count == 0:
            return {}
        return {key: 1.0 / item_count for key in distribution}

    return {key: value / total for key, value in distribution.items()}


def word_status(word):
    """
    Menentukan status vocabulary sebuah token:
    - Known
    - OOV exact  : bentuk persis tidak ada, tetapi lowercase group dikenal
    - OOV total  : benar-benar tidak ada di training vocab
    """
    word = normalize_word(word)
    lower_word = word.lower()

    if word in MODEL_VOCAB:
        return "Known"

    if lower_word in MODEL_LOWER_VOCAB:
        return "OOV exact"

    return "OOV total"


def build_notes(word, position_in_sentence=None):
    """
    Membuat catatan singkat untuk analisis token.
    """
    notes = []
    norm_word = normalize_word(word)
    lower_word = norm_word.lower()

    # status OOV
    status = word_status(norm_word)
    if status != "Known":
        notes.append(status)

    # ambigu
    if len(LOWER_WORD_TAG_DISTRIBUTION.get(lower_word, {})) > 1:
        notes.append("ambigu")

    # kapitalisasi
    if norm_word.isupper() and len(norm_word) > 1:
        notes.append("upper")
    elif norm_word.istitle():
        if position_in_sentence is not None and position_in_sentence > 0:
            notes.append("title-mid")
        else:
            notes.append("title")

    # bentuk khusus
    if " " in norm_word:
        notes.append("multiword")
    if "-" in norm_word:
        notes.append("hyphen")
    if any(ch.isdigit() for ch in norm_word):
        notes.append("digit")

    if not notes:
        return "-"

    return ", ".join(notes)


def sep_line(widths, char="="):
    return "+" + "+".join(char * (w + 2) for w in widths) + "+"


def wrap_cell(text, width):
    text = "" if text is None else str(text)
    if text == "":
        return [""]
    if len(text) <= width:
        return [text]

    words = text.split()
    if not words:
        return [text[:width]]

    lines = []
    current = ""

    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            for i in range(0, len(word), width):
                lines.append(word[i:i + width])
            continue

        trial = word if current == "" else current + " " + word
        if len(trial) <= width:
            current = trial
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def print_table(headers, rows, widths=None):
    """
    Mencetak tabel rapi TANPA garis pemisah antar baris.
    """
    if widths is None:
        widths = []
        for col_idx, header in enumerate(headers):
            max_len = len(str(header))
            for row in rows:
                max_len = max(max_len, len(str(row[col_idx])))
            widths.append(max_len)

    print(sep_line(widths, "="))

    # header
    print("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(sep_line(widths, "="))

    # isi
    for row in rows:
        wrapped_cols = [wrap_cell(row[i], widths[i]) for i in range(len(headers))]
        max_lines = max(len(lines) for lines in wrapped_cols)

        for line_idx in range(max_lines):
            out = []
            for col_idx in range(len(headers)):
                col_lines = wrapped_cols[col_idx]
                value = col_lines[line_idx] if line_idx < len(col_lines) else ""
                out.append(value.ljust(widths[col_idx]))
            print("| " + " | ".join(out) + " |")

    print(sep_line(widths, "="))


# =========================================================
# 1) LOAD DATA
# =========================================================
def load_data(filepath):
    """
    Membaca file TSV dan mengembalikan list of sentences.

    Bentuk output:
    [
        [("token_1", "POS_1"), ("token_2", "POS_2")],
        ...
    ]

    """
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "":
                continue

            if line.startswith("<kalimat"):
                current_sentence = []
                continue

            if line.startswith("</kalimat"):
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                word = normalize_word(" ".join(parts[:-1]))
                tag = normalize_tag(parts[-1])
                current_sentence.append((word, tag))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


# =========================================================
# UNKNOWN WORD HANDLING
# =========================================================
def guess_unknown_distribution(word, tags):
    """
    Menebak distribusi tag untuk kata yang benar-benar tidak ada
    di vocabulary training.

    Heuristic ini tetap sederhana, tetapi memperhatikan:
    - tanda baca
    - angka
    - kapitalisasi
    - multiword
    - beberapa pola afiks bahasa Indonesia
    - beberapa function words umum
    """
    probs = {tag: 1.0 for tag in tags}

    word = normalize_word(word)
    lower_word = word.lower()

    # Tanda baca murni
    if re.fullmatch(r"[.,;:!?()\[\]\"'`\-]+", word):
        for tag in probs:
            probs[tag] = 1e-6
        if "Z" in probs:
            probs["Z"] = 1.0
        return normalize_distribution(probs)

    # Angka / numerik sederhana
    if re.fullmatch(r"\d+([.,]\d+)?", word):
        for tag in probs:
            probs[tag] = 1e-6
        if "CD" in probs:
            probs["CD"] = 1.0
        return normalize_distribution(probs)

    # Singkatan uppercase seperti BISA, BI, PT, DPR
    if word.isupper() and len(word) >= 2:
        if "NNP" in probs:
            probs["NNP"] += 6.0
        if "NN" in probs:
            probs["NN"] += 1.0

    # Title Case condong ke proper noun
    if len(word) > 0 and word[0].isupper() and not word.isupper():
        if "NNP" in probs:
            probs["NNP"] += 3.5

    # Multi-kata yang tidak dikenal sering kali frasa nominal / nama entitas
    if " " in word:
        if "NNP" in probs:
            probs["NNP"] += 2.5
        if "NN" in probs:
            probs["NN"] += 2.0

    # Heuristik awalan / akhiran bahasa Indonesia
    if lower_word.endswith("kan"):
        if "VB" in probs:
            probs["VB"] += 5.0

    if lower_word.endswith("i"):
        if "VB" in probs:
            probs["VB"] += 2.0

    if lower_word.endswith("an"):
        if "NN" in probs:
            probs["NN"] += 3.0
        if "NND" in probs:
            probs["NND"] += 2.0

    if lower_word.endswith("nya"):
        if "NN" in probs:
            probs["NN"] += 2.0
        if "PRP" in probs:
            probs["PRP"] += 2.0
        if "PR" in probs:
            probs["PR"] += 1.0

    for prefix in ("me", "ber", "di", "ter", "se", "ke", "pe", "per"):
        if lower_word.startswith(prefix):
            if "VB" in probs:
                probs["VB"] += 1.2
            break

    # Function words umum
    if lower_word in {"saya", "aku", "kami", "kita", "anda", "dia", "mereka", "ia"}:
        if "PRP" in probs:
            probs["PRP"] += 8.0

    if lower_word in {"ini", "itu", "tersebut"}:
        if "PR" in probs:
            probs["PR"] += 5.0

    if lower_word in {"di", "ke", "dari", "pada", "untuk", "dengan", "dalam", "oleh"}:
        if "IN" in probs:
            probs["IN"] += 6.0

    if lower_word in {"dan", "atau", "tetapi", "serta", "namun"}:
        if "CC" in probs:
            probs["CC"] += 6.0

    if lower_word in {"yang", "karena", "agar", "ketika", "walaupun", "meskipun", "bahwa"}:
        if "SC" in probs:
            probs["SC"] += 6.0

    if lower_word in {"tidak", "bukan", "jangan", "tak", "belum"}:
        if "NEG" in probs:
            probs["NEG"] += 6.0

    return normalize_distribution(probs)


def get_emission_prob(word, tag, emit_prob, tags):
    """
    Mengambil probabilitas emisi P(word | tag) dengan strategi backoff:

    1. Exact form
    2. Known exact word tapi belum pernah pada tag tsb -> smoothing default exact
    3. Lower-form group backoff
    4. Unknown heuristic
    """
    word = normalize_word(word)
    lower_word = word.lower()

    # 1. Exact form
    if tag in emit_prob and word in emit_prob[tag]:
        return emit_prob[tag][word]

    # 2. Exact word known, tapi untuk tag ini belum pernah muncul
    if word in MODEL_VOCAB:
        return DEFAULT_EMIT_PROB.get(tag, EPSILON)

    # 3. Lower-form group backoff
    if lower_word in MODEL_LOWER_VOCAB:
        if tag in LOWER_EMIT_PROB and lower_word in LOWER_EMIT_PROB[tag]:
            return CASE_BACKOFF_WEIGHT * LOWER_EMIT_PROB[tag][lower_word]
        return DEFAULT_LOWER_EMIT_PROB.get(tag, EPSILON)

    # 4. Unknown word heuristic
    unknown_distribution = guess_unknown_distribution(word, tags)
    return unknown_distribution.get(tag, EPSILON)


# =========================================================
# 2) TRAIN HMM
# =========================================================
def train_hmm(sentences):
    """
    Mengestimasi parameter HMM:
    - pi           : initial probability
    - trans_prob   : transition probability
    - emit_prob    : emission probability exact form
    - vocab        : vocabulary exact form
    """
    global MODEL_VOCAB
    global MODEL_LOWER_VOCAB
    global DEFAULT_EMIT_PROB
    global DEFAULT_LOWER_EMIT_PROB
    global LOWER_EMIT_PROB
    global TAG_COUNTS
    global WORD_TAG_DISTRIBUTION
    global LOWER_WORD_TAG_DISTRIBUTION

    initial_tag_counts = Counter()
    tag_counts = Counter()

    transition_counts = defaultdict(Counter)
    outgoing_transition_counts = Counter()

    emission_counts = defaultdict(Counter)
    lower_emission_counts = defaultdict(Counter)

    vocab = set()
    lower_vocab = set()

    WORD_TAG_DISTRIBUTION = defaultdict(Counter)
    LOWER_WORD_TAG_DISTRIBUTION = defaultdict(Counter)

    for sentence in sentences:
        if not sentence:
            continue

        first_word, first_tag = sentence[0]
        initial_tag_counts[first_tag] += 1

        previous_tag = None

        for word, tag in sentence:
            word = normalize_word(word)
            lower_word = word.lower()

            vocab.add(word)
            lower_vocab.add(lower_word)

            tag_counts[tag] += 1
            emission_counts[tag][word] += 1
            lower_emission_counts[tag][lower_word] += 1

            WORD_TAG_DISTRIBUTION[word][tag] += 1
            LOWER_WORD_TAG_DISTRIBUTION[lower_word][tag] += 1

            if previous_tag is not None:
                transition_counts[previous_tag][tag] += 1
                outgoing_transition_counts[previous_tag] += 1

            previous_tag = tag

    tags = sorted(tag_counts.keys())

    num_tags = len(tags)
    vocab_size = len(vocab)
    lower_vocab_size = len(lower_vocab)
    total_sentences = len(sentences)

    MODEL_VOCAB = vocab
    MODEL_LOWER_VOCAB = lower_vocab
    DEFAULT_EMIT_PROB = {}
    DEFAULT_LOWER_EMIT_PROB = {}
    LOWER_EMIT_PROB = {}
    TAG_COUNTS = tag_counts.copy()

    # Initial probability dengan add-one smoothing
    pi = {}
    for tag in tags:
        pi[tag] = (initial_tag_counts[tag] + 1) / (total_sentences + num_tags)

    # Transition probability dengan add-one smoothing
    trans_prob = {}
    for prev_tag in tags:
        trans_prob[prev_tag] = {}
        denominator = outgoing_transition_counts[prev_tag] + num_tags
        for curr_tag in tags:
            trans_prob[prev_tag][curr_tag] = (transition_counts[prev_tag][curr_tag] + 1) / denominator

    # Emission probability exact form
    emit_prob = {}
    for tag in tags:
        emit_prob[tag] = {}
        exact_denominator = tag_counts[tag] + (EMIT_SMOOTHING_K * vocab_size)
        DEFAULT_EMIT_PROB[tag] = EMIT_SMOOTHING_K / exact_denominator

        for word in emission_counts[tag]:
            emit_prob[tag][word] = (emission_counts[tag][word] + EMIT_SMOOTHING_K) / exact_denominator

    # Lower-form backoff probability
    for tag in tags:
        LOWER_EMIT_PROB[tag] = {}
        lower_denominator = tag_counts[tag] + (EMIT_SMOOTHING_K * lower_vocab_size)
        DEFAULT_LOWER_EMIT_PROB[tag] = (EMIT_SMOOTHING_K / lower_denominator) * CASE_BACKOFF_WEIGHT

        for lower_word in lower_emission_counts[tag]:
            LOWER_EMIT_PROB[tag][lower_word] = (lower_emission_counts[tag][lower_word] + EMIT_SMOOTHING_K) / lower_denominator

    return tags, pi, trans_prob, emit_prob, vocab


# =========================================================
# FUNGSI TAMBAHAN UNTUK MELIHAT PROSES VITERBI
# =========================================================
def print_viterbi_table(delta, sentence, tags):
    """
    Menampilkan tabel skor Viterbi secara sederhana.
    """
    print("\nTABEL VITERBI (log probability)")
    print("=" * 100)

    header = "Tag".ljust(8)
    for word in sentence:
        header += word[:12].ljust(14)
    print(header)
    print("-" * 100)

    for tag in tags:
        row = tag.ljust(8)
        for i in range(len(sentence)):
            row += f"{delta[i].get(tag, float('-inf')):.4f}".ljust(14)
        print(row)

    print("=" * 100)


# =========================================================
# 3) VITERBI
# =========================================================
def viterbi(sentence, tags, pi, trans_prob, emit_prob):
    """
    Menjalankan algoritma Viterbi pada satu kalimat.
    Input:
        sentence = list kata
    Output:
        list tag prediksi
    """
    if len(sentence) == 0:
        return []

    sentence = [normalize_word(word) for word in sentence]
    n = len(sentence)

    delta = [{} for _ in range(n)]
    psi = [{} for _ in range(n)]

    # 1. Inisialisasi
    first_word = sentence[0]
    for tag in tags:
        initial_log = safe_log(pi.get(tag, EPSILON))
        emission_log = safe_log(get_emission_prob(first_word, tag, emit_prob, tags))
        delta[0][tag] = initial_log + emission_log
        psi[0][tag] = ""

    # 2. Rekursi
    for i in range(1, n):
        current_word = sentence[i]

        for current_tag in tags:
            emission_log = safe_log(get_emission_prob(current_word, current_tag, emit_prob, tags))

            best_score = -float("inf")
            best_previous_tag = None

            for previous_tag in tags:
                transition_log = safe_log(trans_prob[previous_tag].get(current_tag, EPSILON))
                score = delta[i - 1][previous_tag] + transition_log + emission_log

                if score > best_score:
                    best_score = score
                    best_previous_tag = previous_tag

            delta[i][current_tag] = best_score
            psi[i][current_tag] = best_previous_tag

    # 3. Terminasi
    best_last_tag = None
    best_last_score = -float("inf")
    for tag in tags:
        if delta[n - 1][tag] > best_last_score:
            best_last_score = delta[n - 1][tag]
            best_last_tag = tag

    # 4. Backtracking
    best_path = [best_last_tag]
    for i in range(n - 1, 0, -1):
        best_path.append(psi[i][best_path[-1]])
    best_path.reverse()

    return best_path


def viterbi_debug(sentence, tags, pi, trans_prob, emit_prob):
    """
    Sama seperti viterbi(), tetapi juga mencetak tabel Viterbi.
    """
    if len(sentence) == 0:
        return []

    sentence = [normalize_word(word) for word in sentence]
    n = len(sentence)

    delta = [{} for _ in range(n)]
    psi = [{} for _ in range(n)]

    first_word = sentence[0]
    for tag in tags:
        initial_log = safe_log(pi.get(tag, EPSILON))
        emission_log = safe_log(get_emission_prob(first_word, tag, emit_prob, tags))
        delta[0][tag] = initial_log + emission_log
        psi[0][tag] = ""

    for i in range(1, n):
        current_word = sentence[i]
        for current_tag in tags:
            emission_log = safe_log(get_emission_prob(current_word, current_tag, emit_prob, tags))

            best_score = -float("inf")
            best_previous_tag = None

            for previous_tag in tags:
                transition_log = safe_log(trans_prob[previous_tag].get(current_tag, EPSILON))
                score = delta[i - 1][previous_tag] + transition_log + emission_log

                if score > best_score:
                    best_score = score
                    best_previous_tag = previous_tag

            delta[i][current_tag] = best_score
            psi[i][current_tag] = best_previous_tag

    print_viterbi_table(delta, sentence, tags)

    best_last_tag = None
    best_last_score = -float("inf")
    for tag in tags:
        if delta[n - 1][tag] > best_last_score:
            best_last_score = delta[n - 1][tag]
            best_last_tag = tag

    best_path = [best_last_tag]
    for i in range(n - 1, 0, -1):
        best_path.append(psi[i][best_path[-1]])
    best_path.reverse()

    return best_path


# =========================================================
# 4) EVALUATE
# =========================================================
def evaluate(test_sentences, tags, pi, trans_prob, emit_prob):
    """
    Menghitung token-level accuracy pada test set.
    """
    total_token = 0
    correct_token = 0

    for sentence in test_sentences:
        words = [word for word, _ in sentence]
        gold_tags = [tag for _, tag in sentence]

        predicted_tags = viterbi(words, tags, pi, trans_prob, emit_prob)

        for gold, pred in zip(gold_tags, predicted_tags):
            total_token += 1
            if gold == pred:
                correct_token += 1

    if total_token == 0:
        return 0.0

    return correct_token / total_token


def evaluate_per_tag(test_sentences, tags, pi, trans_prob, emit_prob):
    """
    Menghitung precision, recall, F1, dan accuracy per tag.

    Accuracy per tag di sini didefinisikan sebagai:
        TP / (TP + FP + FN)

    Metrik ini dipakai agar lebih informatif daripada TN-based accuracy
    yang biasanya terlalu besar pada tugas multi-class tagging.
    """
    true_positive = Counter()
    false_positive = Counter()
    false_negative = Counter()

    for sentence in test_sentences:
        words = [word for word, _ in sentence]
        gold_tags = [tag for _, tag in sentence]
        predicted_tags = viterbi(words, tags, pi, trans_prob, emit_prob)

        for gold, pred in zip(gold_tags, predicted_tags):
            if gold == pred:
                true_positive[gold] += 1
            else:
                false_positive[pred] += 1
                false_negative[gold] += 1

    result = {}

    for tag in tags:
        tp = true_positive[tag]
        fp = false_positive[tag]
        fn = false_negative[tag]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        result[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    return result


# =========================================================
# 5) PREDICT SENTENCE
# =========================================================
def predict_sentence(raw_sentence, tags, pi, trans_prob, emit_prob):
    """
    Menerima string kalimat baru, lalu mengembalikan list pasangan (kata, tag).
    """
    words = simple_tokenize(raw_sentence)
    predicted_tags = viterbi(words, tags, pi, trans_prob, emit_prob)
    return list(zip(words, predicted_tags))


# =========================================================
# HELPER TAMBAHAN
# =========================================================
def split_train_test(sentences, train_ratio=0.8):
    """
    Membagi data menjadi train dan test secara sederhana.
    """
    split_index = int(len(sentences) * train_ratio)
    return sentences[:split_index], sentences[split_index:]


def print_prediction_table(pairs):
    """
    Menampilkan hasil prediksi sederhana.
    """
    if len(pairs) == 0:
        print("(kalimat kosong)")
        return

    max_word_length = max(len(word) for word, _ in pairs)
    for word, tag in pairs:
        print(word.ljust(max_word_length), tag)


def print_prediction_detail(words, predicted_tags, gold_tags=None):
    """
    Menampilkan tabel detail prediksi TANPA garis antar baris.
    """
    headers = ["No", "Word", "Pred", "Gold", "Status", "OOV", "Catatan"]
    rows = []

    for i, word in enumerate(words):
        pred = predicted_tags[i]
        gold = gold_tags[i] if gold_tags is not None else "-"
        status = "-"
        if gold_tags is not None:
            status = "BENAR" if pred == gold else "SALAH"

        rows.append([
            i + 1,
            word,
            pred,
            gold,
            status,
            word_status(word),
            build_notes(word, i),
        ])

    widths = [4, 24, 6, 6, 8, 10, 28]
    print_table(headers, rows, widths=widths)


def print_model_info(tags, pi, vocab):
    """
    Menampilkan informasi dasar model.
    """
    print("=" * 60)
    print("INFO MODEL HMM")
    print("=" * 60)
    print("Jumlah tag     :", len(tags))
    print("Jumlah vocab   :", len(vocab))
    print("Daftar tag     :", ", ".join(tags))
    print("\nInitial probability (sample):")

    max_show = min(10, len(tags))
    for i in range(max_show):
        tag = tags[i]
        print(f"  pi[{tag}] = {pi[tag]:.6f}")

    print("=" * 60)


def print_test_sentence_batch(test_sentences, start_index, batch_size):
    """
    Menampilkan daftar kalimat test dalam batch.
    """
    end_index = min(start_index + batch_size, len(test_sentences))
    headers = ["No", "Kalimat", "Panjang"]
    rows = []

    for idx in range(start_index, end_index):
        sentence = test_sentences[idx]
        words = [word for word, _ in sentence]
        rows.append([idx + 1, " ".join(words), len(words)])

    print("\nDAFTAR KALIMAT TEST")
    print_table(headers, rows, widths=[6, 72, 8])
    print(f"Menampilkan kalimat test {start_index + 1} s.d. {end_index} dari {len(test_sentences)}")


def analyze_test_sentence(test_sentences, sentence_number, tags, pi, trans_prob, emit_prob, debug=False):
    """
    Menampilkan analisis detail untuk satu kalimat pada data test.
    """
    if sentence_number < 1 or sentence_number > len(test_sentences):
        print(f"[ERROR] Nomor kalimat harus antara 1 sampai {len(test_sentences)}.")
        return

    sentence = test_sentences[sentence_number - 1]
    words = [word for word, _ in sentence]
    gold_tags = [tag for _, tag in sentence]

    print("\n" + "=" * 80)
    print(f"ANALISIS KALIMAT TEST NOMOR {sentence_number}")
    print("=" * 80)
    print("Kalimat:")
    print(" ", " ".join(words))

    if debug:
        predicted_tags = viterbi_debug(words, tags, pi, trans_prob, emit_prob)
    else:
        predicted_tags = viterbi(words, tags, pi, trans_prob, emit_prob)

    print("\nDetail prediksi:")
    print_prediction_detail(words, predicted_tags, gold_tags)

    correct = sum(1 for gold, pred in zip(gold_tags, predicted_tags) if gold == pred)
    accuracy = correct / len(words) if words else 0.0
    print(f"\nAkurasi kalimat ini: {correct}/{len(words)} = {accuracy * 100:.2f}%")


def analyze_free_sentence(raw_sentence, tags, pi, trans_prob, emit_prob, debug=False):
    """
    Menampilkan analisis detail untuk kalimat bebas.
    """
    words = simple_tokenize(raw_sentence)

    print("\n" + "=" * 80)
    print("ANALISIS KALIMAT BEBAS")
    print("=" * 80)
    print("Kalimat:")
    print(" ", " ".join(words))

    if debug:
        predicted_tags = viterbi_debug(words, tags, pi, trans_prob, emit_prob)
    else:
        predicted_tags = viterbi(words, tags, pi, trans_prob, emit_prob)

    print("\nDetail prediksi:")
    print_prediction_detail(words, predicted_tags, gold_tags=None)


def interactive_mode(test_sentences, tags, pi, trans_prob, emit_prob, batch_size=DEFAULT_BATCH_SIZE):
    """
    Mode interaktif:
    - nomor saja        -> analisis kalimat test nomor tsb
    - debug: 10         -> debug kalimat test nomor 10
    - next:             -> batch berikutnya
    - prev:             -> batch sebelumnya
    - list:             -> tampilkan batch saat ini
    - bebas: <teks>     -> prediksi kalimat bebas
    - debug: <teks>     -> debug kalimat bebas
    - teks bebas biasa  -> prediksi langsung
    - exit              -> keluar
    """
    print("\n" + "=" * 60)
    print("MODE INTERAKTIF TEST SET / KALIMAT BEBAS")
    print("=" * 60)

    batch_start = 0
    print_test_sentence_batch(test_sentences, batch_start, batch_size)

    print("\nPerintah yang tersedia:")
    print("  1              -> tampilkan analisis kalimat test nomor 1")
    print("  debug: 1       -> tampilkan debug Viterbi untuk kalimat test nomor 1")
    print("  next:          -> tampil batch kalimat test berikutnya")
    print("  prev:          -> tampil batch kalimat test sebelumnya")
    print("  list:          -> tampilkan lagi batch saat ini")
    print("  bebas: <teks>  -> prediksi kalimat bebas")
    print("  debug: <teks>  -> debug untuk kalimat bebas")
    print("  exit           -> keluar program")

    while True:
        user_input = input("\nMasukkan perintah / kalimat: ").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Program selesai.")
            break

        if user_input == "":
            print("Input kosong. Silakan isi perintah atau kalimat.")
            continue

        lower_input = user_input.lower()

        if lower_input == "next:":
            if batch_start + batch_size < len(test_sentences):
                batch_start += batch_size
            print_test_sentence_batch(test_sentences, batch_start, batch_size)
            continue

        if lower_input == "prev:":
            batch_start = max(0, batch_start - batch_size)
            print_test_sentence_batch(test_sentences, batch_start, batch_size)
            continue

        if lower_input == "list:":
            print_test_sentence_batch(test_sentences, batch_start, batch_size)
            continue

        if re.fullmatch(r"\d+", user_input):
            analyze_test_sentence(test_sentences, int(user_input), tags, pi, trans_prob, emit_prob, debug=False)
            continue

        if lower_input.startswith("debug:"):
            payload = user_input[6:].strip()

            if payload == "":
                print("[ERROR] Isi setelah 'debug:' kosong.")
                continue

            if re.fullmatch(r"\d+", payload):
                analyze_test_sentence(test_sentences, int(payload), tags, pi, trans_prob, emit_prob, debug=True)
            else:
                analyze_free_sentence(payload, tags, pi, trans_prob, emit_prob, debug=True)
            continue

        if lower_input.startswith("bebas:"):
            payload = user_input[6:].strip()
            if payload == "":
                print("[ERROR] Isi setelah 'bebas:' kosong.")
                continue

            analyze_free_sentence(payload, tags, pi, trans_prob, emit_prob, debug=False)
            continue

        # Selain itu, anggap sebagai kalimat bebas biasa
        analyze_free_sentence(user_input, tags, pi, trans_prob, emit_prob, debug=False)


# =========================================================
# MAIN PROGRAM
# =========================================================
def main():
    """
    Alur utama program:
    1. membaca dataset
    2. membagi train/test
    3. training HMM
    4. evaluasi accuracy
    5. metrik per-tag
    6. mode interaktif
    """
    if len(sys.argv) < 2:
        print("Penggunaan:")
        print("  python3 viterbi_pos.py <path_dataset_tsv>")
        print("")
        print("Contoh:")
        print("  python3 viterbi_pos.py Data_POS-Tag-ID.tsv")
        sys.exit(1)

    dataset_path = sys.argv[1]

    print("[INFO] Membaca dataset dari:", dataset_path)
    sentences = load_data(dataset_path)
    print("[INFO] Total kalimat dibaca:", len(sentences))

    if len(sentences) == 0:
        print("[ERROR] Dataset kosong atau gagal dibaca.")
        sys.exit(1)

    train_sentences, test_sentences = split_train_test(sentences, train_ratio=0.8)

    print("[INFO] Jumlah train sentences:", len(train_sentences))
    print("[INFO] Jumlah test sentences :", len(test_sentences))

    print("[INFO] Training HMM...")
    tags, pi, trans_prob, emit_prob, vocab = train_hmm(train_sentences)
    print("[INFO] Training selesai.")

    print_model_info(tags, pi, vocab)

    print("[INFO] Evaluasi pada test set...\n")
    accuracy = evaluate(test_sentences, tags, pi, trans_prob, emit_prob)
    print(f"[RESULT] Token-level Accuracy: {accuracy * 100:.2f}%")

    print("\n[INFO] Menghitung metrik per-tag ...")
    metrics = evaluate_per_tag(test_sentences, tags, pi, trans_prob, emit_prob)

    headers = ["No", "Tag", "Nama POS", "Precision", "Recall", "F1", "Accuracy"]
    rows = []

    for i, tag in enumerate(tags, start=1):
        rows.append([
            i,
            tag,
            POS_TAG_NAMES.get(tag, "-"),
            f"{metrics[tag]['precision']:.4f}",
            f"{metrics[tag]['recall']:.4f}",
            f"{metrics[tag]['f1']:.4f}",
            f"{metrics[tag]['accuracy']:.4f}",
        ])

    print_table(headers, rows, widths=[4, 5, 32, 10, 10, 10, 10])

    interactive_mode(test_sentences, tags, pi, trans_prob, emit_prob, batch_size=DEFAULT_BATCH_SIZE)


if __name__ == "__main__":
    main()