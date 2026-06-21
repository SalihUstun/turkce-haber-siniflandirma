import os
import re
import sys
from predict import load_model, predict

MODEL_DIR  = "./model"
HABERLER_DIR = "./haberler"

KATEGORI_KLASOR = {
    "ekonomi"     : "0_ekonomi",
    "kultur-sanat": "1_kultur-sanat",
    "magazin"     : "2_magazin",
    "saglik"      : "3_saglik",
    "siyaset"     : "4_siyaset",
    "spor"        : "5_spor",
    "teknoloji"   : "6_teknoloji",
}


def parse_haberler(filepath: str) -> list[str]:
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    texts = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        if lines[0].strip().rstrip(".").isdigit():
            text = "\n".join(lines[1:]).strip()
        else:
            text = block
        if text:
            texts.append(text)
    return texts


def load_all_haberler() -> list[tuple[str, str]]:
    dataset = []
    for kategori, klasor in KATEGORI_KLASOR.items():
        filepath = os.path.join(HABERLER_DIR, klasor, "haberler.txt")
        if not os.path.exists(filepath):
            print(f"[UYARI] Dosya bulunamadi: {filepath}")
            continue
        texts = parse_haberler(filepath)
        for text in texts:
            dataset.append((kategori, text))
    return dataset


def run_tests(tfidf_word, tfidf_char, models, id2label, dataset):
    dogru   = 0
    yanlis  = 0
    yanlis_ornekler = []

    print("\n" + "=" * 70)
    print(f"{'#':<5} {'GERCEK':<16} {'TAHMİN':<16} {'GUVEN':>7}  SONUC")
    print("=" * 70)

    for i, (gercek, text) in enumerate(dataset, 1):
        sonuc    = predict(text, tfidf_word, tfidf_char, models, id2label)
        tahmin   = sonuc["tahmin"]
        guven    = sonuc["guven"]
        dogru_mu = (tahmin == gercek)

        simge = "OK " if dogru_mu else "XX"
        print(f"{i:<5} {gercek:<16} {tahmin:<16} {guven*100:>6.1f}%  {simge}")

        if dogru_mu:
            dogru += 1
        else:
            yanlis += 1
            yanlis_ornekler.append((gercek, tahmin, guven, text))

    toplam   = dogru + yanlis
    accuracy = dogru / toplam * 100

    print("=" * 70)
    print(f"\nSONUCLAR")
    print(f"  Toplam haber : {toplam}")
    print(f"  Dogru        : {dogru}")
    print(f"  Yanlis       : {yanlis}")
    print(f"  Accuracy     : {accuracy:.1f}%")

    if yanlis_ornekler:
        print(f"\n--- YANLIS TAHMINLER ({len(yanlis_ornekler)} adet) ---")
        for gercek, tahmin, guven, text in yanlis_ornekler:
            kelimeler = text.split()
            snip = " ".join(kelimeler[:18]) + (" ..." if len(kelimeler) > 18 else "")
            print(f"\n  Gercek : {gercek}")
            print(f"  Tahmin : {tahmin}  ({guven*100:.1f}%)")
            print(f"  Metin  : {snip}")

    print()
    return accuracy


def kategori_bazli_rapor(tfidf_word, tfidf_char, models, id2label, dataset):
    from collections import defaultdict
    kat_dogru  = defaultdict(int)
    kat_toplam = defaultdict(int)

    for gercek, text in dataset:
        sonuc  = predict(text, tfidf_word, tfidf_char, models, id2label)
        tahmin = sonuc["tahmin"]
        kat_toplam[gercek] += 1
        if tahmin == gercek:
            kat_dogru[gercek] += 1

    print("=" * 50)
    print(f"{'KATEGORİ':<18} {'DOGRU':>6} {'TOPLAM':>7} {'ACC':>8}")
    print("=" * 50)
    for kat in sorted(kat_toplam):
        d = kat_dogru[kat]
        t = kat_toplam[kat]
        acc = d / t * 100
        bar = "#" * int(acc / 5)
        print(f"{kat:<18} {d:>6} {t:>7} {acc:>7.1f}%  {bar}")
    print("=" * 50)


def main():
    tfidf_word, tfidf_char, models, id2label = load_model(MODEL_DIR)

    dataset = load_all_haberler()
    if not dataset:
        print("HATA: Haber verisi yuklenemedi.")
        sys.exit(1)

    print(f"Yuklenen haber sayisi: {len(dataset)}")

    run_tests(tfidf_word, tfidf_char, models, id2label, dataset)
    kategori_bazli_rapor(tfidf_word, tfidf_char, models, id2label, dataset)


if __name__ == "__main__":
    main()
