import os
import sys
import json
import pickle
import argparse
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from mlp_model import NewsMLP

MODEL_DIR = "./model"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


def load_model(model_dir: str):
    required = ["tfidf_word.pkl", "tfidf_char.pkl",
                "mlp_config.json", "id2label.json"]
    for fname in required:
        if not os.path.exists(os.path.join(model_dir, fname)):
            print(f"\nHATA: '{fname}' bulunamadi -> {model_dir}")
            print("Kaggle'dan indirilen model/ klasorunu bu dizine koyun.\n")
            sys.exit(1)

    print(f"\nModel yukleniyor  ->  {model_dir}")
    print(f"Cihaz             ->  {DEVICE.upper()}\n")

    with open(os.path.join(model_dir, "tfidf_word.pkl"), "rb") as f:
        tfidf_word = pickle.load(f)
    with open(os.path.join(model_dir, "tfidf_char.pkl"), "rb") as f:
        tfidf_char = pickle.load(f)

    with open(os.path.join(model_dir, "mlp_config.json")) as f:
        cfg = json.load(f)

    with open(os.path.join(model_dir, "id2label.json"), encoding="utf-8") as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}

    n_ensemble = cfg.pop("n_ensemble", 1)
    models = []
    for i in range(n_ensemble):
        fname = f"mlp_model_{i}.pt" if n_ensemble > 1 else "mlp_model.pt"
        path  = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            path = os.path.join(model_dir, "mlp_model.pt")
        m = NewsMLP(**cfg)
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        m.load_state_dict(state)
        m = m.to(DEVICE)
        m.eval()
        models.append(m)

    print(f"Ensemble  : {len(models)} model")
    print(f"Parametre : {models[0].count_parameters():,}")
    print(f"Kategori  : {len(id2label)}")
    for i, l in sorted(id2label.items()):
        print(f"  {i}  {l}")
    print()
    return tfidf_word, tfidf_char, models, id2label


def predict(text: str, tfidf_word, tfidf_char, models, id2label: dict) -> dict:
    feat = sp.hstack(
        [tfidf_word.transform([text]), tfidf_char.transform([text])],
        format="csr",
    )
    x = torch.tensor(feat.toarray(), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = torch.stack([m(x) for m in models]).mean(0)

    probs   = F.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax().item())

    all_scores = {
        id2label[i]: round(float(probs[i]), 4)
        for i in range(len(id2label))
    }
    all_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

    return {
        "tahmin"     : id2label[pred_id],
        "guven"      : round(float(probs[pred_id]), 4),
        "tum_skorlar": all_scores,
    }


def print_result(text: str, result: dict, verbose: bool = True):
    label   = result["tahmin"]
    percent = result["guven"] * 100

    print("\n" + "=" * 62)
    words = text.split()
    snip  = " ".join(words[:20]) + (" ..." if len(words) > 20 else "")
    print(f"Metin    : {snip}")
    print("-" * 62)
    print(f"Kategori : {label.upper()}")
    print(f"Guven    : {percent:.1f}%")

    if verbose:
        print("-" * 62)
        print("Tum skorlar:")
        for lbl, sc in result["tum_skorlar"].items():
            bar   = "#" * int(sc * 28)
            arrow = " <" if lbl == label else ""
            print(f"  {lbl:<16} {sc*100:5.1f}%  {bar}{arrow}")

    print("=" * 62)


def interactive_mode(tfidf_word, tfidf_char, models, id2label):
    print("Turkce Haber Kategorilendirici (TF-IDF + MLP Ensemble)")
    print(f"Kategoriler: {', '.join(sorted(id2label.values()))}")
    print(f"Cihaz: {DEVICE.upper()}")
    print("-" * 62)
    print("Cikmak icin 'q' veya Ctrl+C\n")

    while True:
        try:
            text = input("Haber metni: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGorüsmek uzere!")
            break

        if not text:
            continue
        if text.lower() in ("q","cikis"):
            print("Gorüsmek uzere!")
            break

        result = predict(text, tfidf_word, tfidf_char, models, id2label)
        print_result(text, result)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Turkce Haber Kategorilendirici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",   default=MODEL_DIR, help="Model klasoru (varsayilan: ./model)")
    parser.add_argument("--text",    type=str,           help="Siniflandirilacak metin")
    parser.add_argument("--no-bars", action="store_true", help="Skor barlarini gizle")
    args = parser.parse_args()

    tfidf_word, tfidf_char, models, id2label = load_model(args.model)

    if args.text:
        result = predict(args.text, tfidf_word, tfidf_char, models, id2label)
        print_result(args.text, result, verbose=not args.no_bars)
    else:
        interactive_mode(tfidf_word, tfidf_char, models, id2label)


if __name__ == "__main__":
    main()
