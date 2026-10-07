import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import jiwer
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

SIZE = 384
TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TrOCR on your own images and labels.")
    p.add_argument("--data_csv", required=True, help="CSV with one row per image")
    p.add_argument("--image_root", default="", help="folder the image paths are relative to")
    p.add_argument("--path_col", default="path", help="CSV column with the image path")
    p.add_argument("--label_col", default="label", help="CSV column with the correct text")
    p.add_argument("--group_col", default="",
                   help="optional CSV column (writer, document, form id). Rows with the same value "
                        "stay together in train or validation, which prevents leakage")
    p.add_argument("--val_csv", default="", help="optional separate validation CSV")
    p.add_argument("--test_csv", default="", help="optional test CSV, scored once at the end")
    p.add_argument("--val_frac", type=float, default=0.1, help="share of data held out when no --val_csv")
    p.add_argument("--base_model", default="microsoft/trocr-base-handwritten",
                   help="Hugging Face model name or a folder with a model you trained before")
    p.add_argument("--out_dir", default="output")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=96, help="longest text, in tokens")
    p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2,
                   help="data loading processes. Keep 0 on Windows if you see errors")
    p.add_argument("--cache_dir", default=os.path.join(tempfile.gettempdir(), "htr_cache"),
                   help="where resized copies of your images are stored")
    p.add_argument("--no_aug", action="store_true", help="turn off augmentation")
    p.add_argument("--photo_aug", action="store_true", help="add perspective warps for phone photos")
    p.add_argument("--smoke", action="store_true", help="tiny quick run to check everything works")
    p.add_argument("--no_zip", action="store_true", help="do not make model.zip at the end")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def load_manifest(csv_path, image_root, path_col, label_col, group_col):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    needed = [path_col, label_col] + ([group_col] if group_col else [])
    for col in needed:
        if col not in df.columns:
            sys.exit(f"I cannot find the column '{col}' in {csv_path}. Columns I see: {list(df.columns)}")
    out = pd.DataFrame()
    out["path"] = [p if os.path.isabs(p) else os.path.join(image_root, p) for p in df[path_col]]
    out["label"] = [norm(s) for s in df[label_col]]
    out["group"] = df[group_col].values if group_col else ""

    exists = out.path.map(os.path.exists)
    if (~exists).any():
        print(f"Warning: {(~exists).sum()} image files not found, skipping them. First ones: "
              f"{list(out.path[~exists].head(3))}")
    empty = out.label == ""
    if empty.any():
        print(f"Warning: {empty.sum()} rows have an empty label, skipping them.")
    out = out[exists & ~empty].reset_index(drop=True)
    if out.empty:
        sys.exit("No usable rows. Check --image_root and the column names.")
    return out


def split_data(df, frac, use_groups, seed):
    rng = np.random.RandomState(seed)
    if use_groups and df.group.nunique() >= 2:
        groups = np.array(sorted(df.group.unique()))
        rng.shuffle(groups)
        n = min(max(1, int(len(groups) * frac)), len(groups) - 1)
        mask = df.group.isin(set(groups[:n]))
    else:
        if use_groups:
            print("Warning: only one group found, splitting rows randomly instead.")
        idx = rng.permutation(len(df))
        n = min(max(1, int(len(df) * frac)), len(df) - 1)
        mask = np.zeros(len(df), dtype=bool)
        mask[idx[:n]] = True
        mask = pd.Series(mask, index=df.index)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def cache_path(cache_dir, path):
    return os.path.join(cache_dir, hashlib.md5(path.encode("utf-8")).hexdigest() + ".png")


def build_cache(df, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    def one(path):
        c = cache_path(cache_dir, path)
        if os.path.exists(c):
            return True
        try:
            Image.open(path).convert("L").resize((SIZE, SIZE), Image.BICUBIC).save(c)
            return True
        except Exception as e:
            print(f"Warning: cannot read {path}: {e}")
            return False

    paths = list(df.path.unique())
    with ThreadPoolExecutor(8) as ex:
        ok = dict(zip(paths, ex.map(one, paths)))
    keep = df.path.map(ok)
    if (~keep).any():
        print(f"Skipping {(~keep).sum()} unreadable images.")
    return df[keep].reset_index(drop=True)


def make_aug(photo_aug):
    steps = [
        T.RandomApply([T.RandomAffine(2, translate=(0.01, 0.03), scale=(0.9, 1.05),
                                      shear=(-5, 5), fill=255)], p=0.7),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3)], p=0.7),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3),
    ]
    if photo_aug:
        steps.insert(0, T.RandomApply([T.RandomPerspective(0.15, p=1.0, fill=255)], p=0.4))
    return T.Compose(steps)


class TextDS(Dataset):
    def __init__(self, df, tok, cache_dir, max_len, aug=None):
        self.df, self.tok, self.cache_dir, self.max_len, self.aug = df, tok, cache_dir, max_len, aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(cache_path(self.cache_dir, r.path)).convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        ids = self.tok(r.label, max_length=self.max_len, truncation=True).input_ids
        return TO_TENSOR(img), ids


def collate(batch):
    pv = torch.stack([b[0] for b in batch])
    longest = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), longest), -100)
    for i, b in enumerate(batch):
        labels[i, :len(b[1])] = torch.tensor(b[1])
    return pv, labels


@torch.no_grad()
def predict(model, tok, df, args, device):
    model.eval()
    loader = DataLoader(TextDS(df, tok, args.cache_dir, args.max_len), batch_size=32,
                        collate_fn=collate, num_workers=args.workers)
    preds = []
    for pv, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model.generate(pv.to(device), max_length=args.max_len)
        preds += [norm(p) for p in tok.batch_decode(out, skip_special_tokens=True)]
    return preds


def metrics(refs, preds):
    return {
        "cer": float(jiwer.cer(list(refs), list(preds))),
        "wer": float(jiwer.wer(list(refs), list(preds))),
        "exact": float(np.mean([r == p for r, p in zip(refs, preds)])),
    }


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: no GPU found. Training on CPU is extremely slow. Use a small dataset and few epochs.")
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_manifest(args.data_csv, args.image_root, args.path_col, args.label_col, args.group_col)
    use_groups = bool(args.group_col)
    if args.val_csv:
        train_df = df
        val_df = load_manifest(args.val_csv, args.image_root, args.path_col, args.label_col, args.group_col)
    else:
        train_df, val_df = split_data(df, args.val_frac, use_groups, args.seed)
    test_df = None
    if args.test_csv:
        test_df = load_manifest(args.test_csv, args.image_root, args.path_col, args.label_col, args.group_col)

    if args.smoke:
        args.epochs = 1
        train_df, val_df = train_df.head(40), val_df.head(20)
        test_df = test_df.head(20) if test_df is not None else None
    print(f"train {len(train_df)} | val {len(val_df)} | test {0 if test_df is None else len(test_df)}")
    if len(train_df) < 50 and not args.smoke:
        print("Warning: very little training data. Expect weak results.")

    chars = Counter("".join(df.label))
    rare = sorted(c for c, n in chars.items() if n < 5)
    print(f"{len(chars)} different characters in the labels. Seen fewer than 5 times: {rare}")

    parts = [train_df, val_df] + ([test_df] if test_df is not None else [])
    allp = build_cache(pd.concat(parts), args.cache_dir)
    good = set(allp.path)
    train_df = train_df[train_df.path.isin(good)].reset_index(drop=True)
    val_df = val_df[val_df.path.isin(good)].reset_index(drop=True)
    if test_df is not None:
        test_df = test_df[test_df.path.isin(good)].reset_index(drop=True)

    proc = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model).to(device)
    tok = proc.tokenizer
    for c in (model.config, model.generation_config):
        c.decoder_start_token_id = tok.cls_token_id
        c.pad_token_id = tok.pad_token_id
        c.eos_token_id = tok.sep_token_id
    model.generation_config.max_length = args.max_len
    model.generation_config.num_beams = 1

    sample = list(train_df.label.head(2000))
    too_long = np.mean([len(tok(s).input_ids) > args.max_len for s in sample])
    if too_long > 0:
        print(f"Warning: {too_long:.1%} of labels are longer than --max_len {args.max_len} tokens "
              f"and will be cut. Raise --max_len.")

    aug = None if args.no_aug else make_aug(args.photo_aug)
    train_loader = DataLoader(TextDS(train_df, tok, args.cache_dir, args.max_len, aug),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.workers)

    use_amp = device == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 100) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dir = os.path.join(args.out_dir, "best")
    best_cer, history = 9.0, []
    val_refs = list(val_df.label)

    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for i, (pv, lab) in enumerate(train_loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(pixel_values=pv.to(device), labels=lab.to(device)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if (i + 1) % 100 == 0:
                print(f"epoch {ep + 1} step {i + 1}/{len(train_loader)} loss {run / 100:.3f}", flush=True)
                run = 0.0
        m = metrics(val_refs, predict(model, tok, val_df, args, device))
        history.append({"epoch": ep + 1, **m})
        print(f"== epoch {ep + 1}: validation CER {m['cer']:.3f} | WER {m['wer']:.3f} | "
              f"exact match {m['exact']:.3f}", flush=True)
        if m["cer"] < best_cer:
            best_cer = m["cer"]
            model.save_pretrained(best_dir)
            proc.save_pretrained(best_dir)
            print(f"   saved the best model so far to {best_dir}", flush=True)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    if test_df is not None and len(test_df):
        best = VisionEncoderDecoderModel.from_pretrained(best_dir).to(device)
        preds = predict(best, tok, test_df, args, device)
        m = metrics(list(test_df.label), preds)
        print(f"TEST ({len(test_df)} images): CER {m['cer']:.3f} | WER {m['wer']:.3f} | exact match {m['exact']:.3f}")
        out = test_df[["path", "label"]].copy()
        out["pred"] = preds
        out["correct"] = out.label == out.pred
        out.sort_values("correct").to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)
        print(f"Row by row results: {os.path.join(args.out_dir, 'test_predictions.csv')} (wrong ones first)")

    if not args.no_zip:
        z = shutil.make_archive(os.path.join(args.out_dir, "model"), "zip", best_dir)
        print(f"Zipped model: {z}")
    print(f"Done. Your model is in: {best_dir}")


if __name__ == "__main__":
    main()
