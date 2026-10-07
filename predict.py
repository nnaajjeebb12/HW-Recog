import argparse
import os
import re
import sys

import torch
from PIL import Image
from torchvision import transforms as T
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

SIZE = 384
TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


class TextReader:
    def __init__(self, model_dir, device=None, max_len=96):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.proc = TrOCRProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(self.device).eval()
        self.tok = self.proc.tokenizer
        self.max_len = max_len

    def _tensor(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("L").resize((SIZE, SIZE), Image.BICUBIC).convert("RGB")
        return TO_TENSOR(image)

    @staticmethod
    def _clean(text, strip_spaces):
        text = re.sub(r"\s+", "" if strip_spaces else " ", text).strip()
        return text

    @torch.no_grad()
    def read(self, images, batch_size=32, strip_spaces=False):
        results = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self._tensor(x) for x in images[i:i + batch_size]]).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
                out = self.model.generate(batch, max_length=self.max_len)
            results += [self._clean(p, strip_spaces)
                        for p in self.tok.batch_decode(out, skip_special_tokens=True)]
        return results

    @torch.no_grad()
    def read_matching(self, image, pattern, num_beams=10, strip_spaces=False):
        x = self._tensor(image).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
            out = self.model.generate(x, max_length=self.max_len, num_beams=num_beams,
                                      num_return_sequences=num_beams)
        guesses = [self._clean(p, strip_spaces) for p in self.tok.batch_decode(out, skip_special_tokens=True)]
        for g in guesses:
            if re.fullmatch(pattern, g):
                return g, "fixed with beam search, please verify"
        return guesses[0], "no match, check by hand"


def main():
    ap = argparse.ArgumentParser(description="Read text from a folder of images.")
    ap.add_argument("--model", required=True, help="folder with your trained model (the 'best' folder)")
    ap.add_argument("--images", required=True, help="folder with the images")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--pattern", default="", help="regular expression the text must fully match")
    ap.add_argument("--strip_spaces", action="store_true", help="remove all spaces from the readings")
    ap.add_argument("--recursive", action="store_true", help="also look inside subfolders")
    args = ap.parse_args()

    files = []
    if args.recursive:
        for root, _, names in os.walk(args.images):
            files += [os.path.join(root, n) for n in names if n.lower().endswith(IMAGE_EXTS)]
    else:
        files = [os.path.join(args.images, n) for n in os.listdir(args.images) if n.lower().endswith(IMAGE_EXTS)]
    files.sort()
    if not files:
        sys.exit(f"No images found in {args.images}")
    print(f"Reading {len(files)} images...")

    reader = TextReader(args.model, max_len=args.max_len)
    preds = reader.read(files, batch_size=args.batch_size, strip_spaces=args.strip_spaces)
    status = ["ok" if p else "empty, check by hand" for p in preds]

    if args.pattern:
        for i, p in enumerate(preds):
            if not re.fullmatch(args.pattern, p):
                preds[i], status[i] = reader.read_matching(files[i], args.pattern,
                                                           strip_spaces=args.strip_spaces)

    import pandas as pd
    df = pd.DataFrame({"file": files, "prediction": preds, "status": status})
    df.to_csv(args.out, index=False)
    print(df.status.value_counts().to_string())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
