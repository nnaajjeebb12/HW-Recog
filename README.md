# Teach a model to read your images

Hi everyone. This guide walks you through training your own text reading model on your own computer, from your data all the way to using it in your project. I built this while working on my own handwriting project, so I'll also tell you where things went wrong for me, so they don't go wrong for you.

You get four files:

- `train.py` trains the model on your data
- `predict.py` uses the trained model to read new images
- `requirements.txt` lists the libraries you need
- this `README.md`

## What this does, and what it does not do

You give the model small images of text, each one with the correct text written out. It learns from those examples. After training, you give it a new image and it gives you back the text.

The important rule: **one image is one line, one word, or one field of text.** A name, a date, a number, a short line from a notebook. That is what the model is built for.

Under the hood this fine tunes a model called TrOCR. It is free and it works with English and other Latin script text. If you need another writing system, this will not work out of the box, so talk to me first.

## What you need on your computer

**Python 3.10 or newer.** Check with `python --version`.

**A GPU makes a huge difference.**

- An NVIDIA GPU is the best option. We trained on a 16 GB card without problems. If yours has less memory and you get an out of memory error, lower `--batch_size` to 4 or 2
- With no GPU, it still runs, but it is extremely slow. Think many hours even for a small dataset. If that is your situation, keep your data small, use `--epochs 1`, and ask me about options

**Disk space and internet.** The first run downloads the base model, about 1.3 GB, from Hugging Face. You need internet for that once. After that it is cached on your machine.

## Step 1: Set up your environment

Open Command Prompt or PowerShell. Go to the folder where you put the four files, then create a clean environment so nothing clashes with your other projects.

```
python -m venv venv
venv\Scripts\activate
```

Install PyTorch first, because the right version depends on your machine. Go to pytorch.org, scroll to the install selector, pick Windows and your GPU type (CUDA for NVIDIA, CPU if you have none), and run the command it gives you.

Then install the rest:

```
pip install -r requirements.txt
```

Check that PyTorch can see your GPU:

```
python -c "import torch; print(torch.cuda.is_available())"
```

`True` means your NVIDIA GPU is ready.

## Step 2: Prepare your data

You need two things.

**1. A folder of images.** PNG or JPG, any size. Each image shows one piece of text.

**2. A CSV file** with one row per image. At least two columns: the image path and the correct text.

```
path,label
img_0001.png,SAN ROQUE NHS
img_0002.png,2003-2004
img_0003.png,"MENDOZA, ANA"
```

The column names `path` and `label` are the defaults. If yours are different, tell the script with `--path_col` and `--label_col`. Paths are relative to your image folder, or full paths if you prefer. If a label contains a comma, put quotes around it, like the last row above.

**Optional third column: a group.** If your data comes from different writers, documents, or forms, add a column that says which one each row came from, for example `writer_07`. Then train with `--group_col group`. This keeps all rows from the same writer or document together, either in training or in validation, never split across both. Without it, the model can score well just because it already saw that person's handwriting, and your numbers will look better than reality. I learned this one the hard way, and it matters a lot.

### Rules for the labels

This is where most people lose accuracy, so be careful.

- Type exactly what is written, including capitals, spaces and punctuation. `1982 - 1983` and `1982-1983` are different labels to the model
- Be consistent. Pick one convention and stick to it everywhere. If you write names as `SURNAME, GIVEN NAME`, do it for every row
- If some text is crossed out, pick one symbol for it (I used `#`) and use it the same way every time
- Do not leave labels empty. The script skips those rows
- Check your labels by looking at some images next to their text. Wrong labels teach the model wrong things

### How much data

More is better, but quality matters more than quantity. A few hundred correct examples can already show you a clear improvement. A few thousand is comfortable. If you have fewer than about 50, the results will be weak.

### Keep a test set

Set aside some real examples that **never** go into training. Put them in their own CSV and use it as the test set. This is the only honest way to know how good your model really is. Do not fix your model by looking at the test set again and again, or it stops being a fair test.

If you generate fake training data (for example, rendering text with fonts), never test on fake data. A model can score nearly perfect on data that looks like what it trained on and still fail on real images. Always test on real ones.

## Step 3: Do a smoke test first

Before you spend hours on training, run a tiny version that finishes in a few minutes. It tells you whether your paths and CSV are right. Type the whole command on one line:

```
python train.py --data_csv labels.csv --image_root images --smoke
```

Change `labels.csv` and `images` to where your own files are. If it ends with "Done", everything works. If it shows an error, read the message. It usually tells you exactly what is wrong, such as a column name that does not match.

## Step 4: Train for real

```
python train.py --data_csv labels.csv --image_root images --group_col group --test_csv test.csv --out_dir my_run --epochs 3
```

Leave out `--group_col` if you have no group column, and leave out `--test_csv` if you have no test file yet.

The script will:

1. Read your CSV and tell you about any problems
2. Resize all images once and save copies in a temporary folder, which makes training much faster
3. Train, printing the loss now and then (it should go down)
4. After each epoch, score itself on the validation data and save the model if it is the best so far
5. At the end, score the test set if you gave one
6. Zip the best model for you

### Useful options

- `--epochs` how many passes over the data. 3 is a good start
- `--lr` learning rate. Keep the default unless the loss turns into `nan`, then try `1e-5`
- `--batch_size` lower it to 4 if you get an out of memory error
- `--photo_aug` adds perspective warps. Use it if your real images are phone photos
- `--no_aug` turns augmentation off
- `--max_len` longest text allowed, in tokens. Raise it if your texts are long
- `--base_model` which model to start from. The default reads handwriting. For printed text, try `microsoft/trocr-base-printed`. You can also give a folder with a model you trained before, to keep improving it with new data
- `--val_csv` your own validation file instead of an automatic split

Using a model you trained before as the starting point is useful when you get new data. Use a lower learning rate, like `1e-5`, and mix in some of your old data so the model does not forget what it learned.

### While it trains

- Keep the terminal open. Closing it stops the training
- Stop your laptop from sleeping. Plug it in and change the power settings, because a sleeping laptop pauses or kills the run
- The first loss line appears after 100 steps, which can take a few minutes. A quiet screen at the start is normal
- Training is saved after every epoch that improves, so if it crashes late, you still have the best model so far in `my_run/best`

## Step 5: Read your results

The script prints three numbers.

- **CER** is the character error rate. It counts how many characters are wrong, missing, or extra, as a share of all characters. 0.05 means about 5 wrong characters in every 100. Lower is better
- **WER** is the same idea for whole words. It is always higher than CER
- **Exact match** is the share of images where the reading was perfect

Be careful with how you read them:

- **Validation score** tells you if training is working. It can look good even when the model fails on new data
- **Test score** is the one to trust, and only if the test set is real data that never touched training
- A test set of 20 images tells you almost nothing. Use as many as you can, at least 100
- Look at the actual mistakes. Open `my_run/test_predictions.csv`, which lists wrong readings first. Patterns show up fast: a letter that keeps getting confused, dropped digits, trouble with long texts. The mistakes tell you what data to add next
- A single score can hide a lot. In my project, numbers looked fine overall while one field type (long ID numbers with repeated digits) kept losing digits. I only noticed by reading the wrong ones

## Step 6: Find your model

When training finishes, your model is in the folder `my_run/best`. It holds the files the model needs, and the big one is `model.safetensors`. There is also `my_run/model.zip` with the same thing zipped, handy for sharing or backing up.

Copy the whole `best` folder, not just one file. All the files must stay together or the model will not load.

## Step 7: Use your model

### On a folder of images

```
python predict.py --model my_run/best --images my_new_images --out predictions.csv
```

You get a CSV with the file name, the reading, and a status column.

### Inside your own Python project

Copy `predict.py` next to your code, and put the `best` folder somewhere your project can reach.

```python
from predict import TextReader

reader = TextReader("path/to/best")
print(reader.read(["snippet1.png", "snippet2.png"]))
```

It also accepts PIL images, so you can pass in a crop straight from your own code. Create the `TextReader` once when your program starts, not for every image, because loading the model takes a few seconds.

### If your text follows a pattern

Many fields have a fixed shape: a phone number, an ID with exactly 12 digits, a year range. Use that to catch mistakes. Pass a pattern and the script retries any reading that does not fit:

```
python predict.py --model my_run/best --images ids --pattern "\d{12}" --strip_spaces
```

The status column then tells you what to trust:

- `ok` means the reading fit the pattern straight away
- `fixed with beam search, please verify` means the first reading was wrong, and a later guess fit. It fits the pattern, but it can still be wrong, so look at it
- `no match, check by hand` means nothing fit

In my project this recovered most of the wrong length readings without any retraining. A reading that fits the pattern is not automatically correct. A swapped digit still passes. So treat the pattern check as a safety net and not as proof.

## Phone photos and real world images

Train with images that look like what the model will see in real life. If your real images are phone photos but your training images are clean scans, results will be worse than your scores suggest. Use `--photo_aug` and include some real photos in training.

Very wide images (a long line of text) get squeezed into a square when the model looks at them, and some text at the end can get lost. If you see readings that stop early, split long lines into two crops.

## When things go wrong

**"I cannot find the column ..."** The column names in your CSV do not match. Use `--path_col` and `--label_col`, or rename the columns.

**"image files not found"** Your `--image_root` is wrong, or the paths in the CSV do not match the folder. Open the CSV and compare a path with a real file on disk.

**Out of memory.** Lower `--batch_size` to 4, then 2.

**Loss is `nan`.** Lower `--lr` to `1e-5`.

**Windows shows errors about processes or multiprocessing.** Add `--workers 0`.

**It says "Using device: cpu" and you have an NVIDIA GPU.** PyTorch was installed without GPU support. Go back to pytorch.org, pick the CUDA option, and reinstall.

**Download errors on the first run.** The base model downloads from Hugging Face, so you need internet, and some school or work networks block it. Try another network.

**Good scores but bad results on real images.** Your test data is too similar to your training data, or your training images do not look like the real ones. Fix the data first, not the code.

## Privacy

Many of you will work with images of real people: names, ID numbers, grades, addresses. Handle them carefully.

- Keep real personal data on your own machine or in storage you have permission to use. Do not upload it to public sites
- Do not share your trained model publicly if it was trained on private data, because models can remember what they saw
- If you can, use sample or fake data while you build and test, and only touch real data at the end, with permission

## A last word

Your first run will not be perfect, and that is normal. Train, look at the mistakes, fix the data, and train again. Most of the improvement comes from better data, not from clever tricks. Good luck, and ask me if you get stuc
