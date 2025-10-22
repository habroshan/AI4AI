# ChestXray14 (NIH)
This dataset is NOT included. Obtain it from the NIH source (accept the license/terms), then place files as follows:

Expected layout (relative to repo root):

data/chestxray14/Data_Entry_2017.csv
images/
<all image files here, any NIH subfolders are fine>

Notes
- The pipeline uses a binary pneumonia vs. other task.
- Image files may be `.png` or `.jpg` — loader is case‑insensitive.
- Fast local storage is recommended.

Quick sanity check:
1) Verify counts: `Data_Entry_2017.csv` ~112k rows; images ~112k files.
2) Run: `python scripts/check_data.py --dataset chestxray14`

If your images live elsewhere (e.g., `/mnt/ssd/NIH/`), you can symlink:
`ln -s /mnt/ssd/NIH/images data/chestxray14/images`
`ln -s /mnt/ssd/NIH/Data_Entry_2017.csv data/chestxray14/Data_Entry_2017.csv`
