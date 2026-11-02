"""Download the PraNet training and test sets (Google Drive links from the PraNet README,
https://github.com/DengPingFan/PraNet), unzip them, and check the folder layout.

    python scripts/download_data.py                                   # -> data/polyp
    python scripts/download_data.py --dest /content/data/polyp \
        --zip-dir /content/drive/MyDrive/advanced_ml/data_zips        # Colab

--zip-dir is a folder that keeps TrainDataset.zip and TestDataset.zip between sessions (e.g. on
Google Drive). Zips found there are used without downloading; zips downloaded by gdown are copied
there, so Google Drive's download quota is hit at most once.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import IMG_EXTS, TEST_SETS, locate_pair_dirs
from src.utils import ROOT

FILES = {
    "TrainDataset": "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb",   # Kvasir-SEG 900 + CVC-ClinicDB 550
    "TestDataset": "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao",    # Kvasir, ClinicDB, ColonDB, ETIS, CVC-300
}
EXPECTED = {"TrainDataset": 1450, "Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380,
            "ETIS-LaribPolypDB": 196, "CVC-300": 60}

MANUAL_HELP = """
Không tải được {name}.zip từ Google Drive (thường do link PraNet vượt quota tải trong ngày).
Cách xử lý:
  1. Mở link sau bằng trình duyệt (đã đăng nhập Google) và tải file zip về máy:
       {link}
  2. Tải file lên Google Drive, vào thư mục --zip-dir (ví dụ MyDrive/advanced_ml/data_zips/),
     giữ nguyên tên {name}.zip.
  3. Chạy lại lệnh này với --zip-dir trỏ đến thư mục đó. Script sẽ giải nén từ Drive.
"""


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS) if folder.is_dir() else 0


def download(name: str, file_id: str, out: Path):
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is not installed: pip install gdown")
    print(f"Downloading {name}.zip from Google Drive ...")
    try:
        result = gdown.download(id=file_id, output=str(out), quiet=False)
    except Exception as e:                       # gdown raises on quota / permission pages
        print(f"gdown error: {e}")
        result = None
    if not result or not out.exists() or not zipfile.is_zipfile(out):
        if out.exists():
            out.unlink()                         # an HTML error page saved as .zip
        sys.exit(MANUAL_HELP.format(name=name, link=f"https://drive.google.com/file/d/{file_id}/view"))


def extract(zip_path: Path, dest: Path, name: str):
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    if not (dest / name).is_dir():               # zip without the top-level folder, or nested deeper
        found = [p for p in dest.rglob(name) if p.is_dir()]
        if not found:
            sys.exit(f"After unzipping, {dest / name} is missing. Contents of {dest}: "
                     f"{sorted(p.name for p in dest.iterdir())}")
        shutil.move(str(found[0]), str(dest / name))
    for junk in dest.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)


def check_layout(dest: Path) -> bool:
    rows = [("TrainDataset", dest / "TrainDataset")] + [(s, dest / "TestDataset" / s) for s in TEST_SETS]
    ok = True
    print(f"\n{'folder':<20}{'images':>8}{'masks':>8}{'expected':>10}   image / mask folders")
    for label, folder in rows:
        try:
            img_dir, mask_dir = locate_pair_dirs(folder)
            n_img, n_mask = count_images(img_dir), count_images(mask_dir)
            where = f"{img_dir.relative_to(folder)} / {mask_dir.relative_to(folder)}"
        except FileNotFoundError as e:
            n_img = n_mask = 0
            where = str(e)
        flag = "" if n_img == n_mask == EXPECTED[label] else "   <-- check"
        ok &= n_img == n_mask and n_img > 0
        print(f"{label:<20}{n_img:>8}{n_mask:>8}{EXPECTED[label]:>10}   {where}{flag}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest_pos", nargs="?", help=argparse.SUPPRESS)      # old usage: positional dest
    ap.add_argument("--dest", default=None, help="output folder (default data/polyp)")
    ap.add_argument("--zip-dir", default=None, help="folder that keeps the zip files between sessions")
    args = ap.parse_args()
    dest = Path(args.dest or args.dest_pos or "data/polyp")
    dest = dest if dest.is_absolute() else ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)
    zip_dir = Path(args.zip_dir) if args.zip_dir else None
    if zip_dir:
        zip_dir.mkdir(parents=True, exist_ok=True)

    for name, file_id in FILES.items():
        if (dest / name).is_dir() and any((dest / name).rglob("*.png")):
            print(f"{name}: already present in {dest}")
            continue
        zip_path = next((p for p in [zip_dir / f"{name}.zip" if zip_dir else None, dest / f"{name}.zip"]
                         if p is not None and p.exists() and zipfile.is_zipfile(p)), None)
        if zip_path is None:
            zip_path = dest / f"{name}.zip"
            download(name, file_id, zip_path)
            if zip_dir:
                shutil.copy2(zip_path, zip_dir / zip_path.name)
                print(f"Saved a copy to {zip_dir} for later sessions")
        extract(zip_path, dest, name)

    if not check_layout(dest):
        sys.exit(f"\nThe data layout in {dest} is not as expected (see data/README.md).")
    print(f"\nData ready in {dest}. Next: python scripts/prepare_splits.py --root {dest}")


if __name__ == "__main__":
    main()
