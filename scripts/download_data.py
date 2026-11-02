"""Download the PraNet training and test sets from the Google Drive links in the PraNet README
(https://github.com/DengPingFan/PraNet) and unzip them into data/polyp/.

    pip install gdown
    python scripts/download_data.py

If Google Drive refuses the download (quota), open the two links in a browser, download the zip
files manually into data/polyp/, and run the script again: it only unzips what is missing.
"""
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT

FILES = {
    "TrainDataset": "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb",   # Kvasir-SEG 900 + CVC-ClinicDB 550
    "TestDataset": "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao",    # Kvasir, ClinicDB, ColonDB, ETIS, CVC-300
}


def main(dest="data/polyp"):
    dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        if (dest / name).is_dir():
            print(f"{name}: already present")
            continue
        zip_path = dest / f"{name}.zip"
        if not zip_path.exists():
            import gdown
            print(f"Downloading {name} ...")
            gdown.download(id=file_id, output=str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        if not (dest / name).is_dir():           # zip without a top-level folder, or nested one level
            found = [p for p in dest.rglob(name) if p.is_dir()]
            if found:
                shutil.move(str(found[0]), str(dest / name))
            else:
                raise RuntimeError(f"After unzipping, {dest / name} is missing. Check the zip layout "
                                   f"and arrange it as described in data/README.md.")
        print(f"{name}: extracted to {dest / name}")
    for junk in dest.glob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)
    print("Done. Next: python scripts/prepare_splits.py")


if __name__ == "__main__":
    main(*sys.argv[1:])
