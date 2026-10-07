"""Restore the two CSVs from the original authors' public archive, with checksums."""
import hashlib
import io
import json
import urllib.request
import zipfile
from .data import ROOT

URL="https://drive.google.com/uc?export=download&id=1p13zKI5Q0QejLB-Q8jQBLRiJDRQNmGgD"
def main():
    expected=json.loads((ROOT/"data/raw/checksums.json").read_text())
    with urllib.request.urlopen(URL,timeout=60) as response:
        archive=zipfile.ZipFile(io.BytesIO(response.read()))
    for name,digest in expected.items():
        data=archive.read(name)
        if hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError(f"Source checksum changed for {name}; inspect before using.")
        (ROOT/"data/raw"/name).write_bytes(data)
        print(f"Verified {name}")

if __name__=="__main__":main()
