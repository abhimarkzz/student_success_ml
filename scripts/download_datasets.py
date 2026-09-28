from pathlib import Path
import argparse
import io
import zipfile
import urllib.request

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

def extract_all_zips(data: bytes, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    queue = [(data, output_dir)]

    while queue:
        zip_bytes, destination = queue.pop()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue

                name = Path(member.filename)

                if name.suffix.lower() == ".zip":
                    nested_bytes = z.read(member)
                    queue.append((nested_bytes, destination))
                else:
                    # Flatten only the final CSV files we need.
                    if name.name in {"student-mat.csv", "student-por.csv"}:
                        out = destination / name.name
                        out.write_bytes(z.read(member))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uci", action="store_true")
    args = parser.parse_args()

    if not args.uci:
        parser.error("Use --uci")

    output_dir = Path("data/raw/uci_student_performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading official UCI archive: {UCI_URL}")

    with urllib.request.urlopen(UCI_URL) as response:
        archive = response.read()

    extract_all_zips(archive, output_dir)

    mat = output_dir / "student-mat.csv"
    por = output_dir / "student-por.csv"

    print(f"UCI data normalized into {output_dir.resolve()}")

    if mat.exists():
        print("  ✓ student-mat.csv")
    else:
        print("  ✗ student-mat.csv NOT FOUND")

    if por.exists():
        print("  ✓ student-por.csv")
    else:
        print("  ✗ student-por.csv NOT FOUND")

    if not mat.exists():
        raise FileNotFoundError(
            f"Could not find {mat}. The UCI archive structure may have changed."
        )

    print(
        "Official UCI source: "
        "https://archive.ics.uci.edu/dataset/320/student+performance"
    )

if __name__ == "__main__":
    main()
