import hashlib
import os
import subprocess
import sys

# Replace with the SHA-256 hash of build_windows_exe.bat
EXPECTED_HASH = "b899ed9f0bba2894dee59ccb3feb481ba058551fea994ed244b792fb49c43580"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_FILE = os.path.join(SCRIPT_DIR, "build_windows_exe.bat")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    if not os.path.exists(BATCH_FILE):
        print("ERROR: build_windows_exe.bat not found.")
        input("Press Enter to exit...")
        sys.exit(1)

    actual_hash = sha256(BATCH_FILE)

    if actual_hash != EXPECTED_HASH.upper():
        print("=" * 50)
        print("ERROR: Integrity check failed!")
        print()
        print("Expected:")
        print(EXPECTED_HASH)
        print()
        print("Found:")
        print(actual_hash)
        print("=" * 50)
        input("Press Enter to exit...")
        sys.exit(1)

    result = subprocess.run(
        ["cmd.exe", "/c", BATCH_FILE],
        cwd=SCRIPT_DIR
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()