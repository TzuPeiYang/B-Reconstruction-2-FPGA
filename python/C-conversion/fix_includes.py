import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Match all #include lines
# ------------------------------------------------------------
INCLUDE_PATTERN = re.compile(
    r'^\s*#include\s+[<"].*[>"].*\n?',
    re.MULTILINE
)

# ------------------------------------------------------------
# Remove all includes
# ------------------------------------------------------------
def remove_includes(text):
    return INCLUDE_PATTERN.sub('', text)

# ------------------------------------------------------------
# Find safe insertion point (after comments/header)
# ------------------------------------------------------------
def find_insert_index(lines):
    in_block_comment = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # handle block comments
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue

        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block_comment = True
            continue

        # skip single-line comments and empty lines
        if stripped.startswith("//") or stripped == "":
            continue

        # first real code line
        return i

    return len(lines)

# ------------------------------------------------------------
# Insert include safely
# ------------------------------------------------------------
def insert_include(text):
    lines = text.splitlines()

    idx = find_insert_index(lines)

    lines.insert(idx, '#include "weights/weights.h"\n')

    return "\n".join(lines) + "\n"

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python replace_includes.py <input.cpp>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    text = remove_includes(text)
    text = insert_include(text)

    out = path  # .with_suffix(".includes.cpp")
    out.write_text(text)

    print(f"[OK] Written: {out}")

if __name__ == "__main__":
    main()