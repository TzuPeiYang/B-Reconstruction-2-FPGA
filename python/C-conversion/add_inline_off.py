import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Match function definitions (onnx2c style + generic C)
# ------------------------------------------------------------
FUNC_PATTERN = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_\s\*\[\]]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;]*\))\s*\{',
    re.S
)

# ------------------------------------------------------------
# Check if pragma already exists in function body
# ------------------------------------------------------------
def has_inline_off(body):
    return "#pragma HLS INLINE OFF" in body


# ------------------------------------------------------------
# Insert pragma after opening brace
# ------------------------------------------------------------
def insert_pragma(match, text):

    start = match.end()  # position right after '{'

    # find end of function by counting braces
    depth = 1
    i = start

    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    func_body = text[start:i]

    if has_inline_off(func_body):
        return match.group(0)  # no change

    insertion = "\n#pragma HLS INLINE OFF\n"

    return match.group(0) + insertion


# ------------------------------------------------------------
# Main transform
# ------------------------------------------------------------
def add_inline_off(text):

    result = []
    last = 0

    for match in FUNC_PATTERN.finditer(text):

        result.append(text[last:match.start()])

        new_header = insert_pragma(match, text)
        result.append(new_header)

        last = match.end()

    result.append(text[last:])

    return "".join(result)


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python add_inline_off.py <input.cpp>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    new_text = add_inline_off(text)

    out = path  # .with_suffix(".no_inline.cpp")
    out.write_text(new_text)

    print(f"[OK] Written: {out}")


if __name__ == "__main__":
    main()