import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Detect ReLU functions
# ------------------------------------------------------------
FUNC_HEADER = re.compile(
    r'FUNC_PREFIX\s+void\s+(node__[A-Za-z0-9_]*Relu[A-Za-z0-9_]*)\s*\((.*?)\)',
    re.S
)

# ------------------------------------------------------------
# Extract functions with brace matching
# ------------------------------------------------------------
def extract_functions(text):
    funcs = []

    for m in FUNC_HEADER.finditer(text):
        name = m.group(1)
        sig = m.group(2)

        start = m.start()
        brace_start = text.find('{', m.end())
        if brace_start == -1:
            continue

        depth = 1
        i = brace_start + 1

        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1

        end = i
        body = text[brace_start+1:end-1]

        funcs.append((start, end, name, sig, body))

    return funcs


# ------------------------------------------------------------
# Remove loops with bound 1
# ------------------------------------------------------------
def remove_trivial_loops(body):
    # remove loops like: for(... < 1 ...)
    return re.sub(
        r'for\s*\([^;]+<\s*1\s*;[^)]*\)\s*\{',
        '',
        body
    ).replace('}', '', 1)  # remove one matching brace


# ------------------------------------------------------------
# Insert PIPELINE pragma at innermost loop
# ------------------------------------------------------------
def insert_pipeline(body):
    lines = body.split('\n')

    # find last "for (" line
    last_for = -1
    for i, line in enumerate(lines):
        if "for" in line and "(" in line:
            last_for = i

    if last_for == -1:
        return body

    # insert pragma after that loop
    indent = re.match(r'\s*', lines[last_for]).group(0)
    lines.insert(last_for + 1, indent + "#pragma HLS PIPELINE II=1")  # or just #pragma HLS PIPELINE

    return "\n".join(lines)


# ------------------------------------------------------------
# Optional: replace if-based ReLU with ternary
# ------------------------------------------------------------
def simplify_relu_math(body):
    body = re.sub(
        r'Y\[(.*?)\]\s*=\s*0;\s*if\s*\(\s*X\[\1\]\s*>\s*0\s*\)\s*\{\s*Y\[\1\]\s*=\s*X\[\1\];\s*\}',
        r'Y[\1] = (X[\1] > 0) ? X[\1] : (fixed)0;',
        body,
        flags=re.S
    )
    return body


# ------------------------------------------------------------
# Apply transformation
# ------------------------------------------------------------
def transform_relu(name, sig, body):

    body = remove_trivial_loops(body)
    # body = simplify_relu_math(body)
    body = insert_pipeline(body)

    return f"""
FUNC_PREFIX void {name}({sig})
{{
{body}
}}
}}
"""


# ------------------------------------------------------------
# Main simplify
# ------------------------------------------------------------
def simplify(text):

    funcs = extract_functions(text)

    for start, end, name, sig, body in reversed(funcs):

        print("[OPTIMIZE RELU]", name)

        new_func = transform_relu(name, sig, body)

        text = text[:start] + new_func + text[end:]

    return text


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_relu.py <input.cpp>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    new_text = simplify(text)

    out = path  # .with_suffix(".relu.cpp")
    out.write_text(new_text)

    print(f"\n[OK] Written: {out}")


if __name__ == "__main__":
    main()