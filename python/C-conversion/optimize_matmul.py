import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Match MatMul functions
# ------------------------------------------------------------
FUNC_HEADER = re.compile(
    r'FUNC_PREFIX\s+void\s+(node__[A-Za-z0-9_]*MatMul)\s*\((.*?)\)',
    re.S
)

# ------------------------------------------------------------
# Extract full function safely (brace-balanced)
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

        depth = 0
        i = brace_start

        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1

        end = i
        funcs.append((start, end, name, sig))

    return funcs


# ------------------------------------------------------------
# Extract dims
# ------------------------------------------------------------
def get_dims(sig, name):
    m = re.search(rf'{name}\s*(\[[^\]]+\](?:\[[^\]]+\])*)', sig)
    if not m:
        return None
    return list(map(int, re.findall(r'\d+', m.group(1))))


# ------------------------------------------------------------
# Generate optimized MatMul
# ------------------------------------------------------------
def generate_matmul(name, sig):
    A = get_dims(sig, "A")
    B = get_dims(sig, "B")
    Y = get_dims(sig, "Y")

    if not A or not B or not Y:
        return None

    M = A[1]   # rows
    K = A[2]   # reduction
    N = B[2]   # cols

    UNROLL = min(16, K)  # safe default

    code = f"""
FUNC_PREFIX void {name}({sig})
{{
#pragma HLS ARRAY_PARTITION variable=A cyclic factor={UNROLL} dim=3
#pragma HLS ARRAY_PARTITION variable=B cyclic factor={UNROLL} dim=2

for (unsigned b = 0; b < 1; b++) {{
for (unsigned i = 0; i < {M}; i++) {{
for (unsigned j = 0; j < {N}; j++) {{

#pragma HLS PIPELINE II=1

    fixed acc = 0;

    for (unsigned k = 0; k < {K}; k++) {{
#pragma HLS UNROLL factor={UNROLL}
        acc += A[0][i][k] * B[0][k][j];
    }}

    Y[0][i][j] = acc;
}}
}}

"""
    return code


# ------------------------------------------------------------
# Replace safely (NO regex body replacement)
# ------------------------------------------------------------
def replace_function(text, name, new_func):
    pattern = re.compile(
        rf'FUNC_PREFIX\s+void\s+{name}\s*\(.*?\)\s*\{{.*?\}}',
        re.S
    )
    return pattern.sub(new_func, text, count=1)


# ------------------------------------------------------------
# Main transform
# ------------------------------------------------------------
def transform(text):
    funcs = extract_functions(text)

    for _, _, name, sig in reversed(funcs):

        if "MatMul" not in name:
            continue

        print("[MATMUL OPT]", name)

        new_func = generate_matmul(name, sig)
        if not new_func:
            continue

        text = replace_function(text, name, new_func)

    return text


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_matmul.py input.cpp")
        return

    path = Path(sys.argv[1])
    text = path.read_text()

    new_text = transform(text)

    out = path # .with_suffix(".matmul.cpp")
    out.write_text(new_text)

    print(f"[OK] Written: {out}")


if __name__ == "__main__":
    main()