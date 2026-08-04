import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Match Conv functions
# ------------------------------------------------------------
FUNC_HEADER = re.compile(
    r'FUNC_PREFIX\s+void\s+(node__[A-Za-z0-9_]*Conv)\s*\((.*?)\)',
    re.S
)

# ------------------------------------------------------------
# Extract full function (brace-safe)
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
# Extract dimensions
# ------------------------------------------------------------
def get_dims(sig, name):
    m = re.search(rf'{name}\s*(\[[^\]]+\](?:\[[^\]]+\])*)', sig)
    if not m:
        return None
    return list(map(int, re.findall(r'\d+', m.group(1))))


# ------------------------------------------------------------
# Generate optimized Conv (3D + 4D)
# ------------------------------------------------------------
def generate_conv(name, sig):
    X = get_dims(sig, "x")
    W = get_dims(sig, "w")
    Y = get_dims(sig, "y")

    if not X or not W or not Y:
        return None

    N = X[0]
    C = X[1]
    spatial = X[2:]
    M = W[0]

    # weight index (handles 3D vs 4D)
    w_idx = ''.join(['[0]' for _ in range(len(W) - 2)])

    # spatial loop variables
    loop_vars = ["h", "i", "j", "k"]  # enough for safety
    dims = spatial
    n_spatial = len(dims)

    loops = []
    for d in range(n_spatial):
        loops.append((loop_vars[d], dims[d]))

    # build indexing string
    idx = ''.join([f'[{v}]' for v, _ in loops])

    # identify pipeline loop (second last spatial)
    pipeline_idx = max(0, n_spatial - 2)  # second-last loop

    code = f"""
FUNC_PREFIX void {name}({sig})
{{
#pragma HLS ARRAY_PARTITION variable=x cyclic factor={C} dim=2
#pragma HLS ARRAY_PARTITION variable=w cyclic factor={C} dim=2

for (uint32_t b = 0; b < {N}; b++) {{
#pragma HLS loop_flatten
for (uint32_t m = 0; m < {M}; m++) {{
#pragma HLS loop_flatten
"""

    # generate spatial loops
    for i, (v, d) in enumerate(loops):
        code += f"for (uint32_t {v} = 0; {v} < {d}; {v}++) {{\n"
        if i == pipeline_idx:
            code += "#pragma HLS PIPELINE II=1\n"

    code += f"""
fixed acc = bias[m];

for (uint32_t c = 0; c < {C}; c++) {{
#pragma HLS UNROLL factor={C}
    acc += x[b][c]{idx} * w[m][c]{w_idx};
}}

y[b][m]{idx} = acc;
"""

    # close loops
    for _ in loops:
        code += "}\n"

    code += """
}
}
}
"""

    return code


# ------------------------------------------------------------
# Replace function safely
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

    for start, end, name, sig in reversed(funcs):

        if "Conv" not in name:
            continue

        new_func = generate_conv(name, sig)
        if not new_func:
            continue

        # sanity check
        if text[end-1] != '}':
            print(f"[ERROR] Bad extraction: {name}")
            continue

        print("[REWRITE]", name)

        text = text[:start] + new_func + text[end:]

    return text


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python rewrite_conv.py input.cpp")
        return

    path = Path(sys.argv[1])
    text = path.read_text()

    new_text = transform(text)

    out = path  # .with_suffix(".conv.cpp")
    out.write_text(new_text)

    print(f"[OK] Output: {out}")


if __name__ == "__main__":
    main()