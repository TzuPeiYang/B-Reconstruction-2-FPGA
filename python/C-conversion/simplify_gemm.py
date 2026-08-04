import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Match GEMM functions
# ------------------------------------------------------------
gemm_pattern = re.compile(
    r'(FUNC_PREFIX\s+void\s+(node__[A-Za-z0-9_]+_Gemm)\s*\((.*?)\)\s*\{)(.*?)\n\}',
    re.S
)

# ------------------------------------------------------------
# Extract dimensions
# ------------------------------------------------------------
def extract_dims(sig):

    # remove const qualifiers to avoid parsing noise
    sig_clean = sig.replace("const", "").replace("static", "")

    def extract(var):
        # match A[...][...]
        m = re.search(rf'\b{var}\s*\[([^\]]+)\](\[[^\]]+\])?', sig_clean)
        if not m:
            return None

        # grab all numbers after variable name
        nums = re.findall(r'\d+', m.group(0))
        return [int(x) for x in nums]

    A = extract("A")
    B = extract("B")

    if not A or not B:
        print("[WARN] failed to parse A/B dims:", sig)
        return None

    # A = [M, K]
    if len(A) >= 2:
        M, K = A[0], A[1]
    else:
        M, K = 1, A[0]

    # B = [N, K]
    if len(B) >= 2:
        N, K2 = B[0], B[1]
    else:
        N, K2 = 1, B[0]

    if K != K2:
        print(f"[WARN] K mismatch A={K}, B={K2}")
        K = min(K, K2)

    return M, K, N


# ------------------------------------------------------------
# Detect layout (must be B[c][i])
# ------------------------------------------------------------
def is_supported(body):
    return "B[c][i]" in body


# ------------------------------------------------------------
# Generate simplified GEMM
# ------------------------------------------------------------
def generate(name, sig, M, K, N):

    return f"""
FUNC_PREFIX void {name}({sig})
{{
#pragma HLS INLINE OFF


const int M = {M};
const int K = {K};
const int N = {N};

for (uint32_t r = 0; r < M; r++)
    for (uint32_t c = 0; c < N; c++) {{
    #pragma HLS PIPELINE II=1

        fixed acc = C[c];

        for (uint32_t i = 0; i < K; i++) {{
            #pragma HLS UNROLL factor={K}
            acc += A[r][i] * B[c][i];
        }}

        Y[r][c] = acc;
    }}
}}
"""


# ------------------------------------------------------------
# Main transform
# ------------------------------------------------------------
def simplify(text):

    output = text
    replacements = []

    for match in gemm_pattern.finditer(text):

        full = match.group(0)
        name = match.group(2)
        sig = match.group(3)
        body = match.group(4)

        dims = extract_dims(sig)
        if dims is None:
            print(f"[SKIP] {name}: cannot parse dims")
            continue

        M, K, N = dims

        if not is_supported(body):
            print(f"[SKIP] {name}: unsupported layout")
            continue

        print(f"[SIMPLIFY] {name}: M={M}, K={K}, N={N}")

        new_func = generate(name, sig, M, K, N)
        replacements.append((full, new_func))

    for old, new in replacements:
        output = output.replace(old, new)

    return output


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python simplify_gemm.py <input.cpp>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    new_text = simplify(text)

    out = path # .with_suffix(".gemm_simple.cpp")
    out.write_text(new_text)

    print(f"\n[OK] Written: {out}")


if __name__ == "__main__":
    main()