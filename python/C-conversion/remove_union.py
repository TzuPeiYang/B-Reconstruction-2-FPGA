import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Match full union block
# ------------------------------------------------------------
UNION_PATTERN = re.compile(
    r'union\s+(tensor_[A-Za-z0-9_]+)\s*\{(.*?)\};',
    re.S
)

# union variable declarations
VAR_DECL_PATTERN = re.compile(
    r'\b(?:static\s+|const\s+|volatile\s+)*union\s+(tensor_[A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*;'
)

# ------------------------------------------------------------
# Lift union body
# ------------------------------------------------------------
def lift_unions(text):
    union_names = set()

    def repl(m):
        uname = m.group(1)
        body = m.group(2)

        union_names.add(uname)

        print(f"[LIFT] {uname}")

        # keep body declarations
        return body

    new_text = UNION_PATTERN.sub(repl, text)

    return new_text, union_names


# ------------------------------------------------------------
# Remove union variable declarations
# ------------------------------------------------------------
def remove_union_vars(text, union_names):

    def repl(m):
        utype = m.group(1)
        if utype in union_names:
            print(f"[REMOVE VAR] {m.group(0)}")
            return ""
        return m.group(0)

    return VAR_DECL_PATTERN.sub(repl, text)


# ------------------------------------------------------------
# Remove tuX. prefixes
# ------------------------------------------------------------
def remove_prefixes(text):
    return re.sub(r'\btu\d+\.', '', text)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python lift_union_body.py <input.cpp>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    text, union_names = lift_unions(text)
    text = remove_union_vars(text, union_names)
    text = remove_prefixes(text)

    out = path  # .with_suffix(".lifted.cpp")
    out.write_text(text)

    print(f"\n[OK] Written: {out}")


if __name__ == "__main__":
    main()