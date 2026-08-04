import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Match full function blocks
# ------------------------------------------------------------
FUNC_PATTERN = re.compile(
    r'(FUNC_PREFIX\s+void\s+(node_[A-Za-z0-9_]+)\s*\((.*?)\)\s*\{.*?\n\})',
    re.S
)

# ------------------------------------------------------------
# Detect constant node (generalized)
# ------------------------------------------------------------
def is_constant_node(name, signature, func_text):

    # must contain "Constant" in name
    if "Constant" not in name:
        return False

    # must contain Constant comment
    if "/* Constant */" not in func_text:
        return False

    # must have trivial body
    if "(void)output" not in func_text:
        return False

    # check arguments: no real inputs
    args = [a.strip() for a in signature.split(",") if a.strip()]

    for a in args:
        # if it's const AND not output → it's a real input → reject
        if "const" in a and "output" not in a:
            return False

    return True


# ------------------------------------------------------------
# Remove function calls
# ------------------------------------------------------------
def remove_calls(text, func_name):

    pattern = re.compile(rf'\b{func_name}\s*\([^;]*\);\s*\n?')
    return pattern.sub('', text)


# ------------------------------------------------------------
# Main transform
# ------------------------------------------------------------
def remove_constant_nodes(text):

    new_text = text
    removed = []

    for match in FUNC_PATTERN.finditer(text):

        full_func = match.group(1)
        name = match.group(2)
        signature = match.group(3)

        if not is_constant_node(name, signature, full_func):
            continue

        print(f"[REMOVE CONST] {name}")

        # remove function
        new_text = new_text.replace(full_func, "")
        removed.append(name)

    # remove calls
    for name in removed:
        new_text = remove_calls(new_text, name)

    return new_text


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python remove_constant_nodes.py <input.cpp>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    new_text = remove_constant_nodes(text)

    out = path # .with_suffix(".no_const.cpp")
    out.write_text(new_text)

    print(f"\n[OK] Written: {out}")


if __name__ == "__main__":
    main()