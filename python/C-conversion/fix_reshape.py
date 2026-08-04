import re

def fix_reshape_pointer_casts(code):
    # Match the full Reshape function
    pattern = re.compile(
        r'(FUNC_PREFIX void \w+\(\s*'
        r'const (\S+)\s+data\[[^\]]*\](?:\[[^\]]*\])*,\s*'  # capture data type
        r'const int64_t shape\[[^\]]*\],\s*'
        r'(\S+)\s+reshaped\[[^\]]*\](?:\[[^\]]*\])*\s*\)'   # capture reshaped type
        r'\s*\{.*?/\*Reshape\*/.*?\})',                       # function body
        re.DOTALL
    )

    def fix_body(m):
        full = m.group(1)
        data_type = m.group(2)      # e.g. ap_fixed<19,6>
        reshaped_type = m.group(3)  # e.g. ap_fixed<19,6>

        # Replace pointer cast on data_ptr
        full = re.sub(
            r'\(ap_fixed<[^>]+>\s*\*\)data',
            f'({data_type}*)data',
            full
        )
        # Replace pointer cast on reshaped_ptr
        full = re.sub(
            r'\(ap_fixed<[^>]+>\s*\*\)reshaped',
            f'({reshaped_type}*)reshaped',
            full
        )
        # Fix the declared pointer types too: ap_fixed<X,Y> *data_ptr
        full = re.sub(
            r'ap_fixed<[^>]+>\s*\*data_ptr',
            f'{data_type} *data_ptr',
            full
        )
        full = re.sub(
            r'ap_fixed<[^>]+>\s*\*reshaped_ptr',
            f'{reshaped_type} *reshaped_ptr',
            full
        )
        return full

    return pattern.sub(fix_body, code)


if __name__ == "__main__":
    import sys

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    with open(input_file, "r") as f:
        code = f.read()

    fixed = fix_reshape_pointer_casts(code)

    with open(output_file, "w") as f:
        f.write(fixed)

    print(f"Done. Written to {output_file}")