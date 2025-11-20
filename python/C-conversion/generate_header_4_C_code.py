import argparse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inpath", required=True)
    ap.add_argument("--outpath", required=True)
    args = ap.parse_args()

    entry_def = ""
    with open(args.inpath, "r") as f:
        for line in f:
            if line[:11] == "void entry(":
                entry_def = line[:-2] + ";"
                break
    
    header = """
#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>

%s

#endif /* NETWORK_H */
    
""" % entry_def
    
    with open(args.outpath, "w") as f:
        f.write(header)


    header_name = args.outpath.split("/")[-1]
    line_to_insert = '#include "%s"' % header_name
    with open(args.inpath, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(line_to_insert + content)