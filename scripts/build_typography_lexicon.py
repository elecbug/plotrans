"""Build the bundled word inventory from a pinned, separately downloaded CMUdict.

Only spellings are retained. Pronunciations and upstream Python code are not
used. Keep the accompanying CMUdict license and provenance JSON with the asset.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dictionary',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    words = set()
    for line in args.dictionary.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith(';;;'):
            continue
        word = re.sub(r'\(\d+\)$','',line.split()[0]).casefold()
        if re.fullmatch('[a-z]+(?:-[a-z]+)*',word):
            words.add(word)
    payload = (json.dumps(sorted(words),separators=(',',':'))+'\n').encode()
    args.output.write_bytes(gzip.compress(payload,mtime=0))
    print(json.dumps({'words':len(words),'bytes':args.output.stat().st_size,
                      'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
