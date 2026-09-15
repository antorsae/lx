#!/usr/bin/env python3
"""Verify an LX521 print pack using only Python's standard library."""
import hashlib
import json
from pathlib import PurePosixPath
import sys
import zipfile


def verify(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive members')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError(f'unsafe archive member: {name}')
        expected = json.loads(z.read('SHA256SUMS.json'))
        if set(names) != set(expected) | {'SHA256SUMS.json', 'START_HERE.txt', 'verify_package.py'}:
            raise ValueError('archive inventory mismatch')
        for name, digest in expected.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError(f'hash mismatch: {name}')
    return len(expected)


if __name__ == '__main__':
    print(f'Verified {verify(sys.argv[1])} files')
