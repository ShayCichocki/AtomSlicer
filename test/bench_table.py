#!/usr/bin/env python3
"""Build a markdown timing table from two slicer logs and a compare.py JSON result.

Usage: bench_table.py fixture ref.log new.log compare.json ref.time new.time
Prints markdown rows (no header) to stdout.
"""
import json, re, sys


def stages(path):
    txt = open(path, errors='replace').read().replace('\r', '\n')
    out, cur = [], None
    for line in txt.split('\n'):
        line = line.strip()
        m = re.match(r'^([A-Z][\w /-]*)\.\.\.$', line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r'^Time:\s*([\d.]+)\s*s', line)
        if m and cur:
            out.append((cur, float(m.group(1))))
            cur = None
        m = re.match(r'^Total time:\s*([\d.]+)\s*s', line)
        if m:
            out.append(('TOTAL', float(m.group(1))))
    return out


def peak_mb(path):
    try:
        m = re.search(r'(\d+)\s+maximum resident set size', open(path).read())
        return int(m.group(1)) / 1e6 if m else None
    except OSError:
        return None


fx, rl, nl, cj, rt, nt = sys.argv[1:7]
r, n = stages(rl), stages(nl)
c = json.load(open(cj))
rd = dict(r)
for name, tn in n:
    tr = rd.get(name)
    sp = ('%.2fx' % (tr / tn)) if tr and tn > 0 else '-'
    print('| %s | %s | %s | %.3f | %s |' % (fx, name, '%.3f' % tr if tr is not None else '-', tn, sp))
pr, pn = peak_mb(rt), peak_mb(nt)
if pr and pn:
    print('| %s | peak RSS (MB) | %.0f | %.0f | - |' % (fx, pr, pn))
print('| %s | **compare** | %s | pos p99 %.4f / max %.4f mm, axis p99 %.3f deg, vol diff %.4f %%, layers %s, identical %s | %s |' % (
    fx, c['result'], c['pos_p99'], c['pos_max'], c['axis_p99_deg'], 100 * c['volume_rel_diff'], c['layers'],
    c['identical_bytes'], '; '.join(c['fails']) or 'ok'))
if 'floor' in c:
    f = c['floor']
    print('| %s | **float floor** (CPU -ffp-contract=off vs ref) | - | pos p99 %.4f / max %.4f mm, axis p99 %.3f deg, vol diff %.4f %%, coverage %.4f | new coverage %.4f; FLOOR verdict **%s** |' % (
        fx, f['pos_p99'], f['pos_max'], f['axis_p99_deg'], 100 * f['volume_rel_diff'], f['coverage'], c['coverage'], c['floor_result']))
