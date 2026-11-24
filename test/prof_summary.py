#!/usr/bin/env python3
"""Summarise an xctrace time-profile XML export: top self (leaf) and inclusive functions.
Usage: prof_summary.py export.xml [topN]"""
import sys, re, collections
import xml.etree.ElementTree as ET
path = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
only_main = '--main' in sys.argv
frames = {}; bts = {}
selfc = collections.Counter(); incl = collections.Counter(); total = 0
def short(n):
    n = re.sub(r'\(.*', '', n)
    n = re.sub(r'<[^<>]*(<[^<>]*>[^<>]*)*>', '<>', n)
    return n[:110]
for ev, el in ET.iterparse(path, events=('end',)):
    if el.tag == 'frame':
        if 'name' in el.attrib: frames[el.get('id')] = el.get('name')
    elif el.tag in ('backtrace', 'tagged-backtrace'):
        if el.get('ref'):
            pass
        else:
            fl = []
            for f in el.iter('frame'):
                fid = f.get('ref') or f.get('id')
                fl.append(frames.get(fid, '?'))
            bts[el.get('id')] = fl
    elif el.tag == 'row':
        bt = el.find('tagged-backtrace')
        if bt is None: bt = el.find('backtrace')
        if bt is None: el.clear(); continue
        fl = bts.get(bt.get('ref') or bt.get('id'), [])
        if not fl: el.clear(); continue
        if only_main and any('ThreadPool' in x for x in fl): el.clear(); continue
        total += 1
        selfc[short(fl[0])] += 1
        for n in set(short(x) for x in fl): incl[n] += 1
        el.clear()
print('samples', total)
print('--- self (leaf) ---')
for n, c in selfc.most_common(top): print('%5.1f%%  %s' % (100.0*c/total, n))
print('--- inclusive ---')
for n, c in incl.most_common(top): print('%5.1f%%  %s' % (100.0*c/total, n))
