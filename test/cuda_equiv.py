#!/usr/bin/env python3
"""Prove a GPU build of AtomSlicer gives the same plan as the CPU build, and time both.

Runs each fixture through the CPU build, optionally a float-floor CPU build (same source,
-ffp-contract=off) and the GPU build (CUDA, or the CUDA CPU emulation), then compares the
waypoint files with compare.py. Portable: Linux / WSL2 / macOS / Windows, Python 3.8+ with
numpy and scipy.

Usage:
  test/cuda_equiv.py --cpu build-cpu/slicer --gpu build-cuda/slicer [--floor build-floor/slicer]
                     [--exact] [--fixtures test/cube6.stl ...] [--args "-d 0.4 -h 0.5 -o 1 -i 0"]
                     [--out test/out-cuda] [--reuse]

--exact   also run the GPU build with ATOM_CUDA_EXACT=1 (cos/sin/atan2 kernels on the CPU).
--reuse   reuse CPU / floor outputs already in --out (same fixture and args).
The GPU log must contain the backend's "using <device>" line, otherwise the run is marked
GPU-NOT-USED (a silent CPU fallback would otherwise pass trivially).
Exit code 0 when every fixture passes strict or FLOOR (when --floor is given), else 1.
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIX = ['cube6.stl', 'cube20.stl', 'holed_block.stl', 'curved_bar.stl']


def stages(text):
    out, cur = [], None
    for line in text.replace('\r', '\n').split('\n'):
        line = line.strip()
        m = re.match(r'^([A-Z][\w /-]*)\.\.\.$', line)
        if m:
            cur = m.group(1); continue
        m = re.match(r'^Time:\s*([\d.]+)\s*s', line)
        if m and cur:
            out.append((cur, float(m.group(1)))); cur = None
        m = re.match(r'^Total time:\s*([\d.]+)\s*s', line)
        if m:
            out.append(('TOTAL', float(m.group(1))))
    return out


def run(binary, stl, args, wp, log, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [binary, stl] + args.split() + ['-W', wp]
    t0 = time.perf_counter()
    with open(log, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        rss = None
        if hasattr(os, 'wait4'):
            _, status, ru = os.wait4(p.pid, 0)
            rc = os.waitstatus_to_exitcode(status) if hasattr(os, 'waitstatus_to_exitcode') else (status >> 8)
            rss = ru.ru_maxrss / (1e6 if sys.platform == 'darwin' else 1e3)   # MB
            p.returncode = rc
        else:
            rc = p.wait()
    wall = time.perf_counter() - t0
    if rc != 0:
        sys.exit('%s failed on %s (exit %s), see %s' % (binary, stl, rc, log))
    return dict(wall=wall, rss_mb=rss, stages=stages(open(log, errors='replace').read()))


def compare(ref, new, rlog, nlog, floor):
    cmd = [sys.executable, os.path.join(HERE, 'compare.py'), ref, new, '--logs', rlog, nlog, '--json']
    if floor:
        cmd += ['--floor', floor]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not r.stdout.strip():
        sys.exit('compare.py failed: ' + r.stderr)
    return json.loads(r.stdout.strip().split('\n')[-1])


def backend_line(log):
    txt = open(log, errors='replace').read()
    m = re.search(r'^\[(cuda|cuda-emu|metal)\] using (.*)$', txt, re.M)
    fallbacks = [l for l in re.findall(r'^\[(?:cuda|cuda-emu)\] .*(?:using CPU|disabling).*$', txt, re.M)
                 if not (m and l == m.group(0))]
    return (m.group(0) if m else None), fallbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--floor')
    ap.add_argument('--exact', action='store_true')
    ap.add_argument('--fixtures', nargs='*')
    ap.add_argument('--args', default='-d 0.4 -h 0.5 -o 1 -i 0')
    ap.add_argument('--out', default=os.path.join(HERE, 'out-cuda'))
    ap.add_argument('--reuse', action='store_true')
    a = ap.parse_args()
    fixtures = a.fixtures or [os.path.join(HERE, f) for f in DEFAULT_FIX]
    os.makedirs(a.out, exist_ok=True)
    tag = re.sub(r'[^A-Za-z0-9]', '_', a.args)
    allok, rows, summary = True, [], []
    for stl in fixtures:
        fx = os.path.splitext(os.path.basename(stl))[0]
        base = os.path.join(a.out, '%s_%s' % (fx, tag))
        res = {'fixture': fx}
        variants = [('cpu', a.cpu, None)]
        if a.floor:
            variants.append(('floor', a.floor, None))
        variants.append(('gpu', a.gpu, None))
        if a.exact:
            variants.append(('gpu_exact', a.gpu, {'ATOM_CUDA_EXACT': '1'}))
        for name, binary, env in variants:
            wp, log = base + '.%s.fxwp' % name, base + '.%s.log' % name
            cached = a.reuse and name in ('cpu', 'floor') and os.path.exists(wp) and os.path.getsize(wp) > 0 and os.path.exists(log)
            if cached:
                res[name] = dict(wall=None, rss_mb=None, stages=stages(open(log, errors='replace').read()))
            else:
                print('running %-9s %s' % (name, fx), flush=True)
                res[name] = run(binary, stl, a.args, wp, log, env)
        for name in [v[0] for v in variants if v[0].startswith('gpu')]:
            line, fb = backend_line(base + '.%s.log' % name)
            c = compare(base + '.cpu.fxwp', base + '.%s.fxwp' % name, base + '.cpu.log', base + '.%s.log' % name,
                        base + '.floor.fxwp' if a.floor else None)
            res[name]['compare'] = c
            res[name]['backend'] = line
            res[name]['fallback_msgs'] = fb
            verdict = c['result']
            if c['result'] != 'PASS' and c.get('floor_result') == 'PASS':
                verdict = 'FLOOR-PASS'
            if not line:
                verdict = 'GPU-NOT-USED'
            res[name]['verdict'] = verdict
            if verdict not in ('PASS', 'FLOOR-PASS'):
                allok = False
            rows.append((fx, name, res, c, verdict, line, fb))
        summary.append(res)

    print('\n| fixture | run | verdict | identical | pos p99 / max mm | axis p99 deg | vol diff % | coverage | layers | floor pos p99 / max | backend |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for fx, name, res, c, verdict, line, fb in rows:
        fl = c.get('floor')
        print('| %s | %s | %s | %s | %.4f / %.4f | %.3f | %.4f | %.4f | %s | %s | %s%s |' % (
            fx, name, verdict, c['identical_bytes'], c['pos_p99'], c['pos_max'], c['axis_p99_deg'],
            100 * c['volume_rel_diff'], c['coverage'], c['layers'],
            ('%.4f / %.4f' % (fl['pos_p99'], fl['pos_max'])) if fl else '-', line or 'none',
            ('; fallbacks: ' + ' / '.join(fb)) if fb else ''))
    print('\n| fixture | stage | cpu s | gpu s | speedup |')
    print('|---|---|---|---|---|')
    for res in summary:
        cs = dict(res['cpu']['stages'])
        for st, tg in res['gpu']['stages']:
            tc = cs.get(st)
            sp = ('%.2fx' % (tc / tg)) if tc and tg > 0 else '-'
            print('| %s | %s | %s | %.3f | %s |' % (res['fixture'], st, '%.3f' % tc if tc is not None else '-', tg, sp))
        w = (res['cpu']['wall'], res['gpu']['wall'])
        if None not in w:
            print('| %s | wall (process) | %.2f | %.2f | %.2fx |' % (res['fixture'], w[0], w[1], w[0] / w[1]))
        r = (res['cpu']['rss_mb'], res['gpu']['rss_mb'])
        if None not in r:
            print('| %s | peak RSS MB | %.0f | %.0f | - |' % (res['fixture'], r[0], r[1]))
    with open(os.path.join(a.out, 'summary_%s.json' % tag), 'w') as f:
        json.dump(summary, f, indent=1)
    print('\nALL %s' % ('OK' if allok else 'NOT OK'))
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
