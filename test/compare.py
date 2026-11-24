#!/usr/bin/env python3
"""Compare two AtomSlicer raw waypoint files (.fxwp) within tolerances.

Usage: compare.py ref.fxwp new.fxwp [--logs ref.log new.log] [--json]

File format: b'FXWP1', uint64 count, float32 layer height, then per waypoint
float32 x y z nx ny nz + uint8 deposition.
A segment i-1 -> i deposits when waypoint i has deposition = 1.
Deposit volume = sum(segment length) * width * height, width = 2h, height = h.
The file has no layer marker, so the layer count comes from the slicer log
("Layers: N") when --logs is given; deposit runs (contiguous deposit
segments) are reported from the file as well.
Exit code 0 = PASS, 1 = FAIL (strict tolerances).

--floor F.fxwp: also compare ref against F, a CPU build of the same source with only the
floating-point contraction changed (-ffp-contract=off). That is the pipeline's own float
noise floor: ulp-level field changes flip discrete choices (atom picks, path routing) and
move whole path pieces by up to a bead width. FLOOR verdict = new vs ref is no worse than
FLOOR_FACTOR x (floor vs ref) on position p99/max, and strict on volume, axis and layers (axis is judged
against the floor too when the floor itself exceeds the strict axis limit: fragmented partitions).

Coverage: fraction of resampled deposit points of one plan lying within half a bead width
(= layer height) of the other plan's deposit points, both ways (min reported).
"""
import argparse, json, re, struct, sys
import numpy as np
from scipy.spatial import cKDTree

TOL = dict(pos_p99=0.02, pos_max=0.05, axis_p99_deg=1.0, vol_rel=0.005, layer_rel=0.01)
STEP = 0.1  # mm resampling step
FLOOR_FACTOR = 2.0


def load(path):
    with open(path, 'rb') as f:
        buf = f.read()
    assert buf[:5] == b'FXWP1', path + ': bad magic'
    n, = struct.unpack_from('<Q', buf, 5)
    lh, = struct.unpack_from('<f', buf, 13)
    rec = np.dtype([('p', '<f4', 3), ('n', '<f4', 3), ('d', 'u1')])
    a = np.frombuffer(buf, dtype=rec, count=n, offset=17)
    return lh, a['p'].astype(np.float64), a['n'].astype(np.float64), a['d'].astype(bool)


def deposit_samples(P, N, D):
    """Resample every deposit segment to STEP spacing; return points, axes, total length, runs."""
    if len(P) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3)), 0.0, 0
    seg = D[1:]
    a, b = P[:-1][seg], P[1:][seg]
    na, nb = N[:-1][seg], N[1:][seg]
    L = np.linalg.norm(b - a, axis=1)
    total = float(L.sum())
    runs = int(np.sum(seg & ~np.concatenate([[False], seg[:-1]])))
    k = np.maximum(1, np.ceil(L / STEP).astype(int))
    idx = np.repeat(np.arange(len(L)), k)
    start = np.concatenate([[0], np.cumsum(k)[:-1]])
    t = (np.arange(k.sum()) - np.repeat(start, k)) / np.repeat(k, k)
    pts = a[idx] + (b[idx] - a[idx]) * t[:, None]
    ax = na[idx] + (nb[idx] - na[idx]) * t[:, None]
    # add the final endpoint of each segment so short runs are represented
    pts = np.vstack([pts, b])
    ax = np.vstack([ax, nb])
    ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-12)
    return pts, ax, total, runs


def layers_from_log(path):
    if not path:
        return None
    m = re.findall(r'^Layers:\s*(\d+)', open(path, errors='replace').read(), re.M)
    return int(m[-1]) if m else None


def nn(src_p, src_a, dst_p, dst_a):
    tree = cKDTree(dst_p)
    d, j = tree.query(src_p, workers=4)
    cosang = np.abs(np.sum(src_a * dst_a[j], axis=1))  # axis sign does not matter for a nozzle
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return d, ang


def geom(s1, a1, s2, a2, lh):
    o = {}
    if len(s1) and len(s2):
        d12, g12 = nn(s1, a1, s2, a2)
        d21, g21 = nn(s2, a2, s1, a1)
        d = np.concatenate([d12, d21]); g = np.concatenate([g12, g21])
        o['pos_p99'] = float(np.percentile(d, 99)); o['pos_max'] = float(d.max())
        o['pos_mean'] = float(d.mean())
        o['axis_p99_deg'] = float(np.percentile(g, 99)); o['axis_max_deg'] = float(g.max())
        o['coverage'] = float(min((d12 <= lh).mean(), (d21 <= lh).mean()))
    else:
        o['pos_p99'] = o['pos_max'] = o['axis_p99_deg'] = o['axis_max_deg'] = float('inf'); o['coverage'] = 0.0
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref'); ap.add_argument('new')
    ap.add_argument('--logs', nargs=2, default=[None, None])
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--floor', default=None)
    args = ap.parse_args()

    lh1, P1, N1, D1 = load(args.ref)
    lh2, P2, N2, D2 = load(args.new)
    s1, a1, L1, r1 = deposit_samples(P1, N1, D1)
    s2, a2, L2, r2 = deposit_samples(P2, N2, D2)
    v1 = L1 * 2 * lh1 * lh1
    v2 = L2 * 2 * lh2 * lh2
    out = dict(waypoints=[len(P1), len(P2)], deposit_runs=[r1, r2],
               layer_height=[lh1, lh2], volume_mm3=[v1, v2])
    out['volume_rel_diff'] = abs(v2 - v1) / v1 if v1 else float('inf')
    lay = [layers_from_log(args.logs[0]), layers_from_log(args.logs[1])]
    out['layers'] = lay

    out.update(geom(s1, a1, s2, a2, lh1))
    if args.floor:
        lh3, P3, N3, D3 = load(args.floor)
        s3, a3, L3, r3 = deposit_samples(P3, N3, D3)
        fl = geom(s1, a1, s3, a3, lh1)
        fl['volume_rel_diff'] = abs(L3 - L1) / L1 if L1 else float('inf')
        out['floor'] = fl

    fails = []
    if lh1 != lh2: fails.append('layer height differs')
    if out['pos_p99'] > TOL['pos_p99']: fails.append('pos p99 %.4f > %.3f' % (out['pos_p99'], TOL['pos_p99']))
    if out['pos_max'] > TOL['pos_max']: fails.append('pos max %.4f > %.3f' % (out['pos_max'], TOL['pos_max']))
    if out['axis_p99_deg'] > TOL['axis_p99_deg']: fails.append('axis p99 %.3f deg > %.1f' % (out['axis_p99_deg'], TOL['axis_p99_deg']))
    if out['volume_rel_diff'] > TOL['vol_rel']: fails.append('volume diff %.3f%% > 0.5%%' % (100 * out['volume_rel_diff']))
    if None not in lay and lay[0] != lay[1] and abs(lay[1] - lay[0]) / max(lay[0], 1) > TOL['layer_rel']:
        fails.append('layers %d vs %d' % tuple(lay))
    out['identical_bytes'] = bool(len(P1) == len(P2) and np.array_equal(P1, P2) and np.array_equal(N1, N2) and np.array_equal(D1, D2))
    out['result'] = 'PASS' if not fails else 'FAIL'
    out['fails'] = fails
    if 'floor' in out:
        fl = out['floor']
        other = [f for f in fails if not f.startswith('pos ')]
        # Fragmented partitions (thousands of layers) let rounding flip which piece prints at which nozzle angle:
        # there the CPU-vs-CPU float floor itself breaks the strict axis limit, so judge axis against the floor too
        # (only when the floor already exceeds the strict limit; otherwise the axis check stays strict).
        if fl['axis_p99_deg'] > TOL['axis_p99_deg'] and out['axis_p99_deg'] <= FLOOR_FACTOR * fl['axis_p99_deg']:
            other = [f for f in other if not f.startswith('axis ')]
        ok = (out['pos_p99'] <= max(TOL['pos_p99'], FLOOR_FACTOR * fl['pos_p99']) and
              out['pos_max'] <= max(TOL['pos_max'], FLOOR_FACTOR * fl['pos_max']) and not other)
        out['floor_result'] = 'PASS' if ok else 'FAIL'

    if args.json:
        print(json.dumps(out))
    else:
        print('waypoints      ref %d  new %d' % tuple(out['waypoints']))
        print('deposit runs   ref %d  new %d' % (r1, r2))
        print('layers (log)   ref %s  new %s' % tuple(lay))
        print('volume mm3     ref %.2f  new %.2f  diff %.4f%%' % (v1, v2, 100 * out['volume_rel_diff']))
        print('pos NN mm      p99 %.5f  max %.5f' % (out['pos_p99'], out['pos_max']))
        print('axis deg       p99 %.4f  max %.4f' % (out['axis_p99_deg'], out.get('axis_max_deg', float('nan'))))
        print('coverage       %.5f (points within one layer height of the other plan)' % out['coverage'])
        print('identical      %s' % out['identical_bytes'])
        if 'floor' in out:
            fl = out['floor']
            print('float floor    pos p99 %.5f max %.5f  axis p99 %.4f  coverage %.5f  vol diff %.4f%%' % (
                fl['pos_p99'], fl['pos_max'], fl['axis_p99_deg'], fl['coverage'], 100 * fl['volume_rel_diff']))
            print('FLOOR RESULT %s (new vs ref within %.0fx the float floor; volume/axis/layers strict)' % (out['floor_result'], FLOOR_FACTOR))
        print('RESULT %s %s' % (out['result'], '; '.join(fails)))
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
