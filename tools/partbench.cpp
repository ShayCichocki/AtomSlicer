// partbench: run Partition alone on a dump written by ATOM_PART_DUMP and write the layers.
// Usage: partbench dump.fxpd out.layers [cone_deg] [up]
// out.layers: uint64 layerCount, then per layer uint64 size + size uint64 atom indices
// (exact order as Partition::getLayers()). Prints time and an FNV-1a hash of the file.
#include "partition.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: partbench dump.fxpd out.layers\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    char magic[4]; uint64_t n; float lh, cone; uint8_t up;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FXPD", 4)) { fprintf(stderr, "bad dump\n"); return 2; }
    fread(&n, 8, 1, f); fread(&lh, 4, 1, f); fread(&cone, 4, 1, f); fread(&up, 1, 1, f);
    std::vector<Vec3> points(n), normals(n);
    fread(points.data(), sizeof(Vec3), n, f);
    fread(normals.data(), sizeof(Vec3), n, f);
    fclose(f);
    // optional overrides to exercise other code paths: partbench dump out [cone_deg] [up 0/1]
    if (argc > 3) cone = float(atof(argv[3]));
    if (argc > 4) up = uint8_t(atoi(argv[4]));
    printf("atoms %llu layerHeight %g cone %g up %d\n", (unsigned long long)n, lh, cone, up);
    auto t0 = std::chrono::steady_clock::now();
    Partition partition(points, normals, lh, cone, up);
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto &layers = partition.getLayers();
    FILE *o = fopen(argv[2], "wb");
    uint64_t h = 1469598103934665603ull;
    auto put = [&](uint64_t v) { fwrite(&v, 8, 1, o); for (int b = 0; b < 8; ++b) { h ^= (v >> (8*b)) & 0xff; h *= 1099511628211ull; } };
    put(layers.size());
    for (auto &l : layers) { put(l.size()); for (auto i : l) put(i); }
    fclose(o);
    printf("Layers: %zu\nPartition time: %.3f s\nhash %016llx\n", layers.size(), t, (unsigned long long)h);
    return 0;
}
