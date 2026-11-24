// fiveax patch: external layer-normal / road-tangent field (file format fiveax.field/1, "FXFD1").
// Header: 5 bytes magic, float origin[3], float spacing, uint32 dims[3]; then per cell (x fastest):
// float normal[3], float tangent[3], float wNormal, float wTangent. Positions are in AtomSlicer's
// model frame (the STL moved so its minimum corner is at the origin). Sampling is nearest-cell.
#pragma once
#include "mathutils.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

class ExternalField {
public:
    static ExternalField &get() { static ExternalField f; return f; }
    bool active() const { return loaded; }
    bool load(const char *path) {
        std::ifstream f(path, std::ios::binary);
        char magic[5];
        f.read(magic, 5);
        if (!f || std::string(magic, 5) != "FXFD1") { std::cout << "Fatal Error: bad field file " << path << std::endl; return false; }
        f.read(reinterpret_cast<char *>(origin), 12);
        f.read(reinterpret_cast<char *>(&spacing), 4);
        f.read(reinterpret_cast<char *>(dims), 12);
        size_t n = size_t(dims[0]) * dims[1] * dims[2];
        data.resize(n * 8);
        f.read(reinterpret_cast<char *>(data.data()), n * 8 * 4);
        loaded = bool(f);
        std::cout << "External field: " << dims[0] << "x" << dims[1] << "x" << dims[2] << " cells of " << spacing << " mm" << std::endl;
        return loaded;
    }
    // Layer normal at p, or `fallback` where the field has no preference.
    Vec3 normal(const Vec3 &p, const Vec3 &fallback) const {
        const float *c = cell(p);
        if (!c || c[6] <= 0.0f) return fallback;
        return Vec3(c[0], c[1], c[2]).normalize();
    }
    // In-layer road direction as AtomSlicer's printing angle (azimuth in XY), or `fallback`.
    float angle(const Vec3 &p, float fallback) const {
        const float *c = cell(p);
        if (!c || c[7] <= 0.0f) return fallback;
        if (fabsf(c[3]) + fabsf(c[4]) < 1e-6f) return fallback;
        return atan2f(c[4], c[3]);
    }
private:
    const float *cell(const Vec3 &p) const {
        int ix = int(floorf((p.x - origin[0]) / spacing)), iy = int(floorf((p.y - origin[1]) / spacing)), iz = int(floorf((p.z - origin[2]) / spacing));
        ix = ix < 0 ? 0 : (ix >= int(dims[0]) ? int(dims[0]) - 1 : ix);
        iy = iy < 0 ? 0 : (iy >= int(dims[1]) ? int(dims[1]) - 1 : iy);
        iz = iz < 0 ? 0 : (iz >= int(dims[2]) ? int(dims[2]) - 1 : iz);
        return &data[(size_t(iz) * dims[1] * dims[0] + size_t(iy) * dims[0] + ix) * 8];
    }
    bool loaded = false;
    float origin[3] = {0, 0, 0}, spacing = 1;
    uint32_t dims[3] = {0, 0, 0};
    std::vector<float> data;
};
