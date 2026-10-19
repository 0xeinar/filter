"""Rasterize the canvas dump written by `dump_layers_for_visual_inspection`.

Pure stdlib: iced's `Geometry` is a triangle mesh with per-vertex linear RGBA,
so all this needs is a scanline fill and a PNG encoder.
"""

import json
import struct
import sys
import zlib

SCALE = 2  # supersample for a less aliased look


def lin_to_srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def write_png(path, width, height, pixels):
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride:(y + 1) * stride])

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def main(src, dst):
    doc = json.load(open(src))
    bx, by, bw, bh = doc["bounds"]
    w = int(bw * SCALE)
    h = int(bh * SCALE)
    # Editor background, matching the plugin's dark theme.
    bg = [0.102, 0.102, 0.118]
    px = list(bg) * (w * h)

    for layer in doc["layers"]:
        for tri in layer["triangles"]:
            pts = [(p[0] * SCALE, p[1] * SCALE) for p in tri]
            cols = [(p[2], p[3], p[4], p[5]) for p in tri]
            minx = max(0, int(min(p[0] for p in pts)))
            maxx = min(w - 1, int(max(p[0] for p in pts)) + 1)
            miny = max(0, int(min(p[1] for p in pts)))
            maxy = min(h - 1, int(max(p[1] for p in pts)) + 1)
            (x0, y0), (x1, y1), (x2, y2) = pts
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-9:
                continue
            for y in range(miny, maxy + 1):
                for x in range(minx, maxx + 1):
                    cx, cy = x + 0.5, y + 0.5
                    a = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
                    b = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
                    c = 1 - a - b
                    if a < 0 or b < 0 or c < 0:
                        continue
                    r = a * cols[0][0] + b * cols[1][0] + c * cols[2][0]
                    g = a * cols[0][1] + b * cols[1][1] + c * cols[2][1]
                    bl = a * cols[0][2] + b * cols[1][2] + c * cols[2][2]
                    al = a * cols[0][3] + b * cols[1][3] + c * cols[2][3]
                    if al <= 0:
                        continue
                    i = (y * w + x) * 3
                    for k, v in enumerate((r, g, bl)):
                        px[i + k] = px[i + k] * (1 - al) + v * al

    out = bytearray(w * h * 3)
    for i, v in enumerate(px):
        out[i] = int(round(lin_to_srgb(v) * 255))
    write_png(dst, w, h, out)
    print(f"wrote {dst} ({w}x{h}) from {len(doc['layers'])} layers")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
