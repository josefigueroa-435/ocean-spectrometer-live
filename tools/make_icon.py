"""Draws ocean_live.ico: a rainbow spectrum with an emission peak."""
import colorsys
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=44, fill=(17, 30, 56, 255))
x0, x1, base = 34, S - 34, S - 52
for x in range(x0, x1):
    f = (x - x0) / (x1 - x0)
    r, g, b = colorsys.hsv_to_rgb(0.78 * (1 - f), 0.85, 1.0)
    d.line([(x, base), (x, base + 16)], fill=(int(r * 255), int(g * 255), int(b * 255), 255))
pts = []
for x in range(x0, x1):
    f = (x - x0) / (x1 - x0)
    y = 0.25 * (1 - (2 * f - 1) ** 2) + 0.75 * 2.718 ** (-((f - 0.58) / 0.05) ** 2)
    pts.append((x, base - 12 - y * 150))
d.line(pts, fill=(255, 255, 255, 255), width=9, joint="curve")
img.save(os.path.join(HERE, "ocean_live.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote ocean_live.ico")
