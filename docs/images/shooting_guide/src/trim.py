"""PNG 오른쪽·아래의 흰 여백을 잘라내고 일정한 여백만 남긴다."""
import sys
from PIL import Image, ImageChops
p, pad = sys.argv[1], int(sys.argv[2])
im = Image.open(p).convert("RGB")
bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
im.crop((0, 0, min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))).save(p, optimize=True)
print(p.rsplit("/", 1)[-1], im.crop((0, 0, min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))).size)
