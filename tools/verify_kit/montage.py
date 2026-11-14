"""One PNG per class with several candidate instances, 8 frames each at equal
steps over the annotated span (first = start, last = end), for a quick check of the annotation."""
import os, subprocess
from PIL import Image, ImageDraw, ImageFont
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def frames_of(video, s, e, nf, fw, tmp, tag):
    out = []
    for i in range(nf):
        t = s + i * (e - s) / (nf - 1); fp = os.path.join(tmp, f'{tag}_{i}.png')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{t:.3f}', '-i', video, '-frames:v', '1', '-vf', f'scale={fw}:-2', fp], check=True)
        out.append(Image.open(fp).convert('RGB'))
    return out


def montage(rows, title, subtitle, out_png, fw):
    """rows: [(label, [frames])]"""
    font = ImageFont.truetype(FONT, 13); fontb = ImageFont.truetype(FONT_B, 14)
    nf = len(rows[0][1]); fh = rows[0][1][0].height
    W = nf * (fw + 4) + 8; H = 44 + len(rows) * (fh + 34)
    im = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(im)
    dr.text((6, 4), title, fill='black', font=fontb); dr.text((6, 22), subtitle, fill='gray', font=font)
    for j, (lab, frames) in enumerate(rows):
        y = 44 + j * (fh + 34); dr.text((6, y), lab, fill='black', font=fontb)
        for i, f in enumerate(frames): im.paste(f, (4 + i * (fw + 4), y + 18))
    im.save(out_png)
