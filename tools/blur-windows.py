"""Not part of the website - kept so the blur can be reproduced or adjusted.

Usage:  python3 tools/blur-windows.py <source.mp4> <output.mp4|.webm> day|night

Output size, frame rate and codec come from the constants below. The frame rate
matches the source footage (~14.8fps) - encoding higher just invents duplicate
frames and inflates the file. Masks are in percentages, so they follow the
output size automatically.

Blur the exterior views (windows, glass doors, balcony) in the walk-through clips.

The camera pans continuously, so each window is described as a *track*: a list of
(second, rect) keyframes mapped by hand from the footage, linearly interpolated
between them at frame rate. Rects are percentages of frame size and are drawn
generously, then feathered, so a slightly-off rect still covers the glass.
"""
import subprocess, sys, numpy as np, cv2

SRC_W, SRC_H = 576, 1024      # what the phone recorded
W, H, FPS = 480, 854, 15      # what gets published
FEATHER = round(22 * W / SRC_W)   # px, softens the rect edge
# Keyframes below were read off 1-frame-per-second contact sheets. On these VFR
# clips ffmpeg's fps=1 filter lands 0.45s later than the wall clock, so shift the
# lookup by that much rather than re-reading every rect.
ANNOT_OFFSET = 0.45
PAD     = 2.0         # % extra around every rect

# (x0, y0, x1, y1) in % of frame. Each list is one window followed across a pan.
TRACKS = {
'day': [
  [(-0.7,(20,0,100,78)), (0.0,(18,0,100,76)), (1.0,(16,0,100,74)), (2.0,(40,0,100,74)), (2.9,(60,0,100,72))],
  [(3.6,(0,10,30,72)),  (4.0,(0,15,28,70)),  (4.5,(0,15,25,68))],
  [(11.6,(0,0,52,62)),  (12.0,(0,0,50,60)),  (12.5,(0,0,45,58))],
  [(13.0,(62,0,100,46)),(14.0,(62,2,100,50)),(15.0,(40,5,80,48)),(16.0,(42,8,82,50)),
   (17.0,(46,5,86,50)), (18.0,(66,0,100,48)),(18.6,(76,0,100,48))],
  [(23.6,(10,0,100,50)),(24.0,(10,0,100,50)),(25.0,(0,10,80,58)), (26.0,(8,12,88,60)),
   (27.0,(4,10,86,60)), (28.0,(12,10,98,66)),(29.0,(14,12,100,70)),
   (30.0,(0,0,100,85)), (31.0,(0,0,100,85)), (32.0,(0,0,100,85)),
   (33.0,(55,0,100,60)),(33.8,(70,0,100,55))],
  [(34.6,(8,16,52,80)), (35.0,(10,18,50,78)),(35.5,(12,20,48,76))],
  [(36.6,(0,30,26,70)), (37.0,(0,32,25,68)), (38.0,(0,30,25,66)), (39.0,(0,26,24,62)),
   (40.0,(4,8,30,52)),  (41.0,(4,8,30,52)),  (41.6,(2,8,28,50))],
  [(51.6,(50,0,88,40)), (52.0,(50,0,88,40)), (53.0,(54,0,100,46)),(54.0,(50,0,93,50)),
   (55.0,(34,8,76,56)), (56.0,(32,5,70,56)), (56.6,(34,8,72,58))],
  [(56.8,(38,12,66,58)),(57.0,(38,12,66,58)),(58.0,(34,10,62,56)),(59.0,(32,10,60,56)),
   (60.3,(32,10,60,56))],
],
'night': [
  [(-0.7,(45,0,85,56)), (0.0,(45,0,85,56)),  (2.0,(45,0,85,56)),  (4.0,(42,0,82,56)), (5.0,(28,0,68,56)),
   (6.0,(30,0,70,56)),  (7.0,(32,0,72,56)),  (8.0,(38,0,78,60)), (9.0,(40,0,80,60)),
   (9.6,(42,0,82,60))],
  [(22.6,(48,0,100,52)),(23.0,(48,0,100,52)),(24.0,(50,0,100,52)),(25.0,(48,0,100,50)),
   (26.0,(40,0,100,50)),(27.0,(18,0,100,52)),(28.0,(16,0,100,52)),(29.0,(14,0,100,52)),
   (30.0,(0,0,100,44)), (31.0,(0,0,100,44)), (32.0,(0,0,100,44)),
   (33.0,(20,0,100,56)),(34.0,(24,0,100,58)),(35.0,(20,0,100,55)),(36.0,(18,0,100,55)),
   (37.0,(16,0,100,55)),(38.0,(14,0,100,55)),(39.0,(12,0,100,55)),
   (40.0,(0,0,100,52)), (41.0,(0,0,100,52)), (42.0,(0,0,100,52)), (43.0,(0,0,100,52)),
   (44.0,(0,0,100,52)), (45.0,(0,0,100,52)),
   (46.0,(42,0,95,65)), (47.0,(42,0,95,65)),
   (48.0,(0,0,78,80)),  (49.0,(0,0,78,80)),
   (50.0,(0,0,90,55)),  (51.0,(0,0,100,58)), (52.0,(0,0,100,55)), (53.0,(0,0,100,55)),
   (54.0,(0,0,100,55)), (55.0,(0,0,100,65)), (55.8,(0,0,100,64)), (56.2,(0,0,100,62))],
],
}


def rect_at(track, t):
    """Interpolated rect for time t, or None when the track isn't running."""
    if t < track[0][0] or t > track[-1][0]:
        return None
    for (t0, r0), (t1, r1) in zip(track, track[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return [a + (b - a) * f for a, b in zip(r0, r1)]
    return track[-1][1]


def mask_at(tracks, t):
    m = np.zeros((H, W), np.float32)
    for tr in tracks:
        r = rect_at(tr, t)
        if r is None:
            continue
        x0 = int(max(0, (r[0] - PAD)) * W / 100); x1 = int(min(100, (r[2] + PAD)) * W / 100)
        y0 = int(max(0, (r[1] - PAD)) * H / 100); y1 = int(min(100, (r[3] + PAD)) * H / 100)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0
    if m.any():
        m = cv2.GaussianBlur(m, (0, 0), FEATHER)
        m = np.clip(m * 1.35, 0, 1)
    return m


def obscure(frame):
    small = cv2.resize(frame, (max(1, W // 24), max(1, H // 24)), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(big, (0, 0), 10)


def frames(path):
    # These clips are variable frame rate (WhatsApp re-encodes them that way), so
    # decode through fps=FPS to force constant rate -- otherwise frame index does
    # not map to wall-clock time and every mask lands in the wrong place.
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path,
                          '-vf', f'fps={FPS},scale={W}:{H}',
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                         stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
    p.stdout.close(); p.wait()


def render(frame, tracks, t):
    m = mask_at(tracks, t)
    if not m.any():
        return frame
    a = cv2.merge([m, m, m])
    return (frame * (1 - a) + obscure(frame) * a).astype(np.uint8)


if __name__ == '__main__':
    src, dst, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    tracks = TRACKS[kind]
    if dst.endswith('.webm'):        # smaller, for browsers that take VP9
        # pix_fmt is not optional here: fed raw BGR frames, ffmpeg otherwise
        # picks 4:4:4 chroma for VP9 and the file comes out ~3x larger.
        codec = ['-c:v', 'libvpx-vp9', '-crf', '36', '-b:v', '0', '-row-mt', '1',
                 '-speed', '2', '-pix_fmt', 'yuv420p',
                 '-c:a', 'libopus', '-b:a', '48k', '-ac', '1']
    else:                            # H.264 fallback, plays everywhere
        codec = ['-c:v', 'libx264', '-profile:v', 'high', '-level', '4.0', '-crf', '33',
                 '-preset', 'slow', '-pix_fmt', 'yuv420p',
                 '-c:a', 'aac', '-b:a', '64k', '-ac', '1', '-movflags', '+faststart']
    enc = subprocess.Popen(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
         '-i', src, '-map', '0:v', '-map', '1:a'] + codec + [dst], stdin=subprocess.PIPE)
    for i, f in enumerate(frames(src)):
        enc.stdin.write(render(f, tracks, i / FPS - ANNOT_OFFSET).tobytes())
    enc.stdin.close(); enc.wait()
    print('wrote', dst)
