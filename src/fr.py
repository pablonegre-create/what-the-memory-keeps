"""Thumbnails for the UCF-Crime example videos: fetch only those videos from the official archive
(HTTP range requests), save 10 evenly spaced frames plus the frames at the CAM peaks, delete the
videos. Needs internet access. Output: out/fig/frames/*.jpg + frames.json."""
import io, os, json, time, zipfile, tempfile
import numpy as np
import cv2
import requests

URL = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
FIG = "out/fig"
os.umask(0o077); os.makedirs(f"{FIG}/frames", exist_ok=True)


class HttpFile(io.RawIOBase):
    def __init__(self, url, block=8 << 20):
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        self.sess = requests.Session()
        self.size = int(self.sess.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, bi):
        if bi not in self.cache:
            a = bi * self.block; b = min(self.size, a + self.block) - 1
            for attempt in range(10):
                try:
                    r = self.sess.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
                    if r.status_code == 206 and len(r.content) == b - a + 1:
                        break
                except requests.RequestException:
                    pass
                time.sleep(10)
            else:
                raise IOError("range failed")
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[bi] = r.content
        return self.cache[bi]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos)); out = bytearray()
        while n > 0:
            bi, off = divmod(self.pos, self.block)
            chunk = self._block(bi)[off:off + n]; out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


meta = [e for e in json.load(open(f"{FIG}/examples.json")) if e["tag"].startswith("ucf")]
arr = np.load(f"{FIG}/examples.npz")
z = zipfile.ZipFile(HttpFile(URL))
members = {os.path.splitext(os.path.basename(i.filename))[0]: i for i in z.infolist() if not i.is_dir()}
out = {}
for k, e in enumerate(meta):
    info = members[e["name"]]
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=FIG, delete=False) as tmp:
        with z.open(info) as src:
            tmp.write(src.read())
        path = tmp.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = list(np.linspace(0.5, nfr / fps - 0.5, 10))
    for init in ("const200", "standard"):
        cam = arr[f"{e['key']}|{init}|cam"]
        times.append((int(np.argmax(cam)) + 0.5) * e["step_s"])
    frames = []
    for j, ts in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        fn = f"{FIG}/frames/ex{k}_{j:02d}.jpg"
        cv2.imwrite(fn, fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append({"file": os.path.basename(fn), "t": float(ts), "peak": ["", "const200", "standard"][max(0, j - 9)] if j >= 10 else ""})
    cap.release(); os.remove(path)
    out[e["key"]] = {"fps": fps, "n_frames": nfr, "frames": frames}
    print(e["key"], len(frames), "frames", flush=True)
json.dump(out, open(f"{FIG}/frames.json", "w"), indent=1)
print("done")
