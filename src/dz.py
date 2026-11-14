"""Fetch the needed videos from the remote archive with HTTP range requests, using several
parallel connections, in batches: complete + in-flight videos never exceed CAP bytes.
Needs internet access; the extracted features are written to data/feat/ucf_i3d."""
import io, os, sys, time, shutil, zipfile, glob, threading
from concurrent.futures import ThreadPoolExecutor
import requests

URL = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
STAGE, FEAT, CAP, WORKERS = "data/stage_ucf", "data/feat/ucf_i3d", 10e9, 6
os.umask(0o077)
os.makedirs(STAGE, exist_ok=True)


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched in 8 MB blocks."""

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
            a = bi * self.block
            b = min(self.size, a + self.block) - 1
            for attempt in range(20):
                try:
                    r = self.sess.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
                    if r.status_code == 206 and len(r.content) == b - a + 1:
                        break
                except requests.RequestException:
                    pass
                time.sleep(min(300, 10 * (attempt + 1)))
            else:
                raise IOError(f"range {a}-{b} failed")
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[bi] = r.content
        return self.cache[bi]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            bi, off = divmod(self.pos, self.block)
            chunk = self._block(bi)[off:off + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


want = {os.path.basename(f)[:-7] for f in glob.glob("data/feat/ucf_clip/UCFClipFeatures/*/*__0.npy")}
z0 = zipfile.ZipFile(HttpFile(URL))
members = {}
for info in z0.infolist():
    stem = os.path.splitext(os.path.basename(info.filename))[0]
    if stem in want and not info.is_dir():
        members.setdefault(stem, info)
log(f"matched {len(members)}/{len(want)} videos ({sum(i.file_size for i in members.values()) / 1e9:.1f} GB)")
if len(sys.argv) > 1 and sys.argv[1] == "list":
    sys.exit()

for f in glob.glob(f"{STAGE}/*.part"):       # leftovers of an interrupted run
    os.remove(f)
todo = [s for s in sorted(members)
        if not os.path.exists(f"{FEAT}/{s}.npy") and not os.path.exists(f"{STAGE}/{s}.mp4")]
lock, reserved, stats = threading.Lock(), [0], {"n": 0, "b": 0, "t0": time.time()}
tl = threading.local()


def complete_bytes():
    tot = 0
    for f in glob.glob(f"{STAGE}/*.mp4"):
        try:                       # the extractor may delete a file at any moment
            tot += os.path.getsize(f)
        except OSError:
            pass
    return tot


def fetch(stem):
    info = members[stem]
    if not hasattr(tl, "z"):
        tl.z = zipfile.ZipFile(HttpFile(URL))
    while True:                                  # reserve room under the cap
        with lock:
            if complete_bytes() + reserved[0] + info.file_size <= CAP or (reserved[0] == 0 and complete_bytes() == 0):
                reserved[0] += info.file_size
                break
        time.sleep(15)
    final, part = f"{STAGE}/{stem}.mp4", f"{STAGE}/{stem}.mp4.part"
    try:
        for attempt in range(5):
            try:
                with tl.z.open(tl.z.getinfo(info.filename)) as src, open(part, "wb") as dst:
                    shutil.copyfileobj(src, dst, 8 << 20)   # CRC checked by zipfile
                os.rename(part, final)
                break
            except Exception as e:
                log(f"  retry {stem}: {e!r}")
                time.sleep(30)
        else:
            log(f"FAILED {stem}")
    finally:
        with lock:
            reserved[0] -= info.file_size
            stats["n"] += 1; stats["b"] += info.file_size
            if stats["n"] % 50 == 0:
                el = time.time() - stats["t0"]
                log(f"{stats['n']}/{len(todo)} videos, {stats['b'] / 1e9:.1f} GB, {stats['b'] / el / 1e6:.1f} MB/s")


with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(fetch, todo))
open(f"{STAGE}/ALL_DONE", "w").close()
log("all videos fetched")
