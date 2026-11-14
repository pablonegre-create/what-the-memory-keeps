"""I3D (Inception, Kinetics-400 RGB) features for the staged videos: centre crop 224,
non-overlapping 16-frame snippets at native fps -> one 1024-d vector per snippet.
Each video is deleted as soon as its features are saved. Exits when the downloader has
finished and the staging folder is empty."""
import os, sys, time, glob
import numpy as np
import cv2
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

sys.path.insert(0, "ext")
from pytorch_i3d import InceptionI3d

STAGE, FEAT, CHUNK = "data/stage_ucf", "data/feat/ucf_i3d", 64   # 64 snippets per chunk
os.umask(0o077)
os.makedirs(FEAT, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)


def frames_iter(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        s = 256 / min(h, w)
        fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
        h, w = fr.shape[:2]
        y, x = (h - 224) // 2, (w - 224) // 2
        yield fr[y:y + 224, x:x + 224]
    cap.release()


class Videos(IterableDataset):
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        wi = get_worker_info()
        mine = self.paths[wi.id::wi.num_workers] if wi else self.paths
        for p in mine:
            buf, idx = [], 0
            for fr in frames_iter(p):
                buf.append(fr)
                if len(buf) == CHUNK * 16:
                    yield p, idx, np.stack(buf), False
                    buf, idx = [], idx + 1
            n = len(buf) // 16 * 16
            yield p, idx, (np.stack(buf[:n]) if n else np.zeros((0, 224, 224, 3), np.uint8)), True


model = InceptionI3d(400, in_channels=3)
model.load_state_dict(torch.load("ext/rgb_imagenet.pt", weights_only=True))
model = model.cuda().eval()


@torch.no_grad()
def embed(frames):
    if len(frames) == 0:
        return np.zeros((0, 1024), np.float32)
    x = frames if torch.is_tensor(frames) else torch.from_numpy(frames)     # DataLoader may convert
    x = x.cuda().float().div_(127.5).sub_(1.0)                               # (N,224,224,3) in [-1,1]
    x = x.view(-1, 16, 224, 224, 3).permute(0, 4, 1, 2, 3).contiguous()      # (S,3,16,224,224)
    out = [model.extract_features(x[k:k + 32]).flatten(1) for k in range(0, len(x), 32)]
    return torch.cat(out).float().cpu().numpy()


done, failed, nframes = 0, [], 0
while True:
    ready = sorted(glob.glob(f"{STAGE}/*.mp4"))
    if not ready:
        if os.path.exists(f"{STAGE}/ALL_DONE") and not glob.glob(f"{STAGE}/*.part"):
            break
        time.sleep(20)
        continue
    acc = {}
    loader = DataLoader(Videos(ready), batch_size=None, num_workers=min(14, len(ready)),
                        prefetch_factor=2, persistent_workers=False)
    for path, idx, frames, last in loader:
        acc.setdefault(path, {})[idx] = embed(frames)
        nframes += len(frames)
        if last:
            parts = acc.pop(path)
            feats = np.concatenate([parts[i] for i in sorted(parts)]) if parts else np.zeros((0, 1024))
            stem = os.path.basename(path)[:-4]
            if len(feats) == 0:
                failed.append(stem)
            else:
                np.save(f"{FEAT}/{stem}.npy", feats.astype(np.float32))
                done += 1
            os.remove(path)
    log(f"{done} videos done, {nframes / 1e6:.2f} M frames, {nframes / (time.time() - t0):.0f} fps; failed {len(failed)}")

# consistency with the CLIP features (same 16-frame steps)
clip = {os.path.basename(f)[:-7]: f for f in glob.glob("data/feat/ucf_clip/UCFClipFeatures/*/*__0.npy")}
diffs = []
for f in glob.glob(f"{FEAT}/*.npy"):
    s = os.path.basename(f)[:-4]
    if s in clip:
        diffs.append(np.load(f, mmap_mode="r").shape[0] - np.load(clip[s], mmap_mode="r").shape[0])
diffs = np.array(diffs)
log(f"finished: {done} videos, failed {failed}; steps I3D - CLIP: "
    f"equal {np.mean(diffs == 0):.3f}, |diff|<=1 {np.mean(np.abs(diffs) <= 1):.3f}, "
    f"min {diffs.min() if len(diffs) else 0} max {diffs.max() if len(diffs) else 0}")
