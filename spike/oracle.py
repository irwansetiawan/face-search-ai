"""
Spike oracle — reference embeddings from the Python insightface package.

Sole purpose: give the TypeScript port something to diff against. A hand-rolled
SCRFD decode / alignment can be subtly wrong in ways that still produce plausible
cosine numbers, so we compare embeddings against the known-good implementation
rather than trusting the separation we happen to see.

Emits JSON on stdout: {file: {"n": <faces>, "embedding": [...512 floats]}} for the
largest face in each image.

  ./orcl/bin/python spike/oracle.py img1.jpg img2.jpg > oracle.json
"""
import json
import os
import sys

import numpy as np
from insightface.app import FaceAnalysis

# Point insightface at the model pack we already downloaded rather than fetching
# another 275 MB copy. It expects <root>/models/<name>/*.onnx.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

app = FaceAnalysis(
    name="buffalo_l",
    root=ROOT,
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))

out = {}
for path in sys.argv[1:]:
    import cv2

    img = cv2.imread(path)  # BGR, and cv2 does NOT apply EXIF rotation
    faces = app.get(img)
    if not faces:
        out[path] = {"n": 0, "embedding": None}
        continue
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = np.asarray(largest.normed_embedding, dtype=np.float64)
    out[path] = {
        "n": len(faces),
        "det_score": float(largest.det_score),
        "bbox": [float(v) for v in largest.bbox],
        "embedding": emb.tolist(),
    }
    print(
        f"{os.path.basename(path)}: {len(faces)} faces, "
        f"largest det={largest.det_score:.3f} "
        f"bbox={[round(float(v)) for v in largest.bbox]}",
        file=sys.stderr,
    )

json.dump(out, sys.stdout)
