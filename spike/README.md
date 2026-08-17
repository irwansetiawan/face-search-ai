# Spike: replacing AWS Rekognition CompareFaces with local InsightFace

Throwaway code kept for the numbers. Nothing here is wired into the app.

## Conclusion

A pure-Node InsightFace pipeline works, is fast on Apple Silicon, and produces
embeddings that agree with the reference Python implementation closely enough that
no match/no-match decision changes. **No Python is needed at runtime.**

That is not the same as "the oracle was unnecessary." `oracle.py` found a defect that
end-to-end results were hiding (see *Two things that cost real accuracy*, below) and
took agreement from 0.94 to 0.992. **Keep it, and re-run `compare-oracle.mjs` after any
change to the pipeline.** It is the only thing standing between a subtle regression and
silently worse matching. Don't delete it because runtime doesn't need Python.

## What's here

| file | purpose |
|---|---|
| `insightface.mjs` | the pipeline: letterbox → SCRFD → decode/NMS → align → ArcFace → L2 |
| `match.mjs` | CLI: `--probe a.jpg --target b.jpg ...`, plus `--selftest` |
| `probe-models.mjs` | model I/O shapes + CoreML-vs-CPU benchmark |
| `dump-crops.mjs` | renders the aligned 112×112 crops so alignment can be eyeballed |
| `oracle.py` | reference embeddings from the Python `insightface` package |
| `compare-oracle.mjs` | diffs our embeddings against the oracle's |
| `decision-diff.mjs` | diffs pairwise *similarities* — the decision-relevant test |

Models live in `../models/buffalo_l/` (gitignored, 275 MB), fetched from the
InsightFace v0.7 GitHub release. Test images in `images/` are Wikimedia public-domain.

## Reproduce

```bash
npm install                                  # adds onnxruntime-node, sharp
node spike/match.mjs --selftest              # transform recovery, exact to 1e-14
node spike/probe-models.mjs                  # EP benchmark
node spike/match.mjs --probe spike/images/obama_portrait.jpg \
  --target spike/images/obama_cabinet.jpg
```

The oracle needs its own Python venv — the one used for these results lived in a
temp dir and is gone, so recreate it:

```bash
python3 -m venv /tmp/orcl
/tmp/orcl/bin/pip install insightface onnxruntime opencv-python-headless
/tmp/orcl/bin/python spike/oracle.py spike/images/*.jpg 2>/dev/null \
  | sed -n 's/^[^{]*\({"spike.*\)/\1/p' > /tmp/oracle.json   # strip ORT stdout chatter
node spike/compare-oracle.mjs /tmp/oracle.json
node spike/decision-diff.mjs  /tmp/oracle.json
```

`oracle.py` reads models from `../models`, so it won't re-download the 275 MB pack.

## Findings

### CoreML is a large win and costs no accuracy

Median over 10 runs after warmup, M-series MacBook Pro:

| model | CPU | CoreML | speedup |
|---|---|---|---|
| `det_10g` (SCRFD detector) | 186 ms | 52 ms | 3.6× |
| `w600k_r50` (ArcFace) | 151 ms | 2.7 ms | 55× |

ORT reported `nodes supported by CoreML: 133` of 153 for the detector — genuine
partitioning, not silent CPU fallback. The official ORT docs table claiming macOS is
CPU-only for the Node binding **is stale**; `CoreMLExecutionProvider` is compiled into
the `darwin/arm64` dylib in `onnxruntime-node@1.27.0`.

Accuracy cost of fp16-on-ANE, measured against the CPU fp32 path: **~0.00005 cosine.**
Free, in practice.

**Session creation costs 0.7 s (detector) and 4.4 s (recognizer)** — one-time graph
compilation. Sessions must be built once at startup, never per request.

End to end: **42–98 ms per photo**, including detection and embedding *every* face
(the 24-face cabinet photo took 97 ms).

### Separation is clean

Probe: official Obama portrait.

| target | cosine |
|---|---|
| Obama, different photo | 0.765 |
| Obama, different photo/era | 0.693 |
| Obama in 24-person cabinet photo | 0.706 (next best face: 0.178) |
| Obama in 23-person G8 photo | 0.727 (next best face: 0.078) |
| Biden portrait | −0.047 |

Same-person ≈ 0.69–0.77, impostors ≈ 0.05 with a worst case of 0.18. The margin is
wide. Note cosine runs −1..1, **not** Rekognition's 0..100 Similarity — thresholds
do not port.

### The port matches the reference

`compare-oracle.mjs`, our embedding vs Python's for the same face:

| image | faces ts/py | bbox drift | cosine(ts, py) |
|---|---|---|---|
| obama_portrait | 1/1 | 0.6 px | 0.9921 |
| obama_alt | 1/1 | 0.1 px | 0.9996 |
| biden_portrait | 1/1 | 0.2 px | 0.9983 |
| g8_group | 23/23 | 0.0 px | 0.9998 |
| obama_cabinet | 24/24 | 0.0 px | 0.9997 |

Face counts match exactly on the crowd photos. The residual on `obama_portrait` is
most likely `cv2.resize`'s fixed-point interpolation, which we don't replicate bit-exactly.

`decision-diff.mjs` asks the question that matters — do pairwise similarities differ
enough to flip a verdict? **Largest disagreement across all pairs: 0.0134**, against a
match/impostor margin of ~0.65. Decisions are equivalent.

### Two things that cost real accuracy if you get them wrong

Both were found by the oracle, not by inspection — end-to-end results looked
"fine" while being quietly degraded:

1. **Letterbox resampling.** Nearest-neighbour + `round()` on the target dimensions
   gave worst-case agreement of **0.9399**. Matching insightface exactly — `int()`
   truncation, bilinear with OpenCV's `(d+0.5)*scale-0.5` half-pixel convention,
   `det_scale = new_height/orig_height` — lifted it to **0.9921**.
2. **The two normalizations differ.** Detector uses `(x−127.5)/128`, recognizer uses
   `(x−127.5)/127.5`. Easy to unify by accident.

The alignment transform is the other high-risk piece; `match.mjs --selftest` recovers
a known similarity transform to 1e-14, and `dump-crops.mjs` renders the actual crops.

### EXIF orientation is handled — measured, not assumed

Rekognition auto-corrects JPEG EXIF orientation and translates bounding boxes into the
corrected space. `loadImage` replicates this with `sharp().rotate()`. Every Wikimedia
test image has `orientation=none`, so this was verified against synthesised copies:

| target | cosine | bbox |
|---|---|---|
| `obama_alt.jpg` baseline | 0.7626 | 301×443 |
| pixels rotated 90° CCW + `orientation=6` | **0.7597** | 301×442 — recovered |
| same pixels, tag stripped | 0.6321 | 550×328 — sideways, degraded |

The tagged case recovers the original geometry and score. The untagged control shows
what dropping `.rotate()` would cost: SCRFD still finds a sideways face, so it fails
*quietly* — degraded score, no error. Phone photos are routinely EXIF-rotated, so this
matters on a real library.

### Don't adopt the `bbox height < 50px` filter without evidence

In the cabinet photo, Obama's face is **33×43 px** and matched at **0.706**, against a
next-best impostor of 0.178. A 50 px minimum would have discarded that correct match.

That refutes a blanket "always drop faces under 50 px" — it does not establish that
small faces are safe in general. This is one small-face true positive in a posed,
well-lit official photograph, with **no small-face impostor data at all**, and the
impostor tail at small sizes is precisely what such a filter defends against. The right
minimum size (if any) is an output of calibration, not a constant from the note.

Relatedly, the note's claim that "Rekognition did these quality filters implicitly" is
false: `QualityFilter`'s default is `NONE`, which is what `compare-face.ts` already
passes. Adding detector-score and size filters is a **new, stricter policy**, not
parity restoration. Recommend not porting that advice without evidence.

## Not done

- Threshold calibration against real labelled data. The numbers above are a handful of
  celebrity photos, not an operating point. Bootstrap the labels by running the existing
  Rekognition path over a representative folder and hand-verifying.
- Any integration with the app (`/submit` still does one Rekognition call per photo,
  re-uploading the probe each time).
- Licensing: `buffalo_l` weights are research-licensed. Unresolved, and unresolved is
  fine for a spike — but not for shipping.
