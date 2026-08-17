# Local face matching with saved people

**Date:** 2026-08-17
**Status:** approved, not yet implemented

## Goal

Replace AWS Rekognition `CompareFaces` with a local InsightFace pipeline running on
Apple Silicon, and add a persistent library of saved people so a person can be searched
for repeatedly without re-uploading their photo.

Done when:

1. The app performs face search with no network calls and no AWS credentials.
2. Scanning a folder embeds the probe once rather than once per photo.
3. A person can be saved with one or more reference photos and reused in a later session.
4. `spike/compare-oracle.mjs` still reports ≥ 0.99 agreement against the Python oracle.

Delivered in two sequenced parts. **Part A** is the migration; **Part B** is saved
people. They touch different files and B depends only on A's stateless search contract.

## Background

The spike (`spike/README.md`) established the numbers this design rests on:

- CoreML gives 3.6× on detection and 55× on recognition, at ~0.00005 cosine accuracy
  cost. Per photo: 42–98 ms including every face.
- Session creation costs 0.7 s + 4.4 s, so sessions must be built once at startup.
- Same-person cosine 0.69–0.77; worst impostor across 47 crowd faces 0.18.
- The TS port agrees with the Python reference to 0.992 worst case, and pairwise
  similarities differ by at most 0.013 — no decision changes.

Two findings contradict the original task note and are binding on this design:

- **No minimum face-size filter.** A 33×43 px face matched correctly at 0.706; a 50 px
  floor would have discarded it.
- **Rekognition applied no quality filtering.** `QualityFilter` defaults to `NONE`,
  which is what the current code passes. Adding filters would be a new, stricter
  policy, not parity.

## Current state

358 lines across three files. No database, no persistence, no tests, no test runner.
`POST /submit` receives both images, calls Rekognition, and returns the raw AWS response
to the browser, which reads `SourceImageFace` / `FaceMatches` / `UnmatchedFaces` directly.
The client loops a directory at concurrency 2, one request per photo, zipping matches.

Consequence: scanning 500 photos uploads and re-analyses the source image 500 times.

## Part A — local matching

### Module layout

```
src/face/pipeline.ts   detection, alignment, embedding. No Express, no HTTP.
src/face/matcher.ts    owns both ORT sessions; serializes inference;
                       embedAll(buffer), embedLargest(buffer)
src/routes/probe.ts    POST /probe
src/routes/search.ts   POST /search
src/server.ts          loads models at boot, wires routes
```

`src/compare-face.ts` is deleted. `@aws-sdk/client-rekognition` is removed from
`package.json`; `AWS_ACCESS_KEY` / `AWS_SECRET_ACCESS_KEY` are removed from `.env.example`.
The old implementation remains recoverable at `git show a84db4c:src/compare-face.ts`.

`pipeline.ts` is a typed port of the validated `spike/insightface.mjs`. It must stay free
of Express so `spike/compare-oracle.mjs` can keep importing it — the oracle is the
regression check for this module and is expected to outlive the spike.

### Pipeline behaviour (fixed by the spike; do not re-derive)

- EXIF orientation applied via `sharp().rotate()` before anything else. Bounding boxes
  are therefore in rotated space.
- Letterbox must match insightface exactly: `int()` truncation on target dimensions,
  bilinear with OpenCV's `(d+0.5)*scale-0.5` half-pixel convention, `det_scale =
  new_height / orig_height`, padding `-127.5/128`.
- Detector normalizes `(x-127.5)/128`; recognizer normalizes `(x-127.5)/127.5`. These
  differ and must not be unified.
- SCRFD: strides 8/16/32, 2 anchors per location, det threshold 0.5, NMS IoU 0.4.
- Alignment: least-squares 2D similarity transform onto the ArcFace 5-point reference.
- Embeddings L2-normalized, so cosine is a dot product.

### Model setup

The 275 MB `buffalo_l` pack is not in git. Add `npm run setup:models`, which downloads
and extracts it to `models/buffalo_l/` from the InsightFace v0.7 GitHub release and is a
no-op when the two required files are already present. It is a documented prerequisite
in the README rather than a `postinstall` hook — a 275 MB silent download on `npm
install` is unwelcome, and only `det_10g.onnx` and `w600k_r50.onnx` are needed of the
five files in the pack.

Server boot fails with an explicit message pointing at this script if the models are
missing, rather than throwing an ORT file-not-found error.

### Model loading and concurrency

Both sessions are created once at boot with `executionProviders: ['coreml', 'cpu']`,
awaited **before** `app.listen()`. Roughly 5 s of startup; log when ready so the delay is
not mistaken for a hang.

Inference is serialized behind a promise-chain mutex in `matcher.ts`. The model is the
bottleneck; concurrent `session.run` calls would contend without improving throughput.
Client-side concurrency stays at 2 so a photo uploads while the previous one is inferring.

### API

```
POST /probe
  multipart: source
  200 → { embedding: number[512],
          face: { box: {top,left,width,height}, score: number } }
  422 → { error: "no_face_detected" }

POST /search
  multipart: target
  field:     probe     — JSON array of embeddings (1 or more)
  field:     personId  — id of a saved person; Part B only, rejected with 400 in Part A
  200 → { threshold: number,
          matched: boolean,
          faces: [ { box: {...}, cosine: number, detScore: number,
                     matched: boolean, bestReference: number } ] }
  400 → { error: "bad_probe" | "unreadable_image" }
```

`probe` is always an **array** of embeddings. A one-off uploaded photo sends a
one-element array; a saved person with three reference photos sends three, or passes
`personId` and lets the server load them. A candidate face scores as the **maximum**
cosine across all references, with `bestReference` naming which one won. This is what
absorbs pose and lighting variation.

`personId` exists so a multi-reference scan does not resend ~30 KB of vectors per photo
across hundreds of requests. It reads durable stored data, not ephemeral session state,
so it does not reintroduce the probe-session lifetime problems that were rejected.

Boxes are always relative 0–1, so `canvasRect` (`index.ts:232`) is untouched. Top-level
`matched` is what directory mode uses to decide whether to zip; the client stops
interpreting scores itself.

### Threshold

`FACE_MATCH_THRESHOLD`, default `0.4`, documented as **provisional**. It sits in the gap
the spike measured (matches 0.69–0.77, worst impostor 0.18) but is not a calibrated
operating point. The UI displays raw cosine, labelled as cosine — Rekognition's 0–100
Similarity scale is gone and its thresholds do not port.

### Error handling

A target photo with **no faces is a 200 with an empty array**, not an error. Rekognition
raised `InvalidParameterException` here, which is wrong for a folder scan where "no
people in this photo" is a routine outcome. Unreadable or corrupt images return 400 and
the client counts them as scanned-not-matched rather than aborting the run.

`multer` temp files under `uploads/` are deleted in a `finally`, so failures do not leak
them. The current code unlinks on the success path only.

### Frontend changes

`src/static/index.ts` gains a two-phase flow: call `/probe` once on submit, then loop
targets against `/search`. The progress counters, zip logic, and canvas drawing are
unchanged. The response is read through a small adapter so the AWS field names
(`FaceMatches`, `SourceImageFace`) disappear from the client.

Fix while in this file: `if (!isDirectory)` at `index.ts:92` tests the function object
rather than calling it, so the single-image branch is dead code and line 94 is
unreachable. Harmless today, actively confusing once the flow changes.

## Part B — saved people

### Storage

```
data/people.json
data/people/<id>/ref-1.jpg          downscaled original, max 1600 px long edge
                 ref-1.thumb.png    aligned 112x112 crop, used as the avatar
```

```json
{
  "pipelineVersion": 1,
  "people": [{
    "id": "p_7f3a2b",
    "name": "Sarah",
    "createdAt": "2026-08-17T09:00:00.000Z",
    "references": [{
      "id": "ref-1",
      "image": "people/p_7f3a2b/ref-1.jpg",
      "thumb": "people/p_7f3a2b/ref-1.thumb.png",
      "embedding": [512 floats],
      "detScore": 0.91,
      "addedAt": "2026-08-17T09:00:00.000Z"
    }]
  }]
}
```

Loaded into memory at boot, written on change. Writes go to a temp file and are renamed,
and are serialized through the same mutex pattern as inference so concurrent requests
cannot interleave a read-modify-write. `data/` is gitignored.

At ~200 people this file is well under a megabyte and a full scan is a few hundred
thousand multiplications — microseconds. SQLite and pgvector are both unnecessary at
this scale; revisit only at thousands of people or concurrent writers.

### Why originals are stored

`pipelineVersion` in the file is compared against the code's version at boot. On
mismatch, every reference is re-embedded from its stored image and the file rewritten.

This is not hypothetical: the spike's letterbox fix changed every embedding it had
computed (agreement 0.94 → 0.992). The pipeline is newly ported and will change again.
Without stored originals, every saved person would silently hold a stale vector and the
only remedy would be re-uploading each photo by hand.

Downscaling to 1600 px loses nothing usable — detection runs at 640×640, so larger
images cost storage without improving any derived result. Roughly 350 KB per reference;
50 people at 3 photos each is about 50 MB.

### API

```
GET    /people                        → [{ id, name, references: [{id, thumb, detScore}] }]
POST   /people        multipart: photo, name        → creates person with one reference
POST   /people/:id/references  multipart: photo     → adds a reference
DELETE /people/:id
DELETE /people/:id/references/:refId
```

`POST` routes reject a photo with no detectable face with 422 `no_face_detected`, and
refuse to delete the last remaining reference of a person (delete the person instead).
Names are not required to be unique; `id` is the key.

### Frontend changes

The source panel gains a toggle: **Upload a photo** or **Choose a saved person**. The
picker lists saved people by avatar and name. After a successful `/probe`, a
"Save as person…" control appears, so saving is a natural continuation of a search that
already worked rather than a separate chore. Managing a person — adding another reference
photo, deleting — is reachable from the picker.

Searching with a saved person selected passes `personId` to `/search`.

## Testing

The project has no test runner. Use built-in `node:test` — no new dependency, and Node 20
is already required.

| what | how |
|---|---|
| similarity transform | existing `--selftest`, recovers a known transform to 1e-14 |
| pipeline regression | `spike/compare-oracle.mjs` ≥ 0.99 vs the Python oracle |
| decision equivalence | `spike/decision-diff.mjs`, max pairwise delta < 0.02 |
| known pairs | Obama/Obama > 0.6, Obama/Biden < 0.2 on the committed spike images |
| EXIF | rotated `orientation=6` copy scores within 0.01 of the upright original |
| store | create / add reference / delete / reload from disk round-trips |
| `pipelineVersion` | bumping it re-embeds from stored images rather than serving stale |
| routes | `/probe` then `/search` returns the expected match on the spike images |

The oracle needs a Python venv, so it is a documented manual check rather than part of
the default `npm test` run.

## Out of scope

- **Threshold calibration.** Needs a labelled set from the user's own photos. The 0.4
  default is provisional and explicitly not validated.
- **A persistent index of scanned folders.** Searching a folder still rescans it. Saved
  people persist the *probe*, not the *gallery*.
- **Licensing.** `buffalo_l` weights are research-licensed and commercial use needs a
  separate licence from InsightFace. Fine for a personal tool; unresolved for shipping.
  Removing Rekognition also removes the licence-clean fallback.
- **Model weights in git.** `models/` stays gitignored and is fetched by a setup step.
