# Introduction

Face Search AI is a web application that analyzes and compares faces using a source image and find matching faces in the target image using machine learning, running entirely on your own machine.

Nobody can notice Leonardo DiCaprio in a crowd, but AI can:

![preview](https://github.com/irwansetiawan/face-search-ai/assets/1826105/65bbb40d-be97-4322-a357-a2218a5d951a)

# Requirements

1. Node 20+ on macOS (Apple Silicon recommended — CoreML gives 3.6x on detection
   and 55x on recognition).
2. Download the face models once: `npm run setup:models` (275 MB).

No cloud account and no API costs — everything runs locally.

# Run in localhost

```
npm install
npm run setup:models
npm run build
npm run server
```

Then open http://localhost:3100/ in your browser.

# Saved people

You can save a source photo as a named person instead of re-uploading it every
session. Saved people can hold more than one reference photo (add more from
the picker to cover different angles or lighting) — a search matches if the
target face resembles *any* of a person's reference photos, not just the
first.

Saved people are stored locally under `data/`:

- `data/people.json` — names and the 512-float face embedding for every
  reference photo.
- `data/people/<id>/` — a downscaled copy of each reference photo, plus a
  small cropped avatar used by the picker.

**`data/` holds personal biometric data (face embeddings and photos) and is
never uploaded anywhere** — everything in this app runs locally, and `data/`
is git-ignored. Back it up or delete it like you would any other personal
file; there's no cloud copy to fall back on.
