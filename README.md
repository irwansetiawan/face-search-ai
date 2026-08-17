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
