import 'dotenv/config';
import express from 'express';
import path from 'path';
import { compareFace } from './compare-face.js';
import multer from 'multer';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

// Serve static files
app.use(express.static(path.join(__dirname, '/static')));

// Submit
const upload = multer({ dest: 'uploads/' })
const submitUploadFields = upload.fields([{name:'source'},{name:'target'}])
app.post('/submit', submitUploadFields, compareFace);

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
