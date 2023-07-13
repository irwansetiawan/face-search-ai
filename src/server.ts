import 'dotenv/config';
import * as express from 'express';
import * as path from 'path';
import { compareFace } from './compare-face';
import * as multer from 'multer';

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
