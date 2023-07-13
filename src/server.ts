import 'dotenv/config';
import * as express from 'express';
import * as path from 'path';

const app = express();
const port = 3000;

// Serve static files
app.use(express.static(path.join(__dirname, '/static')));

// Serve all routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, req.path));
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
