import 'dotenv/config';
import * as express from 'express';
import * as path from 'path';

const app = express();
const port = 3000;

// Serve static files
app.use(express.static(path.join(__dirname)));

// Serve index.html for all routes
app.get('*', (req, res) => {
  console.log('req path', req.path);
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
