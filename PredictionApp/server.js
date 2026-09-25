const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');
const fs = require('fs');

const app = express();

// Serve static files (HTML, CSS) from the 'public' folder
app.use(express.static('public'));

// Configure multer for file upload
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
      const uploadDir = path.join(__dirname, 'uploads');
      console.log('Upload directory path:', uploadDir);      
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir);
            console.log('Uploads directory created:', uploadDir); // Log directory creation
        }
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

// Handle image upload and call Python script for prediction
app.post('/upload', upload.single('image'), (req, res) => {
    if (!req.file) {
        console.error('No file uploaded');
        return res.status(400).json({ error: 'No file uploaded.' });
    }
    console.log('File uploaded successfully:', req.file);
    console.log('File uploaded successfully:', req.file);
    console.log('Uploaded file path:', req.file.path);

    const imagePath = req.file.path; // Ensure this line is present
    const pythonScript = path.join(__dirname, 'predict.py');

    const command = `python "${pythonScript}" "${imagePath}"`;
    console.log('Executing command:', command);

  exec(command, (error, stdout, stderr) => {
      if (error) {
          console.error('Python execution error:', error);
          console.error('stderr:', stderr); // Log stderr for debugging

          return res.status(500).json({
              error: 'Error during prediction',
              details: stderr || error.message
          });
      }

      // Log the output from the Python script
      console.log('Python script output:', stdout.trim());

      try {
          const predictionResponse = JSON.parse(stdout.trim());
          return res.json(predictionResponse);
      } catch (e) {
          console.error('Failed to parse JSON:', e);
          console.error('Output from Python script:', stdout.trim());
          return res.status(500).json({
              error: 'Error parsing prediction response',
              details: stdout.trim()
          });
      }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Unexpected error:', err);
    res.status(500).json({ error: 'Unexpected error occurred' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
