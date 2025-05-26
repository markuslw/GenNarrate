import './App.css';
import React, { useState } from 'react';

function App() {

  const [pdfFile, setPdfFile] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === "application/pdf") {
      setPdfFile(file);

      const formData = new FormData();
      formData.append('file', file);

      const postFile = fetch('http://localhost:8000/api/upload/', {
        method: 'POST',
        body: formData,
        credentials: 'omit',
      });
      
    } else {
      alert("Please upload a valid PDF file.");
    }
  };

  return (
    <div className="root">
      <h1>Upload a PDF</h1>
      <input type="file" accept="application/pdf" onChange={handleFileChange} />
      {pdfFile && (
        <p>Uploaded: <strong>{pdfFile.name}</strong></p>
      )}
    </div>
  );
}

export default App;
