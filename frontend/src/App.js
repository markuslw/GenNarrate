import './App.css';
import React, { useState } from 'react';

function App() {

  const [pdfFile, setPdfFile] = useState(null);

  const [conversation, setConversation] = useState([]);
  const [text, setText] = useState([]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setPdfFile(file);

    const formData = new FormData();
    formData.append('file', file);

    const postFile = fetch('http://localhost:8000/api/upload/', {
      method: 'POST',
      body: formData,
      credentials: 'omit',
    });
  };

  const handleInput = (e) => {
    e.preventDefault();
    console.log("Input text:", text);

    setConversation((conversation) => [...conversation, text]);

    const formData = new FormData();
    formData.append('text', text);

    const postWords = fetch('http://localhost:8000/api/conversate/', {
      method: 'POST',
      body: formData,
      credentials: 'omit',
    }).then(response => response.json())
      .then(data => data["message"])
      .then(message => {
        console.log("Response from server:", message);
        setConversation((conversation) => [...conversation, message]);
      });
  }
  
  const handleConversation = (e) => {
    const inputText = e.target.value;
    setText(inputText);
  }

  return (
    <div className="root">
      <h1>GenNarrate</h1>

      <div className="chatHistory">
        {conversation.map((text, index) => (
          <p key={index} className={index % 2 === 0 ? 'right' : 'left'}>
            {text}
          </p>
        ))}
      </div>
        
      <form className="chatBox" onSubmit={handleInput}>
        <input className="textField" type="text" onChange={handleConversation} />
        <button className="submit" type="submit">Send</button>
        <div className="upload">
          <input type="file" accept="application/pdf" onChange={handleFileChange} />
          {pdfFile && (
            <p>Uploaded: <strong>{pdfFile.name}</strong></p>
          )}
        </div>
      </form>
    </div>
  );
}

export default App;
