import './App.css';
import Logo from './logo.png';
import Mute from './mute.png';
import Open from './open.png';
import React, { useState } from 'react';

function App() {

  const [TTS, setTTS] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState(null);
  const [history, setHistory] = useState([
    { role: "User", text: "Hello Botty!" },
    {role: "Botty", text: "Hello User!"}
  ]);

  /*
    Function to handle request.
  */
  const submitPrompt = (event) => {
    event.preventDefault();

    const formData = new FormData();
    
    if (prompt !== "") {
      formData.append('prompt', prompt);
    }
    if (file !== null) {
      formData.append('file', file);
    }
    if (history.length !== 0) {
      formData.append('history', JSON.stringify(history));
    }
    formData.append('tts', TTS);

    setHistory((history) => [...history, {role: "User", text: prompt}]);
    setFile(null);
    setPrompt("");

    const postWords = fetch('http://localhost:8000/upload/text/', {
      method: 'POST',
      body: formData,
    }).then(response => response.json())
      .then(data => data["message"])
      .then(message => {
        console.log("Response from server:", message);
        setHistory((history) => [...history, {role: "Botty", text: message}]);
      });
  }
  
  /*
    Function to handle text input changes.
  */
  const handlePrompt = (e) => {
    const inputText = e.target.value;
    setPrompt(inputText);
  }

  /*
    Function to handle file input changes.
  */
  const handleFile = (e) => {
    const inputFile = e.target.files[0];
    setFile(inputFile);
  };

  const handleTTS = () => {
    setTTS(!TTS);
    if (TTS === true) {
      
    } else {
      
    }
  };

  return (
    <div className="root">
      <img src={Logo} className='logo' />
      <h1>GenNarrate</h1>

      <button className='ttsButton' onClick={handleTTS} style={TTS ? { backgroundColor: '#3b82f6' } : {}} >
        {TTS ? (
          <img style={{ width: 40, height: 'auto' }} src={Open} />
        ) : (
          <img style={{ width: 40, height: 'auto' }} src={Mute} />
        )}
      </button>

      <div className="chatHistory">
        {history.map((message, index) => (
          <p key={index} className={message.role === "User" ? "right" : "left"}>
            {message.text}
          </p>
        ))}
      </div>
      
      <form className="chatBox" onSubmit={submitPrompt}>
        <input className="textField" type="text" placeholder="Start typing..." onChange={handlePrompt} />
        <div className="upload">
          <input type="file" accept="application/pdf" onChange={handleFile} />
          {file && (
            <p>Uploaded: <strong>{file.name}</strong></p>
          )}
        </div>
      </form>
    </div>
  );
}

export default App;
