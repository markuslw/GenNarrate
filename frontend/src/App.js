import './App.css';
import Logo from './logo.png';
import React, { useState } from 'react';

function App() {

  const [pdfFile, setPdfFile] = useState(null);
  const [conversation, setConversation] = useState([
    { role: "User", text: "Hello Botty!" },
    {role: "Botty", text: "Hello User!"}
  ]);
  const [text, setText] = useState("");

  const handleFileChange = (e) => {
    e.preventDefault();

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

    const formData = new FormData();
    formData.append('prompt', text);
    formData.append('history', JSON.stringify(conversation));

    setConversation((conversation) => [...conversation, { role: "User", text: text }]);
    setText("");

    const postWords = fetch('http://localhost:8000/api/conversate/', {
      method: 'POST',
      body: formData,
      credentials: 'omit',
    }).then(response => response.json())
      .then(data => data["message"])
      .then(message => {
        console.log("Response from server:", message);
        setConversation((conversation) => [...conversation, {role: "Botty", text: message}]);
      });
  }
  
  const handleConversation = (e) => {
    const inputText = e.target.value;
    setText(inputText);
  }

  return (
    <div className="root">
      <img src={Logo} className='logo' />
      <h1>GenNarrate</h1>

      <div className="chatHistory">
        {conversation.map((message, index) => (
          <p key={index} className={message.role === "User" ? "right" : "left"}>
            {message.text}
          </p>
        ))}
      </div>
      
      <form className="chatBox" onSubmit={handleInput}>
        <input className="textField" type="text" placeholder="Start typing..." onChange={handleConversation} />
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
