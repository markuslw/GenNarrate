import './App.css';
import Logo from './imgs/logo.png';
import Mute from './imgs/mute.png';
import Open from './imgs/open.png';
import Typing from './imgs/typing.png';
import Speaker from './imgs/speaker.png';
import React, { useState } from 'react';
import Recorder from 'recorder-js';

function App() {

  /*
    State variables to manage recording and text-to-speech features.
  */
  const [speech, setSpeech] = useState(false);
  const [textToSpeech, setTextToSpeech] = useState(false);

  /*
    State variables to manage audio recording.
  */
  const [isRecording, setIsRecording] = useState(false);
  const [recorder, setRecorder] = useState(null);
  const [stream, setStream] = useState(null);

  /*
    State variables to manage prompt, file, and chat history.
  */
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState(null);
  const [history, setHistory] = useState([
    { role: "User", content: "Hello Botty!" },
    {role: "Botty", content: "Hello User!"}
  ]);

  /*
    Function to send audio data to the server.
  */
  const submitAudio = async (audioBlob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    if (history.length !== 0) formData.append('history', JSON.stringify(history));
    if (textToSpeech === true) formData.append('tts', "true");

    const response = await fetch('http://localhost:8000/upload/speech/', {
      method: 'POST',
      body: formData,
    })
      .catch(error => {
        console.error("Unexpected error:", error);
      });

    if (response) {
      const data = await response.text();
      setHistory(history => [...history, { role: "Botty", content: data }]);
    }
  };

  /*
    Function to handle request.
  */
  const submitPrompt = async (event) => {
    event.preventDefault();

    if (file !== null) {
      submitFile(event);
      return;
    }

    const formData = new FormData();
    if (prompt !== "") formData.append('prompt', prompt);
    if (history.length !== 0) formData.append('history', JSON.stringify(history));
    if (textToSpeech === true) formData.append('tts', "true");

    setHistory((history) => [...history, {role: "User", content: prompt}]);
    setPrompt("");

    const response = await fetch('http://localhost:8000/upload/text', {
      method: 'POST',
      body: formData,
    })
      .catch(error => {
        console.error("Unexpected error:", error);
      });
    
    if (response) {
      if (textToSpeech) {
        const blob = await response.blob();
        const audioURL = URL.createObjectURL(blob);
        const audio = new Audio(audioURL);
        audio.play();

        setHistory(history => [...history, { role: "Botty", content: "Playing audio..." }]);
      } else {
        const data = await response.text();
        setHistory((history) => [...history, { role: "Botty", content: data }]);
      }
    }
  }

  /*
    Function to handle file upload.
  */
  const submitFile = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    if (file !== null) formData.append('file', file);
    setFile(null);

    const response = await fetch('http://localhost:8000/upload/file', {
      method: 'POST',
      body: formData,
    });

    const data = await response.text();
    setHistory((history) => [...history, { role: "Botty", content: data }]);
  }

    /*
    Function to start audio recording.
  */
  const startRecording = async () => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const newRecorder = new Recorder(audioContext, {
      type: 'audio/wav',
    });

    await newRecorder.init(micStream);
    newRecorder.start();

    setRecorder(newRecorder);
    setStream(micStream);
    setIsRecording(true);
  };

  /*
    Function to stop audio recording and send the recorded audio to the server.
  */
  const stopRecording = async () => {
    const { blob } = await recorder.stop();
    stream.getTracks().forEach(track => track.stop());
    submitAudio(blob);
    setIsRecording(false);
  };
  
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

  /*
    Function to handle TTS toggle.
  */
  const handleTextToSpeech = () => {
    setTextToSpeech(!textToSpeech);
  };

  /*
    Function to handle speaking toggle.
  */
  const handleSpeech = () => {
    setSpeech(!speech);
    isRecording ? stopRecording() : startRecording();
  };

  return (
    <div className="root">
      <img alt="" src={Logo} className='logo' />
      <h1>GenNarrate</h1>

      <div className='buttons-container'>
        <button
          className='button'
          onClick={handleTextToSpeech}
          style={textToSpeech ? { backgroundColor: '#3b82f6' } : {}} >
          {textToSpeech ? (
            <img alt="" style={{ width: 40, height: 'auto' }} src={Speaker} />
          ) : (
            <img alt="" style={{ width: 40, height: 'auto' }} src={Typing} />
          )}
        </button>

        <button
          className='button'
          onClick={handleSpeech}
          style={speech ? { backgroundColor: '#3b82f6' } : {}} >
          {speech ? (
            <img alt="" style={{ width: 40, height: 'auto' }} src={Open} />
          ) : (
            <img alt="" style={{ width: 40, height: 'auto' }} src={Mute} />
          )}
        </button>
      </div>

      <div className="chatHistory">
        {history.map((message, index) => (
          <p key={index} className={message.role === "User" ? "message right" : "message left"}>
            {message.content}
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
