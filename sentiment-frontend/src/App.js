import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/predict', { text });
      setScore(response.data.score);
    } catch (error) {
      console.error("Error Occurred!", error);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>LSTM Yorum Puan Tahmini</h1>
        <textarea 
          placeholder="Enter your review..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button onClick={handlePredict} disabled={loading}>
          {loading ? 'Analyzing...' : 'Grade'}
        </button>
        
        {score !== null && (
          <div className="result">
            <h3>Tahmini Skor: {score} / 5.0</h3>
            <div className="stars">
              {"⭐".repeat(Math.round(score))}
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;