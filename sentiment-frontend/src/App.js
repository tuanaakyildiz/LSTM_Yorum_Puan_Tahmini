import React, { useState } from 'react';
import axios from 'axios';
import { useTheme } from './hooks/useTheme';
import InputArea from './components/InputArea';
import ResultDisplay from './components/ResultDisplay';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post('https://tuievolution-sentiment-api.hf.space', { text });
      setScore(response.data.score);
    } catch (error) {
      alert("Sunucuya bağlanılamadı!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`app-wrapper ${theme}`}>
      <nav className="navbar">
        <h2>LSTM Sentiment Analysis</h2>
        <button onClick={toggleTheme} className="theme-toggle">
          {theme === 'light' ? '🌙 Dark Mod' : '☀️ Light Mod'}
        </button>
      </nav>
      
      <main className="content">
        <header>
          <h1>Comment Score Analysis</h1>
          <p>AI scores your comment from 1 to 5</p>
        </header>

        <InputArea 
          text={text} 
          setText={setText} 
          handlePredict={handlePredict} 
          loading={loading} 
        />

        <ResultDisplay score={score} />
      </main>
    </div>
  );
}

export default App;