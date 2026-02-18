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
      const response = await axios.post('http://localhost:5000/predict', { text });
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
        <h2>LSTM Sentiment</h2>
        <button onClick={toggleTheme} className="theme-toggle">
          {theme === 'light' ? '🌙 Koyu Mod' : '☀️ Açık Mod'}
        </button>
      </nav>
      
      <main className="content">
        <header>
          <h1>Yorum Puan Analizi</h1>
          <p>Yapay zeka yorumunuzu 1-5 arası puanlar.</p>
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