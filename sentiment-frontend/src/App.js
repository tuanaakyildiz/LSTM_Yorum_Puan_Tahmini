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
      //{ text } yerine { review: text } yazarak backend'in beklediği formata çevrildi
      const response = await axios.post('https://tuievolution-sentiment-api.hf.space/predict', { 
        review: text 
      });
      
      setScore(response.data.score); // Not: Backend "raw_score" veya "confidence" dönüyorsa burayı ona göre güncellemelisin
    } catch (error) {
      console.error("API Hatası:", error); // Hatayı konsolda detaylı görmek için
      alert("Sunucuya bağlanılamadı! Lütfen tekrar deneyin.");
    } finally {
      setLoading(false);
    }
  };

  //testing in local
/*   const handlePredict = async () => {
    setLoading(true);
    try {
      // 1. URL geçici olarak lokale (kendi bilgisayarına) yönlendirildi
      const response = await axios.post('http://localhost:5000/predict', { 
        review: text 
      });
      
      // 2. DEDEKTİF KODU: Backend'den tam olarak ne geliyor? Tarayıcı konsoluna yazdırıyoruz.
      console.log("API'den Gelen Yanıt:", response.data); 

      // 3. SKOR DEĞİŞKENİ: Eski Flask kodunda veriyi "raw_score" olarak dönüyorduk. 
      // Eğer backend "score" yerine "raw_score" veya başka bir isim dönüyorsa hata alırsın.
      // Garantilemek için ikisini de kontrol ediyoruz:
      setScore(response.data.score || response.data.raw_score); 
      
    } catch (error) {
      console.error("API Hatası:", error);
      alert("Sunucuya bağlanılamadı! Lütfen Flask'ın çalıştığından emin olun.");
    } finally {
      setLoading(false);
    }
  }; */


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