import React from 'react';

const InfoBoxes = () => {
  return (
    <div className="info-container fade-in">
      {/* Turkish Section - Left Side */}
      <div className="info-box info-left">
        <h3>Nasıl Kullanılır? 🇹🇷</h3>
        <p>
          Bu uygulama, yorumlarınızın duygusunu analiz etmek için 
          <strong> LSTM (Deep Learning)</strong> modelini kullanır.
        </p>
        <ul className="info-list">
          <li>Sadece İngilizce yorumlarda çalışır.</li>
          <li>Daha uzun ve detaylı yorumlar, modelin daha doğru sonuçlar vermesini sağlar.</li>
          <li>Skorlar 1 (Çok Olumsuz) ile 5 (Çok Olumlu) arasındadır.</li>
        </ul>
      </div>

      {/* English Section - Right Side */}
      <div className="info-box info-right">
        <h3>How to Use? 🇺🇸</h3>
        <p>
          This app uses an <strong>LSTM (Deep Learning)</strong> model 
          to predict the sentiment score of your comments.
        </p>
        <ul className="info-list">
          <li>This app only works with English comments.</li>
          <li>The more information you give and the lengthier they are, the more accurate results you will receive.</li>
          <li>Scores range from 1 (Very Negative) to 5 (Very Positive).</li>
        </ul>
      </div>
    </div>
  );
};

export default InfoBoxes;