import React from 'react';

const InfoBoxes = () => {
  return (
    <div className="footer-info-container fade-in">
      <div className="footer-grid">
        
        {/* Üst Sol: Türkçe Açıklama */}
        <div className="footer-box">
          <h3>Nasıl Kullanılır? 🇹🇷</h3>
          <p>Bu uygulama, yorumlarınızın duygusunu analiz etmek için <strong>LSTM (Derin Öğrenme)</strong> modelini kullanır.</p>
          <ul className="info-list">
            <li>Sadece İngilizce yorumlarda çalışır.</li>
            <li>Yorum ne kadar uzun ve detaylı olursa, sonuç o kadar tutarlı olur.</li>
            <li>Skorlar 1 (Çok Olumsuz) ile 5 (Çok Olumlu) arasındadır.</li>
          </ul>
        </div>

        {/* Üst Sağ: English Description */}
        <div className="footer-box">
          <h3>How to Use? 🇺🇸</h3>
          <p>This app uses an <strong>LSTM (Deep Learning)</strong> model to predict the sentiment score of your comments.</p>
          <ul className="info-list">
            <li>This app only works with English comments.</li>
            <li>The more information you give and the lengthier they are, the more accurate results you'll receive.</li>
            <li>Scores range from 1 (Very Negative) to 5 (Very Positive).</li>
          </ul>
        </div>

        {/* Alt Sol/Orta: Puan Tablosu */}
        <div className="footer-box">
          <h3>Score Scale / Puan Tablosu 📊</h3>
          <div className="scale-item red">🔴 1.0 - 2.0: Strongly Negative / Çok Olumsuz</div>
          <div className="scale-item orange">🟠 2.0 - 3.0: Negative / Olumsuz</div>
          <div className="scale-item gray">⚪ 3.0: Neutral / Nötr</div>
          <div className="scale-item light-green">🟢 3.0 - 4.0: Positive / Olumlu</div>
          <div className="scale-item bright-green">💎 4.0 - 5.0: Very Positive / Çok Olumlu</div>
        </div>

        {/* Alt Sağ/Orta: Örnekler */}
        <div className="footer-box">
          <h3>Try These / Bunları Deneyin ✨</h3>
          <div className="example-sentences">
            <code>"Absolute masterpiece, I loved it!"</code>
            <code>"A complete waste of my time."</code>
            <code>"The plot was fine but acting was slow."</code>
          </div>
        </div>

      </div>
    </div>
  );
};

export default InfoBoxes;