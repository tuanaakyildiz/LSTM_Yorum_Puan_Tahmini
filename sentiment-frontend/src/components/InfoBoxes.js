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
            <code>"This place is a total disaster. The staff was incredibly rude and the food was literally inedible. Avoid this place at all costs!"</code>
            <code>"It was an okay experience. Some parts of the meal were decent, but the atmosphere was a bit too loud for my taste. Nothing special."</code>
            <code>"We had a really nice time here. The food was good and the waiter was very friendly. A solid spot for a casual dinner."</code>
            <code>"We had a really nice time at this place. The food was quite good and arrived quickly. Our waiter was very friendly and made sure we had everything we needed. It is a solid spot for a casual dinner and I would definitely come back again."</code>
            <code>"Absolute masterpiece! I loved every single bite. The best service I have ever received and the atmosphere was perfect. Highly recommend!"</code>
          </div>
        </div>

      </div>
    </div>
  );
};

export default InfoBoxes;