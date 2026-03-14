import React from 'react';

const InfoBoxes = () => {
  return (
    <div className="footer-info-container fade-in">
      <div className="footer-grid">
        {/* Left Box: Instructions */}
        <div className="footer-box">
          <h3>How to Use 🚀</h3>
          <p>This application uses an <strong>LSTM (Deep Learning)</strong> model to predict the sentiment score of your comments.</p>
          <ul className="info-list">
            <li>Works exclusively with English comments.</li>
            <li>Longer, detailed comments provide higher accuracy.</li>
            <li>Results are processed in real-time via the AI backend.</li>
          </ul>
        </div>

        {/* Middle Box: Score Scale */}
        <div className="footer-box">
          <h3>Score Scale 📊</h3>
          <div className="scale-item red">🔴 1.0 - 2.0: Strongly Negative</div>
          <div className="scale-item orange">🟠 2.0 - 3.0: Negative/Disappointed</div>
          <div className="scale-item gray">⚪ 3.0: Neutral/Objective</div>
          <div className="scale-item light-green">🟢 3.0 - 4.0: Positive/Satisfied</div>
          <div className="scale-item bright-green">💎 4.0 - 5.0: Extremely Positive</div>
        </div>

        {/* Right Box: Try These */}
        <div className="footer-box">
          <h3>Try These Examples ✨</h3>
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