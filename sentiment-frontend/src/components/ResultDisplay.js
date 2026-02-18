const ResultDisplay = ({ score }) => {
  if (score === null) return null;
  const roundedScore = Math.round(score);

  return (
    <div className="result-card fade-in">
      <h3>Tahmini Skor</h3>
      <div className="score-value">{score.toFixed(1)} <span className="total-score">/ 5.0</span></div>
      <div className="stars">
        {[...Array(5)].map((_, i) => (
          <span key={i} className={i < roundedScore ? "star filled" : "star"}>★</span>
        ))}
      </div>
    </div>
  );
};
export default ResultDisplay;