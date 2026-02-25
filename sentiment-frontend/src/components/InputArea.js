const InputArea = ({ text, setText, handlePredict, loading }) => (
  <div className="input-container">
    <textarea
      placeholder="Enter your comment"
      value={text}
      onChange={(e) => setText(e.target.value)}
      className="custom-textarea"
    />
    <button className="predict-btn" onClick={handlePredict} disabled={loading || !text.trim()}>
      {loading ? <span className="loader"></span> : 'Score'}
    </button>
  </div>
);
export default InputArea;