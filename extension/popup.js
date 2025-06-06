document.addEventListener("DOMContentLoaded", () => {
  const select = document.getElementById("symbol");
  const output = document.getElementById("output");

  function fetchPrediction(symbol) {
    output.innerHTML = "<p class='loading'>Loading prediction...</p>";
    fetch(`http://127.0.0.1:5000/predict?symbol=${symbol}`)
      .then((response) => response.json())
      .then((data) => {
        output.innerHTML = `
          <p><strong>Current Price:</strong> $${data.current_price}</p>
          <p><strong>Predicted Price:</strong> $${data.predicted_price}</p>
          <p class="signal">🔔 Signal: ${data.signal}</p>
        `;
      })
      .catch((error) => {
        output.innerHTML = "<p style='color:red;'>Error fetching data.</p>";
        console.error(error);
      });
  }

  select.addEventListener("change", () => {
    fetchPrediction(select.value);
  });

  fetchPrediction(select.value);
});
