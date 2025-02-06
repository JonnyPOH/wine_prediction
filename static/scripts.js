async function predictWineCountry() {
  const inputField = document.getElementById("features");
  const features = inputField.value.split(",").map(Number);

  const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: features })
  });

  const data = await response.json();
  document.getElementById("prediction").innerText = "Predicted Country: " + data.predicted_country;
}
