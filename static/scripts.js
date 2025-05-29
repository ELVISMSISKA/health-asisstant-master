document.getElementById("simulationForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const age = document.getElementById("age")?.value; // Optional chaining in case age is removed
    const initial_infected = document.getElementById("initial_infected").value;
    const contact_rate = document.getElementById("contact_rate").value;
    const recovery_rate = document.getElementById("recovery_rate").value;
    const disease_duration = document.getElementById("disease_duration").value;

    function addAgeGroup() {
    const container = document.getElementById("ageGroupInputs");
    const groupNumber = container.children.length / 3 + 1;

    const groupHTML = `
        <h4>Age Group ${groupNumber}</h4>
        <label for="contact_rate[]">Contact Rate (β):</label>
        <input type="number" step="0.01" name="contact_rate[]" required>

        <label for="recovery_rate[]">Recovery Rate (γ):</label>
        <input type="number" step="0.01" name="recovery_rate[]" required>

        <label for="initial_infected[]">Initial Infected:</label>
        <input type="number" name="initial_infected[]" required>
    `;
    container.insertAdjacentHTML("beforeend", groupHTML);
}

    // POST the data to Flask backend
    fetch("/simulate", {
        method: "POST",
        body: new URLSearchParams({
            ...(age ? { "age": age } : {}), // Include age only if it exists
            "initial_infected": initial_infected,
            "contact_rate": contact_rate,
            "recovery_rate": recovery_rate,
            "disease_duration": disease_duration
        })
    })
    .then(response => response.json())
    .then(data => {
        const time = data.time;
        const susceptible = data.susceptible;
        const infected = data.infected;
        const recovered = data.recovered;
        const description = data.description;

        // Plot the results using Chart.js
        const ctx = document.getElementById("chart").getContext("2d");
        new Chart(ctx, {
            type: "line",
            data: {
                labels: time,
                datasets: [
                    {
                        label: "Susceptible",
                        data: susceptible,
                        borderColor: "blue",
                        fill: false
                    },
                    {
                        label: "Infected",
                        data: infected,
                        borderColor: "red",
                        fill: false
                    },
                    {
                        label: "Recovered",
                        data: recovered,
                        borderColor: "green",
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: "top"
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Days" // Label for X-axis
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: "Population Count" // Label for Y-axis
                        }
                    }
                }
            }
        });

        // Display the description below the chart
        document.getElementById("description").innerHTML = description;
    });
});
document.getElementById("healthForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const age = document.getElementById("age").value;
    const bmi = document.getElementById("bmi").value;
    const bloodPressure = document.getElementById("blood_pressure").value;
    const cholesterol = document.getElementById("cholesterol").value;

    const formData = new FormData();
    formData.append('age', age);
    formData.append('bmi', bmi);
    formData.append('blood_pressure', bloodPressure);
    formData.append('cholesterol', cholesterol);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("predictionOutput").textContent =  `Predicted Health Risk: ${parseFloat(data.prediction).toFixed(2)}`;
    })
    .catch(error => {
        console.error("Error:", error);
    });
});
