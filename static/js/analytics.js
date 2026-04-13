function markChartReady(canvasId) {
    const canvas = document.getElementById(canvasId);
    const shell = canvas?.closest("[data-chart-shell]");
    if (shell) {
        shell.classList.remove("is-loading");
    }
}

function renderAnalyticsChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return;
    new Chart(canvas, config);
    markChartReady(canvasId);
}

document.addEventListener("DOMContentLoaded", () => {
    const payloadElement = document.getElementById("analyticsPayload");
    if (!payloadElement) return;

    const payload = JSON.parse(payloadElement.textContent);

    renderAnalyticsChart("pieChart", {
        type: "pie",
        data: {
            labels: payload.pie.labels,
            datasets: [{
                data: payload.pie.values,
                backgroundColor: ["#E63946", "#2DC653"]
            }]
        }
    });

    renderAnalyticsChart("lineChart", {
        type: "line",
        data: {
            labels: payload.line.labels,
            datasets: [{
                label: "Scans",
                data: payload.line.values,
                borderColor: "#0A6E6E",
                backgroundColor: "rgba(10, 110, 110, 0.14)",
                fill: true,
                tension: 0.35
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, ticks: { precision: 0 } }
            }
        }
    });

    renderAnalyticsChart("ageBarChart", {
        type: "bar",
        data: {
            labels: payload.age_bar.labels,
            datasets: [{
                label: "Anemic cases",
                data: payload.age_bar.values,
                backgroundColor: "#E63946"
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, ticks: { precision: 0 } }
            }
        }
    });

    renderAnalyticsChart("confidenceBarChart", {
        type: "bar",
        data: {
            labels: payload.confidence_bar.labels,
            datasets: [{
                label: "Scans",
                data: payload.confidence_bar.values,
                backgroundColor: "#0D1B2A"
            }]
        },
        options: {
            indexAxis: "y",
            scales: {
                x: { beginAtZero: true, ticks: { precision: 0 } }
            }
        }
    });
});
