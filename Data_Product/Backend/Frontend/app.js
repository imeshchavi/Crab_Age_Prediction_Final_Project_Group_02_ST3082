
document.addEventListener('DOMContentLoaded', () => {
    const predictForm = document.getElementById('predictForm');
    const resultDisplay = document.getElementById('resultDisplay');

    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Get values
            const length = parseFloat(document.getElementById('length').value);
            const height = parseFloat(document.getElementById('height').value);
            const diameter = parseFloat(document.getElementById('diameter').value);
            const shucked_weight_ratio = parseFloat(document.getElementById('shucked_weight_ratio').value);

            const submitBtn = predictForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerText;
            const shuckedError = document.getElementById('shucked-error');

            // --- Validation: shucked_weight_ratio must be > 0 and <= 1 ---
            shuckedError.style.display = 'none'; // reset
            if (isNaN(shucked_weight_ratio) || shucked_weight_ratio <= 0 || shucked_weight_ratio > 1) {
                shuckedError.style.display = 'block';
                document.getElementById('shucked_weight_ratio').focus();
                document.getElementById('shucked_weight_ratio').style.border = '1px solid #e03c3c';
                return; // stop calculation
            }
            // Reset border if valid
            document.getElementById('shucked_weight_ratio').style.border = '1px solid #333';

            // Show loading state
            submitBtn.innerText = 'CALCULATING...';
            submitBtn.disabled = true;

            try {
                // Call API
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        length,
                        height,
                        diameter,
                        shucked_weight_ratio
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    // Show result in MONTHS (Rounded to 2 decimals)
                    const ageMonths = data.predicted_age_months.toFixed(2);

                    document.getElementById('ageValue').innerText = `${ageMonths} Months`;
                    // Pass the numeric value for description logic
                    document.getElementById('ageDescription').innerText = getAgeDescription(data.predicted_age_months);

                    resultDisplay.style.display = 'block';
                    // Scroll to result
                    resultDisplay.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert('Error: ' + data.error);
                }

            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while connecting to the server.');
            } finally {
                submitBtn.innerText = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }
});

function getAgeDescription(ageInMonths) {
    // Thresholds in Months (Data range ~4 to 20+)
    if (ageInMonths < 8) return "Young Crab - Tender meat, but best left to grow.";
    if (ageInMonths < 12) return "Prime Adult Crab - Perfect balance of flavor.";
    return "Mature Giant - The prize of the ocean, rich and substantial.";
}
