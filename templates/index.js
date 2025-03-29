// Show the popup
function showPopup() {
    document.getElementById("successPopup").style.display = "flex";
    setTimeout(closePopup, 3000); // Close the popup after 3 seconds
}

// Close the popup
function closePopup() {
    document.getElementById("successPopup").style.display = "none";
}

// Handle form submission
document.getElementById("contactForm").addEventListener("submit", function(event) {
    event.preventDefault();  // Prevent form from submitting
    showPopup();  // Show success popup
    // Optionally, you can add AJAX code here to submit the form data to the server
});
