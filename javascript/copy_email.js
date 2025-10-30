// Function to copy email to clipboard and show tooltip
function copyEmailToClipboard(event, email) {
    event.preventDefault();
    
    // Detect language from page
    const isRussian = document.documentElement.lang === 'ru' || window.location.pathname.includes('/ru/');
    const message = isRussian ? 'Скопировано в буфер обмена!' : 'Copied to clipboard!';
    
    // Copy to clipboard
    navigator.clipboard.writeText(email).then(function() {
        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.textContent = message;
        tooltip.style.cssText = `
            position: fixed;
            left: ${event.clientX}px;
            top: ${event.clientY - 40}px;
            background-color: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: "Open Sans", sans-serif;
            font-size: 14px;
            z-index: 10000;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        
        document.body.appendChild(tooltip);
        
        // Fade in
        setTimeout(() => {
            tooltip.style.opacity = '1';
        }, 10);
        
        // Fade out and remove
        setTimeout(() => {
            tooltip.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(tooltip);
            }, 300);
        }, 2000);
    }).catch(function(err) {
        console.error('Failed to copy email: ', err);
    });
    
    return false;
}
