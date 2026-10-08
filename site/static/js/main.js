document.addEventListener('DOMContentLoaded', () => {
    if (typeof mediumZoom === 'function') {
        mediumZoom('[data-zoomable]', { background: 'rgba(255, 255, 255, 0.96)', margin: 24 });
    }
});
