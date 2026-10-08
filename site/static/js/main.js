document.addEventListener('DOMContentLoaded', () => {
    if (typeof mediumZoom === 'function') {
        mediumZoom('[data-zoomable]', { background: 'rgba(255, 255, 255, 0.96)', margin: 24 });
    }

    // Highlight the contents entry for the section being read.
    const links = [...document.querySelectorAll('.toc a')];
    const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    const setActive = () => {
        const y = window.innerHeight * 0.35;
        let current = null;
        for (const s of sections) {
            if (s.getBoundingClientRect().top <= y) current = s;
        }
        links.forEach(a => a.classList.toggle('active', current !== null && a.getAttribute('href') === '#' + current.id));
    };
    window.addEventListener('scroll', setActive, { passive: true });
    setActive();
});
