
document.addEventListener('DOMContentLoaded', function() {
    const tocLinks = document.querySelectorAll('.toc-link');
    const sections = document.querySelectorAll('section[id]'); // Targets sections with IDs
    
    function removeActiveClasses() {
        tocLinks.forEach(link => link.classList.remove('active'));
    }
    
    function setActiveLink(id) {
        removeActiveClasses();
        const activeLink = document.querySelector(`.toc-link[href="#${id}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                setActiveLink(entry.target.id);
            }
        });
    }, {
        root: null,
        rootMargin: '-20% 0px -70% 0px',
        threshold: 0
    });
    
    sections.forEach(section => observer.observe(section));
});