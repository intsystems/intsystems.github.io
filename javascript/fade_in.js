// Fade-in on scroll effect using Intersection Observer
(function() {
  'use strict';
  
  // Page fade-in on load
  function initPageFadeIn() {
    // Check if user prefers reduced motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    
    if (prefersReducedMotion) {
      // Skip animation for users who prefer reduced motion
      document.body.classList.add('page-loaded');
    } else {
      // Add class after a tiny delay to ensure styles are loaded
      requestAnimationFrame(function() {
        document.body.classList.add('page-loaded');
      });
    }
  }
  
  // Force scroll to top on page load/refresh
  if ('scrollRestoration' in history) {
    history.scrollRestoration = 'manual';
  }
  
  // Configuration
  const config = {
    threshold: 0.1, // Элемент должен быть виден хотя бы на 10%
    rootMargin: '0px 0px -100px 0px' // Триггер за 100px до края экрана
  };

  // Create observer
  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        // Add visible class when element is in viewport
        entry.target.classList.add('visible');
        // Stop observing this element
        observer.unobserve(entry.target);
      }
    });
  }, config);

  // Observe all elements with fade-in classes
  function initFadeIn() {
    const selectors = [
      '.fade-in-section',
      '.fade-in-left',
      '.fade-in-right',
      '.fade-in-up',
      '.fade-in-down',
      '.fade-in-scale'
    ];
    
    selectors.forEach(function(selector) {
      const elements = document.querySelectorAll(selector);
      elements.forEach(function(element) {
        observer.observe(element);
      });
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      initPageFadeIn();
      initFadeIn();
    });
  } else {
    initPageFadeIn();
    initFadeIn();
  }
  
  // Scroll to top after page is fully loaded (including images and styles)
  window.addEventListener('load', function() {
    // Small delay to ensure all layout shifts are complete
    setTimeout(function() {
      window.scrollTo(0, 0);
    }, 0);
  });
})();
