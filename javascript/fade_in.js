// Fade-in on scroll effect using Intersection Observer
(function() {
  'use strict';
  
  // Page fade-in on load (отключено - вызывало белый flash)
  function initPageFadeIn() {
    // Больше не используется
  }
  
  // Плавная прокрутка к якорям вместо принудительного скролла вверх
  if ('scrollRestoration' in history) {
    history.scrollRestoration = 'auto';  // Изменено с 'manual' на 'auto'
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
      initFadeIn();
    });
  } else {
    initFadeIn();
  }
})();
