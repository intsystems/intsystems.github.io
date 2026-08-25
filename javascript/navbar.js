// Navbar behavior: mobile menu collapse + dropdowns.
// Replaces Bootstrap 3 collapse.js / dropdown.js (and the jQuery dependency).
(function () {
  'use strict';

  function closeAllDropdowns() {
    document.querySelectorAll('.dropdown.open, .nav-item.open').forEach(function (el) {
      el.classList.remove('open');
      var toggle = el.querySelector('[data-toggle="dropdown"]');
      if (toggle) toggle.setAttribute('aria-expanded', 'false');
    });
  }

  function init() {
    // Mobile menu toggle (button with data-toggle="collapse")
    document.querySelectorAll('[data-toggle="collapse"]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = document.querySelector(btn.getAttribute('data-target'));
        if (target) target.classList.toggle('in');
      });
    });

    // Dropdowns (click toggles, click elsewhere / Escape closes)
    document.querySelectorAll('[data-toggle="dropdown"]').forEach(function (toggle) {
      toggle.addEventListener('click', function (event) {
        event.preventDefault();
        event.stopPropagation();
        var parent = toggle.parentElement;
        var wasOpen = parent.classList.contains('open');
        closeAllDropdowns();
        if (!wasOpen) {
          parent.classList.add('open');
          toggle.setAttribute('aria-expanded', 'true');
        }
      });
    });

    document.addEventListener('click', closeAllDropdowns);
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') closeAllDropdowns();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
