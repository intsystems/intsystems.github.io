// Предотвращение белого flash при сохранении плавных переходов
(function() {
  'use strict';
  
  // Страница видна сразу (без opacity: 0)
  document.documentElement.style.visibility = 'visible';
  document.body.style.visibility = 'visible';
  document.body.style.opacity = '1';
  
  // Плавные переходы обрабатываются через CSS анимацию .content
  // Не нужно ничего делать в JS
  
})();
