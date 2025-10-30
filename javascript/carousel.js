document.addEventListener('DOMContentLoaded', function() {
    // Проверяем, существует ли элемент карусели на странице
    const carouselElement = document.querySelector('#carousel-demo');
    
    if (carouselElement) {
        // Инициализируем карусель без динамической загрузки CSS
        bulmaCarousel.attach('#carousel-demo', {
            slidesToScroll: 1,
            slidesToShow: 1,
            autoplay: true,
            autoplaySpeed: 5000,
            loop: true,
            pagination: false,
            navigation: true,
            navigationSwipe: true,
            breakpoints: [
                { changePoint: 768, slidesToShow: 1, slidesToScroll: 1 },
                { changePoint: 1024, slidesToShow: 1, slidesToScroll: 1 }
            ]
        });
    }
});