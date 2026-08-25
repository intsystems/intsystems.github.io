// Home page photo carousel: CSS scroll-snap + small controller.
// Replaces bulma-carousel (no external dependencies). Behavior kept:
// one slide at a time, prev/next arrows, autoplay every 5s, loop,
// swipe on touch devices (native scrolling).
document.addEventListener('DOMContentLoaded', function () {
    var carousel = document.querySelector('#carousel-demo');
    if (!carousel) return;
    var items = carousel.querySelectorAll('.carousel-item');
    if (items.length < 2) return;

    // Wrapper to position the arrows over the carousel
    var wrap = document.createElement('div');
    wrap.className = 'carousel-wrap';
    carousel.parentNode.insertBefore(wrap, carousel);
    wrap.appendChild(carousel);

    function makeButton(direction, label) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'carousel-nav carousel-nav--' + direction;
        btn.setAttribute('aria-label', label);
        return btn;
    }
    var prev = makeButton('prev', 'Previous slide');
    var next = makeButton('next', 'Next slide');
    wrap.appendChild(prev);
    wrap.appendChild(next);

    var index = 0;

    function goTo(i, smooth) {
        index = (i + items.length) % items.length;
        carousel.scrollTo({
            left: index * carousel.clientWidth,
            behavior: smooth === false ? 'auto' : 'smooth'
        });
    }

    prev.addEventListener('click', function () { goTo(index - 1); restart(); });
    next.addEventListener('click', function () { goTo(index + 1); restart(); });

    // Keep index in sync with manual swipes
    var scrollTimer;
    carousel.addEventListener('scroll', function () {
        clearTimeout(scrollTimer);
        scrollTimer = setTimeout(function () {
            index = Math.round(carousel.scrollLeft / carousel.clientWidth);
        }, 100);
    }, { passive: true });

    // Keep the current slide aligned when the window is resized
    window.addEventListener('resize', function () { goTo(index, false); });

    // Autoplay (5s, as before), paused on hover/touch and
    // disabled entirely for prefers-reduced-motion users
    var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var timer = null;

    function start() {
        if (reducedMotion || timer) return;
        timer = setInterval(function () { goTo(index + 1); }, 5000);
    }
    function stop() {
        clearInterval(timer);
        timer = null;
    }
    function restart() { stop(); start(); }

    wrap.addEventListener('mouseenter', stop);
    wrap.addEventListener('mouseleave', start);
    carousel.addEventListener('touchstart', stop, { passive: true });
    carousel.addEventListener('touchend', start, { passive: true });
    start();
});
