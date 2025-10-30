const parseNumber = (value, fallback) => {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const tocRoot = document.querySelector('[data-toc]');

if (tocRoot) {
  const config = {
    scrollThreshold: parseNumber(tocRoot.dataset.scrollThreshold, 0),
    hideDelay: parseNumber(tocRoot.dataset.hideDelay, 1600),
    headingSelector: tocRoot.dataset.headingSelector || 'h2,h3',
    tocDepth: parseNumber(tocRoot.dataset.tocDepth, null),
    sticky: tocRoot.dataset.tocSticky === 'true',
    contentSelector: tocRoot.dataset.tocContentSelector || '[data-toc-content]',
    velocityDistance: Math.max(parseNumber(tocRoot.dataset.velocityDistance, 360), 24),
    velocityWindow: Math.max(parseNumber(tocRoot.dataset.velocityWindow, 480), 120)
  };

  const contentRoot = document.querySelector(config.contentSelector);

  if (!contentRoot) {
    tocRoot.remove();
    // eslint-disable-next-line no-console
    console.warn('[toc] content container not found for selector', config.contentSelector);
  } else {
    const headings = collectHeadings(contentRoot, config.headingSelector, config.tocDepth);

    if (!headings.length) {
      renderEmptyMessage(tocRoot);
      tocRoot.hidden = false;
    } else {
    assignMissingIds(headings);
      buildTocList(tocRoot, headings);
      applyScrollableState(tocRoot);
      bindBehaviour(tocRoot, headings, config);
      tocRoot.hidden = false;
    }
  }
}

function collectHeadings(root, selector, depthLimit) {
  const selectors = selector
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);

  if (!selectors.length) {
    return [];
  }

  const rawHeadings = Array.from(root.querySelectorAll(selectors.join(',')));

  const validHeadings = rawHeadings.filter((heading) => {
    if (heading.dataset.tocSkip === 'true') {
      return false;
    }
    const level = extractHeadingLevel(heading);
    if (!Number.isFinite(level)) {
      return false;
    }

    const text = heading.dataset.tocLabel || heading.textContent;
    if (!text || !text.trim()) {
      return false;
    }

    return true;
  });

  if (!validHeadings.length) {
    return [];
  }

  const baseLevel = validHeadings.reduce((carry, heading) => Math.min(carry, extractHeadingLevel(heading)), Infinity);
  const maxDepth = Number.isInteger(depthLimit) ? baseLevel + depthLimit - 1 : Infinity;

  return validHeadings.filter((heading) => extractHeadingLevel(heading) <= maxDepth);
}

function renderEmptyMessage(tocElement) {
  const body = tocElement.querySelector('[data-toc-body]');
  if (!body) {
    return;
  }

  const message = document.createElement('p');
  message.className = 'toc__empty';
  message.textContent = tocElement.getAttribute('aria-label') || 'Contents';
  body.innerHTML = '';
  body.appendChild(message);
}

function assignMissingIds(headings) {
  const slugMap = new Map();

  headings.forEach((heading) => {
    if (heading.id) {
      return;
    }

    const base = slugify(heading.dataset.tocLabel || heading.textContent.trim());
    const count = slugMap.get(base) || 0;
    slugMap.set(base, count + 1);
    heading.id = count === 0 ? base : `${base}-${count + 1}`;
  });
}

function slugify(text) {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/[\s_-]+/g, '-');
}

function buildTocList(tocElement, headings) {
  const listRoot = tocElement.querySelector('[data-toc-list]');
  if (!listRoot) {
    return;
  }

  listRoot.innerHTML = '';
  const baseLevel = headings.reduce((minLevel, heading) => Math.min(minLevel, extractHeadingLevel(heading)), Infinity);
  let currentLevel = baseLevel;
  let currentList = listRoot;
  const listStack = [listRoot];
  let previousLevel = baseLevel;

  headings.forEach((heading) => {
    const rawLevel = Math.max(extractHeadingLevel(heading), baseLevel);
    let level = rawLevel;

    if (level > previousLevel + 1) {
      level = previousLevel + 1;
    }

    if (level > currentLevel) {
      for (let i = currentLevel; i < level; i += 1) {
        const parentItem = currentList.lastElementChild;
        if (!parentItem) {
          break;
        }
        const nestedList = document.createElement('ul');
        parentItem.appendChild(nestedList);
        currentList = nestedList;
        listStack.push(currentList);
      }
    } else if (level < currentLevel) {
      const stepsUp = Math.min(listStack.length - 1, currentLevel - level);
      for (let i = 0; i < stepsUp; i += 1) {
        listStack.pop();
      }
      currentList = listStack[listStack.length - 1] || listRoot;
    }

    currentLevel = level;
    previousLevel = level;

    const listItem = document.createElement('li');
    const link = document.createElement('a');
    const label = heading.dataset.tocLabel || heading.textContent.trim();
    link.className = `toc__link toc__link--level-${level}`;
    link.href = `#${heading.id}`;
    link.textContent = label;
    link.setAttribute('data-toc-target', heading.id);
    listItem.appendChild(link);
    currentList.appendChild(listItem);
  });
}

function bindBehaviour(tocElement, headings, cfg) {
  const links = Array.from(tocElement.querySelectorAll('[data-toc-target]'));
  if (!links.length) {
    return;
  }

  const headingsById = new Map(headings.map((heading) => [heading.id, heading]));
  let hideTimer = null;
  let visible = cfg.sticky;
  let lastScrollTop = window.scrollY || window.pageYOffset;
  let lastScrollTime = performance.now();
  let lastDirection = 0;
  let burstDistance = 0;
  let burstStartTime = lastScrollTime;
  const minDistance = cfg.velocityDistance;
  const windowMs = cfg.velocityWindow;

  const clearHideTimer = () => {
    if (hideTimer) {
      window.clearTimeout(hideTimer);
      hideTimer = null;
    }
  };

  const hideToc = () => {
    if (cfg.sticky) {
      return;
    }
    clearHideTimer();
    tocElement.classList.remove('is-visible');
    visible = false;
  };

  const scheduleHide = () => {
    if (cfg.sticky) {
      return;
    }
    clearHideTimer();
    hideTimer = window.setTimeout(hideToc, cfg.hideDelay);
  };

  const showToc = () => {
    if (!visible) {
      tocElement.classList.add('is-visible');
      visible = true;
    }
  };

  const updateActiveLink = () => {
    const activeId = detectActiveHeading(headings);
    links.forEach((link) => {
      const isActive = link.dataset.tocTarget === activeId;
      link.classList.toggle('toc__link--active', isActive);
      if (isActive) {
        link.setAttribute('aria-current', 'true');
      } else {
        link.removeAttribute('aria-current');
      }
    });
  };

  const resetActiveLinks = () => {
    links.forEach((link) => {
      link.classList.remove('toc__link--active');
      link.removeAttribute('aria-current');
    });
  };

  const onScroll = throttle(() => {
    const scrollTop = window.scrollY || window.pageYOffset;
    const now = performance.now();
  const delta = scrollTop - lastScrollTop;
  let direction = delta === 0 ? 0 : delta > 0 ? 1 : -1;

    if (direction === 0) {
      if (now - burstStartTime > windowMs) {
        burstDistance = 0;
        lastDirection = 0;
      }
    } else if (direction !== lastDirection || now - burstStartTime > windowMs) {
      burstDistance = Math.abs(delta);
      burstStartTime = now;
      lastDirection = direction;
    } else {
      burstDistance += Math.abs(delta);
    }

    const burstDuration = now - burstStartTime;
    const passedThreshold = scrollTop > cfg.scrollThreshold || direction < 0;
    const quickBurst =
      direction !== 0 && burstDistance >= minDistance && burstDuration <= windowMs && passedThreshold;

    if (quickBurst) {
      showToc();
      scheduleHide();
      updateActiveLink();
      burstDistance = 0;
      burstStartTime = now;
      lastDirection = 0;
      direction = 0;
    } else if (!cfg.sticky && scrollTop <= cfg.scrollThreshold) {
      hideToc();
      resetActiveLinks();
    } else if (visible) {
      updateActiveLink();
    }

    lastDirection = direction;
    lastScrollTop = scrollTop;
    lastScrollTime = now;
  });

  window.addEventListener('scroll', onScroll, { passive: true });

  const onResize = throttle(() => {
    updateActiveLink();
    applyScrollableState(tocElement);
  }, 120);

  window.addEventListener('resize', onResize);

  tocElement.addEventListener('mouseenter', () => {
    showToc();
    clearHideTimer();
    updateActiveLink();
  });

  tocElement.addEventListener('focusin', () => {
    showToc();
    clearHideTimer();
    updateActiveLink();
  });

  tocElement.addEventListener('mouseleave', scheduleHide);
  tocElement.addEventListener('focusout', (event) => {
    if (!tocElement.contains(event.relatedTarget)) {
      scheduleHide();
    }
  });

  links.forEach((link) => {
    link.addEventListener('click', (event) => {
      const targetId = link.dataset.tocTarget;
      const targetHeading = headingsById.get(targetId);
      if (!targetHeading) {
        return;
      }

      event.preventDefault();
      const smooth = window.matchMedia('(prefers-reduced-motion: no-preference)').matches;
      targetHeading.scrollIntoView({
        behavior: smooth ? 'smooth' : 'auto',
        block: 'start'
      });
      updateActiveLink();
      scheduleHide();
    });
  });

  if (cfg.sticky) {
    tocElement.classList.add('is-pinned');
    updateActiveLink();
  } else {
    onScroll();
  }

  const body = tocElement.querySelector('[data-toc-body]');
  if (body && typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(() => applyScrollableState(tocElement));
    observer.observe(body);
  }
}

function detectActiveHeading(headings) {
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
  let bestId = null;
  let bestVisibility = 0;

  headings.forEach((heading) => {
    const rect = heading.getBoundingClientRect();
    if (rect.bottom <= 0 || rect.top >= viewportHeight) {
      return;
    }

    const visibleTop = Math.max(rect.top, 0);
    const visibleBottom = Math.min(rect.bottom, viewportHeight);
    const visibleHeight = Math.max(0, visibleBottom - visibleTop);
    const visibilityRatio = visibleHeight / Math.max(rect.height, 1);

    if (visibilityRatio > bestVisibility) {
      bestVisibility = visibilityRatio;
      bestId = heading.id;
    }
  });

  if (bestId) {
    return bestId;
  }

  const scrollTop = window.scrollY || window.pageYOffset;
  for (let index = headings.length - 1; index >= 0; index -= 1) {
    if (headings[index].offsetTop <= scrollTop + 80) {
      return headings[index].id;
    }
  }

  return headings[0]?.id || null;
}

function extractHeadingLevel(heading) {
  const match = /H(\d)/.exec(heading.tagName);
  return match ? Number.parseInt(match[1], 10) : NaN;
}

function throttle(callback, wait = 80) {
  let ticking = false;
  let lastArgs = [];

  return function throttled(...args) {
    lastArgs = args;
    if (!ticking) {
      window.requestAnimationFrame(() => {
        callback.apply(this, lastArgs);
        ticking = false;
      });
      ticking = true;
    }
  };
}

function applyScrollableState(tocElement) {
  const body = tocElement.querySelector('[data-toc-body]');
  if (!body) {
    return;
  }

  const epsilon = 4;
  const hasScroll = body.scrollHeight - body.clientHeight > epsilon;
  tocElement.classList.toggle('has-scroll', hasScroll);
}
