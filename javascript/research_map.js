// Research map (/materials/research-map/, markup in _includes/research_map.html).
// One selection at a time: a direction, a topic chip, an advisor hub or a single work (a point).
// The selection highlights matching points, shows the hub edges of those works
// and filters the list of works below the map; clicking it again or Escape resets.
// Without a selection the list is replaced by a hint (without JS it stays fully visible);
// a single selected work also shows its abstract.

const mapRoot = document.querySelector('[data-research-map]');

if (mapRoot) {
  const nodes = [...mapRoot.querySelectorAll('.research-map__nodes .research-map__node')]; // not the legend
  const edges = [...mapRoot.querySelectorAll('.research-map__edge')];
  const hubs = [...mapRoot.querySelectorAll('.research-map__hub')];
  const labels = [...mapRoot.querySelectorAll('.research-map__label')];
  const chips = [...mapRoot.querySelectorAll('.research-map__chip')];
  const directionButtons = [...mapRoot.querySelectorAll('[data-direction]')];
  const rows = [...mapRoot.querySelectorAll('.research-map__work')];
  const openBlocks = [...mapRoot.querySelectorAll('[data-open-topic]')];
  const abstracts = [...mapRoot.querySelectorAll('[data-abstract]')];
  const panelTitle = mapRoot.querySelector('[data-panel-title]');
  const panelCount = mapRoot.querySelector('[data-panel-count]');
  const panelHint = mapRoot.querySelector('[data-panel-hint]');
  const defaultTitle = panelTitle.textContent;
  let selection = null; // { kind: 'direction' | 'topic' | 'advisor' | 'work', id, title, topics? }

  const matches = (el) => {
    if (!selection) return true;
    if (selection.kind === 'work') return el.dataset.id === selection.id;
    if (selection.kind === 'direction') return el.dataset.topics.split(' ').some((t) => selection.topics.includes(t));
    const ids = selection.kind === 'topic' ? el.dataset.topics : el.dataset.advisors;
    return ids.split(' ').includes(selection.id);
  };

  const render = () => {
    mapRoot.classList.toggle('is-selected', Boolean(selection));
    const active = new Set(nodes.filter((n) => selection && matches(n)).map((n) => n.dataset.id));
    nodes.forEach((n) => n.classList.toggle('is-active', active.has(n.dataset.id)));

    const activeHubs = new Set();
    edges.forEach((e) => {
      const on = active.has(e.dataset.work)
        && (selection.kind !== 'advisor' || e.dataset.advisor === selection.id);
      e.classList.toggle('is-active', on);
      if (on) activeHubs.add(e.dataset.advisor);
    });
    hubs.forEach((h) => {
      const on = activeHubs.has(h.dataset.advisor);
      h.classList.toggle('is-active', on);
      h.setAttribute('aria-pressed', String(selection?.kind === 'advisor' && h.dataset.advisor === selection.id));
    });
    const activeTopics = selection?.kind === 'topic' ? [selection.id] : selection?.topics || [];
    labels.forEach((l) => l.classList.toggle('is-active', activeTopics.includes(l.dataset.topic)));
    directionButtons.forEach((d) => d.setAttribute('aria-pressed', String(selection?.kind === 'direction' && d.dataset.direction === selection.id)));
    chips.forEach((c) => {
      const pressed = selection ? selection.kind === 'topic' && c.dataset.topic === selection.id : 'reset' in c.dataset;
      c.setAttribute('aria-pressed', String(pressed));
    });
    openBlocks.forEach((b) => { b.hidden = !(selection?.kind === 'topic' && b.dataset.openTopic === selection.id); });

    let shown = 0;
    rows.forEach((r) => {
      r.hidden = !selection || !matches(r);
      if (!r.hidden) shown += 1;
    });
    panelHint.hidden = Boolean(selection);
    abstracts.forEach((el) => { el.hidden = selection?.kind !== 'work'; });
    panelTitle.textContent = selection?.title || defaultTitle;
    panelCount.textContent = selection ? shown : rows.length;
  };

  const select = (kind, id, title, topics) => {
    const same = selection && selection.kind === kind && selection.id === id;
    selection = same ? null : { kind, id, title, topics };
    render();
  };

  chips.forEach((chip) => chip.addEventListener('click', () => {
    if ('reset' in chip.dataset) {
      selection = null;
      render();
    } else {
      select('topic', chip.dataset.topic, chip.firstChild.textContent.trim());
    }
  }));

  directionButtons.forEach((button) => button.addEventListener('click', () => {
    select('direction', button.dataset.direction, button.firstChild.textContent.trim(), button.dataset.directionTopics.split(' '));
  }));

  hubs.forEach((hub) => {
    const choose = () => select('advisor', hub.dataset.advisor, hub.dataset.name);
    hub.addEventListener('click', choose);
    hub.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        choose();
      }
    });
  });

  // Titles in the list are buttons: the keyboard way to select one work and see its abstract.
  rows.forEach((row) => row.querySelector('.research-map__work-title').addEventListener('click', (event) => {
    select('work', row.dataset.id, event.currentTarget.textContent);
  }));

  nodes.forEach((node) => node.addEventListener('click', () => {
    const row = rows.find((r) => r.dataset.id === node.dataset.id);
    select('work', node.dataset.id, row?.querySelector('.research-map__work-title')?.textContent);
  }));

  render();

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && selection) {
      selection = null;
      render();
    }
  });

  // Zoom and pan through the SVG viewBox. Labels, hubs and strokes keep their screen size and
  // points grow only with sqrt(zoom), so zooming in pulls crowded clusters apart.
  // The page keeps scrolling with a plain wheel or a one-finger swipe: zoom with Ctrl/Cmd + wheel
  // (a trackpad pinch arrives as Ctrl + wheel), the buttons, or a pinch once the map is zoomed in.
  const svg = mapRoot.querySelector('.research-map__svg');
  const hint = mapRoot.querySelector('[data-zoom-hint]');
  const [, , width, height] = svg.getAttribute('viewBox').split(' ').map(Number);
  const MAX_ZOOM = 8;
  const baseRadius = new Map(nodes.map((n) => [n, Number(n.getAttribute('r'))]));
  let view = { x: 0, y: 0, w: width, h: height };

  const zoomLevel = () => width / view.w;

  const applyView = () => {
    const k = zoomLevel();
    svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`);
    // Text and hubs are divided by zoom x on-screen scale of the SVG, so they stay ~13px on any
    // screen; the scale is floored at 0.5 so labels do not swamp the map on a phone at 1x.
    mapRoot.style.setProperty('--rm-zoom', k * Math.max(svg.clientWidth / width, 0.5));
    mapRoot.classList.toggle('is-zoomed', k > 1.001);
    nodes.forEach((n) => n.setAttribute('r', (baseRadius.get(n) / Math.sqrt(k)).toFixed(2)));
  };

  // Zoom by `factor` keeping the user-space point (cx, cy) under the same screen position.
  const zoomAt = (factor, cx = view.x + view.w / 2, cy = view.y + view.h / 2) => {
    const k = Math.min(MAX_ZOOM, Math.max(1, zoomLevel() * factor));
    const w = width / k;
    const h = height / k;
    view = {
      x: Math.min(width - w, Math.max(0, cx - ((cx - view.x) * w) / view.w)),
      y: Math.min(height - h, Math.max(0, cy - ((cy - view.y) * h) / view.h)),
      w,
      h,
    };
    applyView();
  };

  const toUser = (clientX, clientY) => {
    const box = svg.getBoundingClientRect();
    return {
      x: view.x + ((clientX - box.left) / box.width) * view.w,
      y: view.y + ((clientY - box.top) / box.height) * view.h,
    };
  };

  mapRoot.querySelectorAll('[data-zoom]').forEach((button) => button.addEventListener('click', () => {
    if (button.dataset.zoom === 'reset') {
      view = { x: 0, y: 0, w: width, h: height };
      applyView();
    } else {
      zoomAt(button.dataset.zoom === 'in' ? 1.6 : 1 / 1.6);
    }
  }));

  window.addEventListener('resize', applyView);
  applyView();

  let hintTimer = null;
  svg.addEventListener('wheel', (event) => {
    if (!event.ctrlKey && !event.metaKey) {
      hint.classList.add('is-visible');
      clearTimeout(hintTimer);
      hintTimer = setTimeout(() => hint.classList.remove('is-visible'), 1200);
      return;
    }
    event.preventDefault();
    const p = toUser(event.clientX, event.clientY);
    // A trackpad pinch sends small deltas, a mouse wheel notch ~100: cap one step at ~1.65x.
    zoomAt(Math.exp(-Math.max(-50, Math.min(50, event.deltaY)) * 0.01), p.x, p.y);
  }, { passive: false });

  // Drag to pan (mouse: any zoom level above 1; touch: only when zoomed, see touch-action in CSS).
  // A drag longer than a few pixels swallows the click that follows, so it does not select a point.
  const pointers = new Map();
  let drag = null;
  let pinch = null;
  let suppressClick = false;

  svg.addEventListener('pointerdown', (event) => {
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    suppressClick = false; // a flag left by a gesture that produced no click must not eat this one
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y), zoom: zoomLevel() };
      drag = null; // a drag that turns into a pinch ends here
      suppressClick = true;
      mapRoot.classList.remove('is-dragging');
    } else if (zoomLevel() > 1.001) {
      drag = { x: event.clientX, y: event.clientY, view: { ...view }, moved: false, id: event.pointerId };
    }
  });

  svg.addEventListener('pointermove', (event) => {
    if (!pointers.has(event.pointerId)) return;
    if (event.pointerType === 'mouse' && event.buttons === 0) {
      endPointer(event); // the button was released outside the window before the drag started
      return;
    }
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const p = toUser((a.x + b.x) / 2, (a.y + b.y) / 2);
      zoomAt((pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y)) / pinch.dist / zoomLevel(), p.x, p.y);
      return;
    }
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    if (!drag.moved) {
      drag.moved = true;
      svg.setPointerCapture(drag.id); // only now, so that a plain click still reaches its target
      mapRoot.classList.add('is-dragging');
    }
    const box = svg.getBoundingClientRect();
    view.x = Math.min(width - view.w, Math.max(0, drag.view.x - (dx / box.width) * view.w));
    view.y = Math.min(height - view.h, Math.max(0, drag.view.y - (dy / box.height) * view.h));
    applyView();
  });

  function endPointer(event) {
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinch = null;
    if (drag && drag.id === event.pointerId) {
      suppressClick = drag.moved;
      drag = null;
      mapRoot.classList.remove('is-dragging');
    }
  }
  // On window: before pointer capture starts, a release outside the SVG would not reach it.
  window.addEventListener('pointerup', endPointer);
  window.addEventListener('pointercancel', endPointer);

  svg.addEventListener('click', (event) => {
    if (suppressClick) {
      event.stopPropagation();
      suppressClick = false;
    }
  }, true);
}
