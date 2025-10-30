document.addEventListener('DOMContentLoaded', () => {
  const tables = document.querySelectorAll('table');

  tables.forEach((table) => {
    if (table.closest('.table-scroll') || table.dataset.noScroll === 'true') {
      return;
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'table-scroll';
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');

    const ariaLabel = table.getAttribute('aria-label')
      || table.querySelector('caption')?.textContent
      || 'Scrollable data table';

    wrapper.setAttribute('aria-label', ariaLabel.trim());

    table.parentNode?.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });
});
