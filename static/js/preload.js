(function() {
  if ((localStorage.getItem('theme') || 'light') === 'light') {
    document.documentElement.classList.add('light');
    const hljsDark = document.getElementById('hljsDark');
    const hljsLight = document.getElementById('hljsLight');
    if (hljsDark) hljsDark.disabled = true;
    if (hljsLight) hljsLight.disabled = false;
  }

  const isSidebarCollapsed = localStorage.getItem('sidebarCollapsed') === '1';
  const sidebar = document.getElementById('sidebar');
  if (isSidebarCollapsed && sidebar) {
    sidebar.classList.add('collapsed');
    sidebar.style.transition = 'none !important';
    setTimeout(() => {
      sidebar.style.transition = '';
    }, 50);
  }
})();
