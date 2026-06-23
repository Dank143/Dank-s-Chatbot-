(function() {
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
