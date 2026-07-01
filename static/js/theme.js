import { $ } from './state.js';

export function loadTheme() {
  applyTheme(localStorage.getItem('theme') || 'light');
}

export function applyTheme(theme) {
  const isLight = theme === 'light';
  document.documentElement.classList.toggle('light', isLight);
  $('themeIconSun').style.display  = isLight ? 'block' : 'none';
  $('themeIconMoon').style.display = isLight ? 'none'  : 'block';
  $('hljsDark').disabled  = isLight;
  $('hljsLight').disabled = !isLight;
}

export function toggleTheme() {
  const isLight = document.documentElement.classList.contains('light');
  const next = isLight ? 'dark' : 'light';
  localStorage.setItem('theme', next);
  applyTheme(next);
}
