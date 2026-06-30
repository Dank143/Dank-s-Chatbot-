import { $ } from './state.js';

export function loadTheme() {
  applyTheme(localStorage.getItem('theme') || 'light');
}

export function applyTheme(theme) {
  const isLight = theme === 'light';
  document.body.classList.toggle('light', isLight);
  $('themeIconSun').style.display  = isLight ? 'block' : 'none';
  $('themeIconMoon').style.display = isLight ? 'none'  : 'block';
  $('hljsDark').disabled  = isLight;
  $('hljsLight').disabled = !isLight;
}

export function toggleTheme() {
  const isLight = document.body.classList.contains('light');
  const next = isLight ? 'dark' : 'light';
  localStorage.setItem('theme', next);
  applyTheme(next);
}
