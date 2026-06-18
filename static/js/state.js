export const state = {
  chats: [],
  activeChatId: null,
  provider: localStorage.getItem('provider') || null,
  modelsNim: [],
  modelsOllama: [],
  defaultModelNim: null,
  defaultModelOllama: null,
  models: [],
  selectedModel: null,
  defaultModel: null,
  streaming: false,
  abortController: null,
  pendingFiles: [],
  webSearch: false,
  debugMode: false,
  searchTriggers: [
    'search', 'lookup', 'google', 'bing', 'find', 'news', 'breaking', 'weather', 'trending',
    'look up', 'look for', 'find out', 'right now', 'as of now', 'as of today', 'at the moment',
    'this week', 'this month', 'this year',
    'how much is', 'how much does', 'how much are',
    "what's the price", 'what is the price', 'what time is it', 'what time in',
    'current price', 'current rate', 'current score', 'current standings',
    'stock price', 'live score', 'live results',
    'latest news', 'recent news', 'any news',
    'what are the latest', 'what is the latest', 'what happened', 'what has happened',
    'who won', 'who is winning', "today's", 'this morning', 'this afternoon', 'this evening',
  ],
  autoSearchDetect: localStorage.getItem('autoSearchDetect') !== '0',
  hasKeyNim: false,
  hasKeyOllama: false,
};

export const $ = (id) => document.getElementById(id);

export const starredList      = $('starredList');
export const recentList       = $('recentList');
export const starredLabel     = $('starredLabel');
export const recentsLabel     = $('recentsLabel');
export const messagesEl       = $('messages');
export const welcomeEl        = $('welcome');
export const inputAreaEl      = $('inputArea');
export const messageInput     = $('messageInput');
export const sendBtn          = $('sendBtn');
export const modelSelectorBtn = $('modelSelectorBtn');
export const modelSelectorLbl = $('modelSelectorLabel');
export const dropdownBackdrop = $('dropdownBackdrop');
export const dropdownList     = $('dropdownList');
export const modelSearch      = $('modelSearch');
export const chatTitleDisplay = $('chatTitleDisplay');
export const renameBtn        = $('renameBtn');
export const sidebar          = $('sidebar');
export const lightbox         = $('lightbox');
export const lightboxImg      = $('lightboxImg');

export function escHtml(str = '') {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
export const escAttr = escHtml;

export function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

export function autoResize() {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 220) + 'px';
}

export function updateSendBtn() {
  sendBtn.disabled = (!messageInput.value.trim() && state.pendingFiles.length === 0) || state.streaming;
}

export function setStopMode(on) {
  $('sendIcon').style.display = on ? 'none' : '';
  $('stopIcon').style.display = on ? '' : 'none';
  sendBtn.title = on ? 'Stop Generating' : 'Send';
  sendBtn.classList.toggle('stop-mode', on);
  sendBtn.disabled = false;
}

export function setWebSearch(on) {
  state.webSearch = on;
  const btn = $('webSearchToggle');
  if (!btn) return;
  btn.classList.toggle('active', on);
  btn.title = on ? 'Web Search: ON' : 'Web Search: OFF';
}

export function setAutoSearchDetect(on) {
  state.autoSearchDetect = on;
  localStorage.setItem('autoSearchDetect', on ? '1' : '0');
}

export function setDebugMode(on) {
  state.debugMode = on;
  const btn = $('debugToggleBtn');
  if (btn) {
    btn.classList.toggle('active', on);
    btn.title = on ? 'Debug Mode: ON' : 'Debug Mode: OFF';
  }
  document.querySelectorAll('.search-debug-panel').forEach(el => {
    el.style.display = on ? '' : 'none';
  });
}

export function needsWebSearch(text) {
  if (!state.autoSearchDetect || !state.searchTriggers.length) return false;
  const lower = text.toLowerCase();
  return state.searchTriggers.some(t => lower.includes(t));
}

// Client timestamp for date-aware replies.
export function clientTime() {
  return new Date().toLocaleString('en-US', {
    weekday: 'short', year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
  });
}

export function beginStreaming() {
  state.streaming = true;
  state.abortController = new AbortController();
  setStopMode(true);
}

export function endStreaming() {
  state.streaming = false;
  state.abortController = null;
  setStopMode(false);
}

export function setProvider(p) {
  state.provider = p;
  if (p) {
    localStorage.setItem('provider', p);
  } else {
    localStorage.removeItem('provider');
  }
  
  // Switch models to active provider's lists
  if (p === 'nim') {
    state.models = state.modelsNim;
    state.defaultModel = state.defaultModelNim;
  } else if (p === 'ollama') {
    state.models = state.modelsOllama;
    state.defaultModel = state.defaultModelOllama;
  } else {
    state.models = [];
    state.defaultModel = null;
  }
  
  // If selected model isn't in new provider, fallback to default
  if (!p || !state.models.find(m => m.id === state.selectedModel)) {
    state.selectedModel = state.defaultModel;
  }
  
  // Sync the UI pills
  document.querySelectorAll('.pill-opt').forEach(btn => {
    btn.classList.toggle('active', p && btn.dataset.provider === p);
  });
}
