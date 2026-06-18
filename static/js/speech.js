const micBtn = document.getElementById('micBtn');
const input  = document.getElementById('messageInput');

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
  // browser doesn't support — keep button hidden
} else {
  micBtn.style.display = '';

  const rec = new SpeechRecognition();
  rec.continuous      = true;
  rec.interimResults  = true;
  rec.lang            = '';        // auto-detect from browser locale

  let running      = false;
  let baseText     = '';           // text already committed before this session

  rec.onresult = (e) => {
    let interim = '';
    let final   = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) final += t;
      else                       interim += t;
    }
    baseText += final;
    input.value = (baseText + interim).trim();
    input.dispatchEvent(new Event('input'));
  };

  rec.onerror = (e) => {
    if (e.error === 'no-speech') return;
    stop();
  };

  rec.onend = () => {
    if (running) rec.start();   // restart on silence to keep continuous
  };

  function start() {
    running  = true;
    baseText = input.value ? input.value + ' ' : '';
    rec.start();
    micBtn.classList.add('recording');
    micBtn.title = 'Stop recording';
  }

  function stop() {
    running = false;
    rec.stop();
    micBtn.classList.remove('recording');
    micBtn.title = 'Speech to text';
  }

  micBtn.addEventListener('click', () => running ? stop() : start());
}
