document.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('laylaWidget');
  const panel = document.getElementById('laylaWindow');
  const input = document.getElementById('laylaIn');
  const messages = document.getElementById('laylaMsgs');
  const state = { responses: null };

  if (!root || !panel || !input || !messages) return;

  const addMessage = (text, type) => {
    const node = document.createElement('div');
    node.className = `msg ${type}`;
    node.textContent = text;
    messages.appendChild(node);
    messages.scrollTop = messages.scrollHeight;
  };

  const findReply = (query) => {
    const fallback = 'Анализирую запрос. В открытых базах совпадений не найдено. Рекомендуется ручная проверка специалистом.';
    if (!state.responses) return fallback;
    const normalized = query.toLowerCase();
    const item = Object.values(state.responses).find(entry =>
      entry.keys.some(key => normalized.includes(key.toLowerCase()))
    );
    return item?.text || fallback;
  };

  const send = () => {
    const value = input.value.trim();
    if (!value) return;
    addMessage(value, 'user');
    input.value = '';
    window.setTimeout(() => addMessage(findReply(value), 'ai'), 600);
  };

  const close = () => panel.classList.remove('active');
  const open = () => {
    panel.classList.add('active');
    window.setTimeout(() => input.focus(), 100);
  };

  fetch(panel.dataset.db || '/data/layla-db.json')
    .then(response => (response.ok ? response.json() : null))
    .then(data => { state.responses = data?.responses || null; })
    .catch(() => { state.responses = null; });

  document.getElementById('laylaTrigger')?.addEventListener('click', event => {
    event.stopPropagation();
    panel.classList.contains('active') ? close() : open();
  });

  document.getElementById('laylaClose')?.addEventListener('click', event => {
    event.stopPropagation();
    close();
  });

  document.getElementById('laylaSend')?.addEventListener('click', send);
  input.addEventListener('keydown', event => { if (event.key === 'Enter') send(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
  document.addEventListener('click', event => { if (!root.contains(event.target)) close(); });
});
