(function () {
  'use strict';

  // Elements
  const btnSearch = document.getElementById('btn-search');
  const btnChatSend = document.getElementById('btn-chat-send');
  const qGlobal = document.getElementById('q-global');
  const chatInput = document.getElementById('chat-input');
  const chatMessages = document.getElementById('chat-messages');
  const fJurisdiction = document.getElementById('f-jurisdiction');
  const fType = document.getElementById('f-type');
  const fArticle = document.getElementById('f-article');
  const resultsArea = document.getElementById('results-area');
  const resultsCount = document.getElementById('results-count');

  if (!btnSearch || !resultsArea) return;

  const dataUrl = (window.DATALAW_CONFIG && window.DATALAW_CONFIG.libraryDataUrl) || '/data/legal_library.json';
  let db = [];
  let currentResults = [];

  // Load Data
  fetch(dataUrl)
    .then(r => r.json())
    .then(json => {
      db = Array.isArray(json) ? json : [];
      console.log('Legal Library loaded:', db.length, 'documents');
    })
    .catch(err => {
      console.error('Failed to load library:', err);
      appendMessage('ai', 'Ошибка загрузки базы данных. Пожалуйста, проверьте соединение.');
    });

  // Events
  btnSearch.addEventListener('click', handleMainSearch);
  qGlobal.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleMainSearch(); });
  
  btnChatSend.addEventListener('click', handleChat);
  chatInput.addEventListener('keypress', (e) => { 
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChat();
    }
  });

  // Main Logic
  function handleMainSearch() {
    const q = qGlobal.value.trim();
    if (!q) return;
    
    appendMessage('user', q);
    executeAnalysis(q);
    qGlobal.value = '';
  }

  function handleChat() {
    const q = chatInput.value.trim();
    if (!q) return;
    
    appendMessage('user', q);
    executeAnalysis(q);
    chatInput.value = '';
    chatInput.style.height = 'auto';
  }

  function executeAnalysis(query) {
    const jur = fJurisdiction.value;
    const type = fType.value;
    const art = fArticle.value.toLowerCase();
    
    // 1. Root-based matching for Russian
    const q = query.toLowerCase();
    const queryWords = q.split(/\s+/).filter(w => w.length > 2);
    const queryRoots = queryWords.map(w => w.slice(0, -2)); // Simple stemming

    const results = db.map(item => {
      let score = 0;
      const text = (item.text || '').toLowerCase();
      const title = (item.title || '').toLowerCase();
      const summary = (item.summary || '').toLowerCase();
      const keywords = (item.metadata && item.metadata.keywords ? item.metadata.keywords.join(' ') : '').toLowerCase();

      // Full phrase boost
      if (title.includes(q)) score += 100;
      if (summary.includes(q)) score += 60;
      if (text.includes(q)) score += 40;

      // Root matching
      queryRoots.forEach(root => {
        if (title.includes(root)) score += 20;
        if (summary.includes(root)) score += 15;
        if (text.includes(root)) score += 10;
        if (keywords.includes(root)) score += 25;
      });

      // Filters
      let match = true;
      if (jur && item.jurisdiction !== jur) match = false;
      if (type && item.type !== type) match = false;
      if (art && !(item.article || '').toLowerCase().includes(art)) match = false;

      return { ...item, score, match };
    })
    .filter(item => item.match && item.score > 0)
    .sort((a, b) => b.score - a.score);

    currentResults = results.slice(0, 10);
    
    // 2. UI Updates
    renderSidebar(currentResults);
    
    // 3. AI Synthesis
    setTimeout(() => {
      const response = generateAIResponse(query, currentResults);
      appendMessage('ai', response);
    }, 600);
  }

  function generateAIResponse(query, docs) {
    if (docs.length === 0) {
      return `Я проанализировала базу данных по вашему запросу «${query}», но не нашла прямых совпадений. Попробуйте использовать другие термины или проверьте фильтры.`;
    }

    const q = query.toLowerCase();
    const citations = (idx) => `<cite>${idx + 1}</cite>`;
    
    let synthesis = `На основании вашего запроса и анализа ${docs.length} найденных документов, представляю краткий синтез практики:<br><br>`;
    
    // Theme-based responses
    if (q.includes('пытк') || q.includes('бесчеловеч') || q.includes('статья 3')) {
      synthesis += `Вопрос запрета пыток и бесчеловечного обращения (Статья 3 Конвенции) является абсолютным правом. В найденных материалах ${citations(0)} подчеркивается, что никакие обстоятельства не могут оправдать нарушение этого запрета. `;
      if (docs[1]) synthesis += `Практика указывает на необходимость тщательного расследования любых заявлений о жестоком обращении со стороны представителей государства ${citations(1)}. `;
    } else if (q.includes('семь') || q.includes('частн') || q.includes('статья 8')) {
      synthesis += `Защита частной и семейной жизни (Статья 8) охватывает широкий спектр вопросов: от депортации до вмешательства в переписку. Суд требует соблюдения баланса между интересами общества и правами личности ${citations(0)}. `;
      if (docs[1]) synthesis += `Особое внимание уделяется сохранению связей между родителями и детьми в сложных трансграничных спорах ${citations(1)}. `;
    } else if (q.includes('экстрадиц') || q.includes('выдач') || q.includes('интерпол')) {
      synthesis += `В делах об экстрадиции и Интерполе ключевым является риск нарушения прав человека в запрашивающем государстве. Наличие «Красного уведомления» требует немедленного правового аудита на предмет политической мотивации ${citations(0)}. `;
    } else {
      synthesis += `Ключевые выводы по вашему запросу содержатся в деле <strong>${docs[0].case}</strong>, где рассматривался вопрос ${docs[0].article} ${citations(0)}. `;
      if (docs[1]) synthesis += `Также релевантна позиция по делу ${docs[1].case}, касающаяся юрисдикции ${docs[1].jurisdiction} ${citations(1)}. `;
    }

    synthesis += `<br><br>Фрагменты этих и других решений представлены справа. Нажмите на любой из них для детального изучения.`;
    
    return synthesis;
  }

  function renderSidebar(items) {
    resultsCount.textContent = items.length;
    if (items.length === 0) {
      resultsArea.innerHTML = '<div class="empty-state">Ничего не найдено.</div>';
      return;
    }

    resultsArea.innerHTML = items.map((item, idx) => `
      <div class="fragment-card" data-id="${item.id}">
        <span class="fragment-card__id">[${idx + 1}] ${item.jurisdiction}</span>
        <div class="fragment-card__title">${item.title}</div>
        <div class="fragment-card__text">${item.summary}</div>
      </div>
    `).join('');

    // Click behavior
    resultsArea.querySelectorAll('.fragment-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = card.dataset.id;
        const doc = items.find(d => d.id === id);
        appendMessage('ai', `<strong>Детальный фрагмент решения ${doc.case}:</strong><br><br>${doc.text}`);
      });
    });
  }

  function appendMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `chat-message chat-message--${role}`;
    
    const inner = `
      <div class="chat-message__avatar">${role === 'ai' ? 'L' : 'U'}</div>
      <div class="chat-message__content">
        <div class="chat-message__author">${role === 'ai' ? 'Layla AI' : 'Вы'}</div>
        <div class="chat-message__text">${text}</div>
      </div>
    `;
    msg.innerHTML = inner;
    chatMessages.appendChild(msg);
    
    // Force scroll to bottom (immediate + delayed for safety)
    const scrollToBottom = () => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    };
    
    scrollToBottom();
    setTimeout(scrollToBottom, 50);
    setTimeout(scrollToBottom, 200);

    // Handle citations in new message
    msg.querySelectorAll('cite').forEach(cite => {
      cite.addEventListener('click', () => {
        const idx = parseInt(cite.textContent) - 1;
        if (currentResults[idx]) {
          const card = resultsArea.querySelector(`[data-id="${currentResults[idx].id}"]`);
          if (card) {
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
            card.classList.add('fragment-card--active');
            setTimeout(() => card.classList.remove('fragment-card--active'), 2000);
          }
        }
      });
    });
  }

  // Auto-resize textarea
  chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
  });

})();
