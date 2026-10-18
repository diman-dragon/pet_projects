function buildSchedule(filterGroup) {
  if (typeof SCHEDULE === 'undefined') return;

  const bar = document.getElementById('filter-bar');
  if (bar) {
    const groups = [...new Set(SCHEDULE.filter(m => m.group).map(m => m.group))].sort();
    bar.innerHTML = `
      <span class="filter-label">Группа:</span>
      <button class="filter-chip ${filterGroup === 'all' ? 'active' : ''}" data-filter="all">Все</button>
      ${groups.map(g => `<button class="filter-chip ${filterGroup === g ? 'active' : ''}" data-filter="${g}">Гр. ${g}</button>`).join('')}
    `;
    bar.querySelectorAll('.filter-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        activeFilter = btn.dataset.filter;
        buildSchedule(activeFilter);
      });
    });
  }

  const matches = filterGroup === 'all'
    ? SCHEDULE.filter(m => m.stage === 'Group Stage')
    : SCHEDULE.filter(m => m.stage === 'Group Stage' && m.group === filterGroup);

  const byDate = {};
  matches.forEach(m => { if (!byDate[m.date]) byDate[m.date] = []; byDate[m.date].push(m); });

  const container = document.getElementById('schedule-container');
  if (!container) return;
  container.innerHTML = '';

  const today = new Date().toISOString().split('T')[0];
  let todayId = null;

  Object.keys(byDate).sort().forEach(date => {
    const grp = document.createElement('div');
    grp.className = 'date-group';
    if (date === today) {
        grp.id = 'today-group';
        todayId = 'today-group';
        grp.classList.add('is-today');
    }
    const label = (typeof DATE_LABELS !== 'undefined' && DATE_LABELS[date]) ? DATE_LABELS[date] : date;
    grp.innerHTML = `<div class="date-heading"><span>${label}</span>${date === today ? ' <span class="today-badge">СЕГОДНЯ</span>' : ''}<span class="date-heading-line"></span></div>`;

    byDate[date].forEach(m => {
      const matchKey = `${m.home} vs ${m.away}`;
      const res = RESULTS[matchKey] || null;
      const card = document.createElement('div');
      card.className = 'match-card clickable' + (res ? ' has-result' : '');
      card.onclick = () => showView('match-details', { matchKey });
      
      card.innerHTML = `
        <div class="group-pill" onclick="event.stopPropagation(); showView('group-details', {groupLetter: '${m.group}'})">Гр. ${m.group}</div>
        <div class="team-block" onclick="event.stopPropagation(); showView('team-details', {teamName: '${m.home}'})">
          <span class="team-logo-wrap">${teamLogo(m.home, 32)}</span>
          <span class="team-name">${m.home}</span>
        </div>
        <div class="score-block">
          ${res ? `<span class="score-value">${res.score}</span>` : `<span class="score-pending">vs</span>`}
          <div class="time-stadium-wrap">
            <span class="score-time">${m.time}</span>
            ${m.stadium ? `<div class="match-stadium">${m.stadium}</div>` : ''}
          </div>
        </div>
        <div class="team-block away" onclick="event.stopPropagation(); showView('team-details', {teamName: '${m.away}'})">
          <span class="team-logo-wrap">${teamLogo(m.away, 32)}</span>
          <span class="team-name">${m.away}</span>
        </div>
      `;
      grp.appendChild(card);
    });
    container.appendChild(grp);
  });

  // Автофокус строго на сегодня, если есть матчи
  if (todayId && currentView === 'schedule') {
    setTimeout(() => {
        const el = document.getElementById(todayId);
        if (el) {
            // Используем стандартный скролл, настроив отступ через CSS
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 250);
  }
}
