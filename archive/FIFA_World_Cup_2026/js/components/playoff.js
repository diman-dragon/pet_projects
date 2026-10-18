function getTeamForSlot(slot) {
  // slot format example: '1A', '2C', 'Winner 89', 'Loser 101'
  if (slot.startsWith("Winner ")) {
    const matchId = parseInt(slot.split(" ")[1]);
    const match = SCHEDULE.find(m => m.id === matchId);
    if (!match) return slot;
    const res = getResult(match.home, match.away);
    if (!res) return slot;
    return parseInt(res.home) > parseInt(res.away) ? match.home : match.away;
  }
  if (slot.startsWith("Loser ")) {
    const matchId = parseInt(slot.split(" ")[1]);
    const match = SCHEDULE.find(m => m.id === matchId);
    if (!match) return slot;
    const res = getResult(match.home, match.away);
    if (!res) return slot;
    return parseInt(res.home) < parseInt(res.away) ? match.home : match.away;
  }
  
  const rank = parseInt(slot[0]);
  const group = slot[1];
  
  const allStats = calculateAllTeamsStats();
  const groupTeams = allStats.filter(t => t.group === group);
  const team = groupTeams.find(t => t.rankInGroup === rank);
  
  return team ? team.team : slot;
}

function buildPlayoff() {
  const stages = [
    { key: 'Round of 32',   label: '1/16 финала' },
    { key: 'Round of 16',   label: '1/8 финала' },
    { key: 'Quarter-final', label: '1/4 финала' },
    { key: 'Semi-final',    label: '1/2 финала' },
    { key: 'Final',         label: '🏆 Финал' },
  ];

  const container = document.getElementById('playoff-container');
  if (!container) return;
  container.innerHTML = `<div class="bracket-container"></div>`;
  const bracket = container.querySelector('.bracket-container');

  stages.forEach(({ key, label }) => {
    const stageMatches = SCHEDULE.filter(m => m.stage === key);
    if (!stageMatches.length) return;

    const column = document.createElement('div');
    column.className = 'bracket-column';
    column.innerHTML = `<div class="bracket-stage-header">${label}</div>`;

    stageMatches.forEach(m => {
      const homeTeam = getTeamForSlot(m.home) || m.home;
      const awayTeam = getTeamForSlot(m.away) || m.away;
      
      const matchKey = `${homeTeam} vs ${awayTeam}`;
      const res = getResult(homeTeam, awayTeam);
      
      const getLogo = (name) => {
          const enName = (typeof window.RU_TO_EN !== 'undefined' && window.RU_TO_EN[name]) ? window.RU_TO_EN[name] : name;
          return (typeof LOGOS !== 'undefined' && LOGOS[enName]) ? LOGOS[enName] : null;
      };
      
      const homeLogo = getLogo(homeTeam);
      const awayLogo = getLogo(awayTeam);
      
      const card = document.createElement('div');
      card.className = 'bracket-match clickable';
      card.onclick = () => showView('match-details', { matchKey });
      
      card.innerHTML = `
        <div class="bm-team ${res && res.home > res.away ? 'winner' : ''}">
          <span class="bm-flag">${homeLogo ? `<img src="${homeLogo}" style="width:20px;height:20px;vertical-align:middle;margin-right:8px;">` : ''}</span>
          <span class="bm-name">${homeTeam}</span>
          <span class="bm-score">${res ? res.home : ''}</span>
        </div>
        <div class="bm-team ${res && res.away > res.home ? 'winner' : ''}">
          <span class="bm-flag">${awayLogo ? `<img src="${awayLogo}" style="width:20px;height:20px;vertical-align:middle;margin-right:8px;">` : ''}</span>
          <span class="bm-name">${awayTeam}</span>
          <span class="bm-score">${res ? res.away : ''}</span>
        </div>
        <div class="bm-meta">
            <div class="bm-time">${m.time} • ${DATE_LABELS[m.date] || m.date}</div>
            <div class="bm-stadium">🏟 ${m.stadium || 'TBD'}</div>
            ${res ? '<div class="bm-result">Матч завершен</div>' : ''}
        </div>
      `;
      column.appendChild(card);
    });

    bracket.appendChild(column);
  });
}
