function buildGroups() {
  const container = document.getElementById('groups-container');
  if (!container) return;
  container.innerHTML = '';

  const groups = 'ABCDEFGHIJKL'.split('');
  
  groups.forEach(letter => {
    const groupMatches = SCHEDULE.filter(m => m.group === letter);
    const table = calculateGroupTable(groupMatches);
    
    const card = document.createElement('div');
    card.className = 'group-card';
    
    card.innerHTML = `
      <div class="group-header">
        <span class="group-letter">Группа ${letter}</span>
        <button class="view-all-btn" onclick="event.stopPropagation(); showView('group-details', { groupLetter: '${letter}' })">Аналитика</button>
      </div>
      <div class="table-wrapper">
        <table class="full-table">
          <thead>
            <tr>
              <th class="th-pos">#</th>
              <th class="th-team">Команда</th>
              <th>И</th>
              <th>В</th>
              <th>Н</th>
              <th>П</th>
              <th>З-П</th>
              <th>О</th>
            </tr>
          </thead>
          <tbody>
            ${table.map((row, idx) => `
              <tr>
                <td><span class="pos-badge pos-${idx+1}">${idx+1}</span></td>
                <td class="td-team">
                    <div class="team-identity" onclick="event.stopPropagation(); showView('team-details', {teamName: '${row.team}'})">
                        ${teamLogo(row.team, 18)}
                        <span class="team-name-table">${row.team}</span>
                    </div>
                </td>
                <td>${row.played}</td>
                <td>${row.win}</td>
                <td>${row.draw}</td>
                <td>${row.loss}</td>
                <td class="td-gd">${row.gf}-${row.ga}</td>
                <td class="stat-cell pts">${row.pts}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>

      <div class="matrix-grid-title">Сетка результатов</div>
      <div class="table-wrapper">
        <table class="results-matrix compact">
           <thead>
             <tr>
               <th class="matrix-corner-label">Команда</th>
               ${table.map(row => {
                 const short = TEAM_SHORT[row.team] || row.team.substring(0,3).toUpperCase();
                 return `<th title="${row.team}" class="matrix-th-col">${short}</th>`;
               }).join('')}
             </tr>
           </thead>
           <tbody>
             ${table.map(row => {
               const short = TEAM_SHORT[row.team] || row.team.substring(0,3).toUpperCase();
               return `
                 <tr>
                   <td class="matrix-row-label" title="${row.team}">
                     <div class="row-label-content">${teamLogo(row.team, 16)} ${short}</div>
                   </td>
                   ${table.map(col => {
                      if (row.team === col.team) return '<td class="matrix-self"></td>';
                      const res = getMatchResult(row.team, col.team);
                      const date = getMatchDate(row.team, col.team);
                      const matchKey = `${row.team} vs ${col.team}`;
                      return `<td class="matrix-res-cell" onclick="showView('match-details', {matchKey: '${matchKey}'})" title="${row.team} vs ${col.team}: ${res || date || 'Не сыгран'}" style="cursor:pointer">
                                <span class="matrix-score">${res ? res : `<span style="color:#aaa; font-size: 0.65rem;">${date}</span>`}</span>
                              </td>`;
                   }).join('')}
                 </tr>
               `;
             }).join('')}
           </tbody>
        </table>
      </div>
    `;
    container.appendChild(card);
  });
}

function getMatchResult(home, away) {
    const res = RESULTS[`${home} vs ${away}`] || 
                (window.MATCH_STATS && window.MATCH_STATS.stats && window.MATCH_STATS.stats[`${home} vs ${away}`] 
                  ? {
                      score: window.MATCH_STATS.stats[`${home} vs ${away}`].score.replace(' - ', ':')
                    } 
                  : null);
    if (res) return res.score.replace(' ', '');

    const resRev = RESULTS[`${away} vs ${home}`] || 
                   (window.MATCH_STATS && window.MATCH_STATS.stats && window.MATCH_STATS.stats[`${away} vs ${home}`] 
                     ? {
                         score: window.MATCH_STATS.stats[`${away} vs ${home}`].score.replace(' - ', ':')
                       } 
                     : null);
    if (resRev) {
        const parts = resRev.score.replace(' ', '').split(':');
        return `${parts[1]}:${parts[0]}`;
    }

    return null;
}

function getMatchDate(home, away) {
    const match = SCHEDULE.find(m => (m.home === home && m.away === away) || (m.home === away && m.away === home));
    if (match) {
        const parts = match.date.split('-'); // 2026-06-11
        return `${parts[2]}.${parts[1]}`;
    }
    return '';
}

function calculateGroupTable(matches) {
  const stats = {};
  matches.forEach(m => {
    [m.home, m.away].forEach(t => {
      if (!stats[t]) stats[t] = { team: t, played: 0, win: 0, draw: 0, loss: 0, gf: 0, ga: 0, pts: 0 };
    });
  });

  matches.forEach(m => {
    const res = RESULTS[`${m.home} vs ${m.away}`] || 
                (window.MATCH_STATS && window.MATCH_STATS.stats && window.MATCH_STATS.stats[`${m.home} vs ${m.away}`] 
                  ? {
                      home: parseInt(window.MATCH_STATS.stats[`${m.home} vs ${m.away}`].score.split(' - ')[0]),
                      away: parseInt(window.MATCH_STATS.stats[`${m.home} vs ${m.away}`].score.split(' - ')[1])
                    } 
                  : null);

    if (res && res.home !== undefined) {
      const h = m.home, a = m.away;
      stats[h].played++; stats[a].played++;
      stats[h].gf += res.home; stats[h].ga += res.away;
      stats[a].gf += res.away; stats[a].ga += res.home;

      if (res.home > res.away) {
        stats[h].win++; stats[h].pts += 3; stats[a].loss++;
      } else if (res.home < res.away) {
        stats[a].win++; stats[a].pts += 3; stats[h].loss++;
      } else {
        stats[h].draw++; stats[h].pts += 1; stats[a].draw++; stats[a].pts += 1;
      }
    }
  });

  return Object.values(stats).sort((a, b) => {
    // 1. Points
    if (b.pts !== a.pts) return b.pts - a.pts;

    // 2. Head-to-Head
    const headToHead = matches.find(m => (m.home === a.team && m.away === b.team) || (m.home === b.team && m.away === a.team));
    if (headToHead) {
      const res = RESULTS[`${headToHead.home} vs ${headToHead.away}`] || 
                  (window.MATCH_STATS && window.MATCH_STATS.stats && window.MATCH_STATS.stats[`${headToHead.home} vs ${headToHead.away}`] 
                    ? {
                        home: parseInt(window.MATCH_STATS.stats[`${headToHead.home} vs ${headToHead.away}`].score.split(' - ')[0]),
                        away: parseInt(window.MATCH_STATS.stats[`${headToHead.home} vs ${headToHead.away}`].score.split(' - ')[1])
                      } 
                    : null);
      if (res && res.home !== undefined) {
          const aGoals = headToHead.home === a.team ? res.home : res.away;
          const bGoals = headToHead.home === b.team ? res.home : res.away;
          if (aGoals > bGoals) return -1;
          if (bGoals > aGoals) return 1;
      }
    }

    // 3. Goal Difference
    const diffA = a.gf - a.ga, diffB = b.gf - b.ga;
    if (diffB !== diffA) return diffB - diffA;

    // 4. Goals Scored
    return b.gf - a.gf;
  });
}
