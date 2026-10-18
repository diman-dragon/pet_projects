function buildMasterTable(sortColumn = 'status', direction = 'asc') {
    const container = document.getElementById('view-master-table');
    if (!container) return;

    let allStats = calculateAllTeamsStats();
    
    const statusPriority = { 'qualified': 1, 'eliminated': 2, 'pending': 3 };

    // Interactive sorting
    allStats.sort((a, b) => {
        let valA = a[sortColumn];
        let valB = b[sortColumn];
        
        if (sortColumn === 'status') {
            const pA = statusPriority[a.status] || 99;
            const pB = statusPriority[b.status] || 99;
            return direction === 'asc' ? pA - pB : pB - pA;
        }

        if (typeof valA === 'string') {
            return direction === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
        return direction === 'asc' ? valA - valB : valB - valA;
    });
    
    const getHeaderClass = (col) => (window.currentSort.column === col ? `sort-active ${window.currentSort.direction}` : '');

    container.innerHTML = `
        <div class="master-table-container">
            <h2>Турнирная таблица (Групповой этап)</h2>
            <div class="table-wrapper">
                <table class="full-table master-table">
                    <thead>
                        <tr>
                            <th class="${getHeaderClass('rankInGroup')}" onclick="sortTable('rankInGroup')">#</th>
                            <th class="th-team ${getHeaderClass('team')}" onclick="sortTable('team')">Команда</th>
                            <th class="${getHeaderClass('group')}" onclick="sortTable('group')">Гр.</th>
                            <th class="${getHeaderClass('played')}" onclick="sortTable('played')">И</th>
                            <th class="${getHeaderClass('win')}" onclick="sortTable('win')">В</th>
                            <th class="${getHeaderClass('draw')}" onclick="sortTable('draw')">Н</th>
                            <th class="${getHeaderClass('loss')}" onclick="sortTable('loss')">П</th>
                            <th class="${getHeaderClass('gf')}" onclick="sortTable('gf')">З-П</th>
                            <th class="${getHeaderClass('pts')}" onclick="sortTable('pts')">О</th>
                            <th class="${getHeaderClass('status')}" onclick="sortTable('status')">Статус</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${allStats.map((row, idx) => `
                            <tr class="status-${row.status}">
                                <td>${row.rankInGroup}</td>
                                <td class="td-team">
                                    <div class="team-identity" onclick="showView('team-details', {teamName: '${row.team}'})">
                                        ${teamLogo(row.team, 20)}
                                        <span class="team-name-table">${row.team}</span>
                                    </div>
                                </td>
                                <td>${row.group}</td>
                                <td>${row.played}</td>
                                <td>${row.win}</td>
                                <td>${row.draw}</td>
                                <td>${row.loss}</td>
                                <td>${row.gf}-${row.ga}</td>
                                <td class="stat-cell pts">${row.pts}</td>
                                <td><span class="badge-status status-${row.status}">${row.statusLabel}</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

// Global variable for sorting state
window.currentSort = { column: 'status', direction: 'asc' };

function sortTable(column) {
    if (window.currentSort.column === column) {
        window.currentSort.direction = window.currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        window.currentSort.column = column;
        window.currentSort.direction = column === 'status' ? 'asc' : 'desc';
    }
    buildMasterTable(window.currentSort.column, window.currentSort.direction);
}

function calculateAllTeamsStats() {
    const stats = {};
    const groupMatches = SCHEDULE.filter(m => m.group);

    // 1. Initial calculation
    groupMatches.forEach(m => {
        [m.home, m.away].forEach(t => {
            if (!stats[t]) stats[t] = { team: t, played: 0, win: 0, draw: 0, loss: 0, gf: 0, ga: 0, pts: 0, group: m.group };
        });
    });

    groupMatches.forEach(m => {
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

    // 2. Pre-calculate statuses to use in sorting
    const teamsWithStatus = Object.values(stats).map(team => {
        // Find group, calculate rank, check status (logic from end)
        const groupTeams = Object.values(stats).filter(t => t.group === team.group);
        // Temporary sort to find rank
        const sortedGroupTemp = [...groupTeams].sort((a, b) => b.pts - a.pts || (a.gf-a.ga) - (b.gf-b.ga));
        const rankInGroup = sortedGroupTemp.findIndex(t => t.team === team.team) + 1;
        
        const remainingMatches = SCHEDULE.filter(m => m.group === team.group && !RESULTS[`${m.home} vs ${m.away}`] && !RESULTS[`${m.away} vs ${m.home}`]);
        const maxPts = team.pts + (remainingMatches.filter(m => m.home === team.team || m.away === team.team).length * 3);
        
        let status = 'pending';
        let statusLabel = 'Борьба';
        
        // Быстрая проверка на основе очков/результатов
        if (team.pts >= 6) {
            status = 'qualified';
            statusLabel = 'Точно выходит';
        } else if (team.played >= 2 && team.pts === 0) {
            status = 'eliminated';
            statusLabel = 'Точно вылетает';
        } else {
            const thirdPlace = sortedGroupTemp[2];
            const secondPlace = sortedGroupTemp[1];
            
            // Добавляем проверку, сыграла ли команда все свои матчи
            const teamMatches = SCHEDULE.filter(m => m.group === team.group && (m.home === team.team || m.away === team.team));
            const playedMatches = teamMatches.filter(m => RESULTS[`${m.home} vs ${m.away}`] || RESULTS[`${m.away} vs ${m.home}`]);
            const allMatchesPlayed = playedMatches.length === teamMatches.length;

            if (rankInGroup <= 2) {
                // Если все матчи сыграны - они точно вышли
                if (allMatchesPlayed) {
                    status = 'qualified';
                    statusLabel = 'Точно выходит';
                } else {
                    // Иначе проверяем математическую гарантию
                    const worstPotential3rd = thirdPlace ? (thirdPlace.pts + remainingMatches.filter(m => m.home === thirdPlace.team || m.away === thirdPlace.team).length * 3) : 0;
                    if (team.pts > worstPotential3rd) {
                        status = 'qualified';
                        statusLabel = 'Точно выходит';
                    }
                }
            } else if (rankInGroup === 3) {
                 statusLabel = 'Борьба (3-е место)';
            } else if (rankInGroup === 4) {
                 if (secondPlace && maxPts < secondPlace.pts) {
                    status = 'eliminated';
                    statusLabel = 'Точно вылетает';
                 }
            }
        }
        
        return { ...team, status, statusLabel, rankInGroup };
    });

    // 3. Sort groups
    const groups = {};
    'ABCDEFGHIJKL'.split('').forEach(l => groups[l] = []);
    teamsWithStatus.forEach(t => groups[t.group].push(t));
    
    const statusPriority = { 'qualified': 1, 'eliminated': 2, 'pending': 3 };

    Object.keys(groups).forEach(l => {
        groups[l].sort((a, b) => {
            // 0. Qualification Status - STRICT SORTING
            const pA = statusPriority[a.status] || 99;
            const pB = statusPriority[b.status] || 99;
            return pA - pB;
        });
    });
    
    // Flatten and finalize
    return Object.values(groups).flat();
}
