function renderGroupDetails(groupLetter) {
    const container = document.getElementById('view-group-details');
    if (!container) return;
    
    const groupMatches = SCHEDULE.filter(m => m.group === groupLetter);
    const table = calculateGroupTable(groupMatches);
    
    container.innerHTML = `
        <div class="group-details-container">
            <button class="back-btn" onclick="showView('groups')">← Назад к группам</button>
            <h2>Аналитика Группы ${groupLetter}</h2>
            
            <div class="stats-overview">
                <div class="overview-card">
                    <div class="overview-val">${groupMatches.filter(m => getMatchResult(m.home, m.away)).length} / ${groupMatches.length}</div>
                    <div class="overview-label">Сыграно матчей</div>
                </div>
                <div class="overview-card">
                    <div class="overview-val">${table.reduce((sum, t) => sum + t.gf, 0)}</div>
                    <div class="overview-label">Всего голов</div>
                </div>
                <div class="overview-card">
                    <div class="overview-val">${(table.reduce((sum, t) => sum + t.gf + t.ga, 0) / groupMatches.length).toFixed(1)}</div>
                    <div class="overview-label">Голов за матч</div>
                </div>
            </div>

            <div class="stats-panel">
                <div class="panel-title">Вероятности выхода (базовая)</div>
                <div class="top-teams-chart">
                    ${table.map((t, idx) => {
                        const remaining = groupMatches.filter(m => (m.home === t.team || m.away === t.team) && !getMatchResult(m.home, m.away)).length;
                        const maxPts = t.pts + (remaining * 3);
                        return `
                            <div class="chart-row">
                                <div class="chart-team">${teamLogo(t.team, 20)} ${t.team}</div>
                                <div class="chart-bar-wrap">
                                    <div class="chart-bar" style="width: ${(maxPts / 9) * 100}%"></div>
                                </div>
                                <div class="chart-val">${t.pts} (${maxPts})</div>
                            </div>
                        `;
                    }).join('')}
                </div>
                <p style="margin-top: 15px; font-size: 0.85rem; color: var(--text-tertiary);">
                    *В скобках указано максимально возможное количество очков.
                </p>
            </div>
        </div>
    `;
}
