function renderMatchDetails(matchKey) {
    const container = document.getElementById('view-match-details');
    if (!container) return;

    const match = typeof getMatchAnalytics === 'function' ? getMatchAnalytics(matchKey) : null;
    const [rawHome, rawAway] = splitMatchKey(matchKey);
    const hName = match ? match.home : rawHome;
    const aName = match ? match.away : rawAway;
    const fallback = RESULTS && RESULTS[`${hName} vs ${aName}`] ? RESULTS[`${hName} vs ${aName}`] : null;
    const score = match ? scoreFromValue(match.score) : {
        home: fallback ? fallback.home : 0,
        away: fallback ? fallback.away : 0
    };
    const homeStats = match ? (match.stats[hName] || {}) : {};
    const awayStats = match ? (match.stats[aName] || {}) : {};

    function row(name) {
        const h = homeStats[name];
        const a = awayStats[name];
        if ((h === undefined || h === '') && (a === undefined || a === '')) return '';
        const hN = parseStatValue(h);
        const aN = parseStatValue(a);
        const total = hN + aN || 1;
        return `
            <div class="mini-stat-row">
                <div class="ms-label">${esc(STAT_LABELS[name] || name)}</div>
                <div class="ms-bars">
                    <div class="ms-bar-val left">${formatStatValue(name, h)}</div>
                    <div class="ms-bar-bg">
                        <div class="ms-bar-fill left" style="width:${Math.max(hN / total * 100, hN ? 4 : 0)}%"></div>
                        <div class="ms-bar-fill right" style="width:${Math.max(aN / total * 100, aN ? 4 : 0)}%"></div>
                    </div>
                    <div class="ms-bar-val right">${formatStatValue(name, a)}</div>
                </div>
            </div>`;
    }

    const visibleStats = STAT_ORDER.map(row).filter(Boolean).join('');
    const homeShots = parseStatValue(homeStats.totalShots);
    const awayShots = parseStatValue(awayStats.totalShots);
    const homeSot = parseStatValue(homeStats.shotsOnTarget);
    const awaySot = parseStatValue(awayStats.shotsOnTarget);
    const story = [
        `${score.home + score.away} голов`,
        `${homeShots + awayShots || 0} ударов`,
        `${homeSot + awaySot || 0} ударов в створ`
    ];
    if (match && match.venue) story.push(match.venue);

    container.innerHTML = `
        <div class="detail-header">
            <button class="back-btn" onclick="goBack()">← Назад</button>
            <h2>Статистика матча</h2>
        </div>
        <div class="detailed-card full-width match-detail-shell">
            <div class="dc-header match-detail-header">
                <div class="dc-team" onclick="showView('team-details', {teamName: ${jsArg(hName)}})">
                    ${teamLogo(hName, 64)}
                    <span>${esc(hName)}</span>
                </div>
                <div class="score-stack">
                    <div class="dc-score main-score">${score.home} : ${score.away}</div>
                    ${match && match.date ? `<div class="score-date">${esc(match.date)}</div>` : ''}
                </div>
                <div class="dc-team right" onclick="showView('team-details', {teamName: ${jsArg(aName)}})">
                    <span>${esc(aName)}</span>
                    ${teamLogo(aName, 64)}
                </div>
            </div>

            <div class="match-story-strip">
                ${story.map(item => `<span>${esc(item)}</span>`).join('')}
            </div>

            <div class="match-kpi-grid">
                <div class="match-kpi"><b>${homeShots}</b><span>${esc(hName)} ударов</span></div>
                <div class="match-kpi"><b>${homeSot}:${awaySot}</b><span>удары в створ</span></div>
                <div class="match-kpi"><b>${awayShots}</b><span>${esc(aName)} ударов</span></div>
            </div>

            <div class="dc-body big-stats">
                ${visibleStats || `
                    <div class="placeholder-msg">
                        <p>Детальная статистика пока недоступна.</p>
                        <p style="font-size: 0.8rem; margin-top: 10px; color: var(--text-tertiary);">
                            Нажмите «Загрузить» в шапке, чтобы подтянуть свежие данные ESPN.
                        </p>
                    </div>
                `}
            </div>
        </div>
    `;
}

function loadMatchDetails() {
    renderView('match-details', { matchKey: 'Australia vs Türkiye' });
}
