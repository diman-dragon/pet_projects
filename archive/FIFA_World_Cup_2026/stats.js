const STAT_LABELS = {
    possessionPct: 'Владение',
    totalShots: 'Удары',
    shotsOnTarget: 'В створ',
    shotPct: 'Точность ударов',
    wonCorners: 'Угловые',
    foulsCommitted: 'Фолы',
    yellowCards: 'ЖК',
    redCards: 'КК',
    offsides: 'Офсайды',
    saves: 'Сэйвы',
    goalAssists: 'Голевые передачи',
    shotAssists: 'Ключевые передачи',
    accuratePasses: 'Точные передачи',
    totalPasses: 'Передачи',
    passPct: 'Точность передач',
    accurateCrosses: 'Точные навесы',
    totalCrosses: 'Навесы',
    crossPct: 'Точность навесов',
    effectiveTackles: 'Успешные отборы',
    totalTackles: 'Отборы',
    tacklePct: 'Точность отборов',
    interceptions: 'Перехваты',
    blockedShots: 'Блок. удары',
    totalClearance: 'Выносы'
};

const STAT_ORDER = [
    'possessionPct', 'totalShots', 'shotsOnTarget', 'shotPct', 'wonCorners',
    'foulsCommitted', 'yellowCards', 'redCards', 'offsides', 'saves',
    'goalAssists', 'shotAssists', 'accuratePasses', 'totalPasses', 'passPct',
    'accurateCrosses', 'totalCrosses', 'effectiveTackles', 'totalTackles',
    'interceptions', 'blockedShots', 'totalClearance'
];

// Поля, которые встречаются ТОЛЬКО в реальной посматчевой статистике (boxscore).
// Превью несыгранных матчей ESPN отдаёт всего 4 поля-сводки (totalGoals, goalDifference,
// goalAssists, goalsConceded) — этого недостаточно, чтобы матч считался состоявшимся.
const REAL_MATCH_STAT_KEYS = ['totalShots', 'shotsOnTarget', 'possessionPct', 'foulsCommitted', 'totalPasses', 'wonCorners'];

function hasMatchTelemetry(stats) {
    if (!stats) return false;
    return REAL_MATCH_STAT_KEYS.some(key => stats[key] !== undefined && stats[key] !== null && stats[key] !== '');
}

const TEAM_ALIASES = {
    'United States': 'USA',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Czech Republic': 'Czechia',
    'DR Congo': 'DR Congo',
    'Congo DR': 'DR Congo',
    'Curacao': 'Curacao',
    'Curaçao': 'Curacao',
    'Türkiye': 'Türkiye',
    'Turkiye': 'Türkiye',
    'TГјrkiye': 'Türkiye'
};

function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}

function jsArg(value) {
    return esc(JSON.stringify(String(value ?? '')));
}

function normalizeTeamName(name) {
    if (!name) return '';
    const source = TEAM_ALIASES[name] || name;
    return typeof t === 'function' ? t(source) : source;
}

function parseStatValue(value) {
    if (value === null || value === undefined || value === '') return 0;
    const cleaned = String(value).replace('%', '').replace(',', '.');
    const parsed = parseFloat(cleaned);
    return Number.isFinite(parsed) ? parsed : 0;
}

function statsArrayToObject(stats) {
    if (!stats) return {};
    if (!Array.isArray(stats)) return { ...stats };
    return stats.reduce((acc, item) => {
        if (item && item.name) acc[item.name] = item.displayValue;
        return acc;
    }, {});
}

function formatPct(value) {
    const n = parseStatValue(value);
    if (!n) return '0%';
    return `${n % 1 === 0 ? n.toFixed(0) : n.toFixed(1)}%`;
}

function formatStatValue(name, value) {
    const n = parseStatValue(value);
    if (['possessionPct', 'shotPct', 'passPct', 'crossPct', 'longballPct', 'tacklePct'].includes(name)) {
        const pct = n > 0 && n <= 1 ? n * 100 : n;
        return `${pct % 1 === 0 ? pct.toFixed(0) : pct.toFixed(1)}%`;
    }
    return Number.isFinite(n) ? String(n % 1 === 0 ? n.toFixed(0) : n.toFixed(1)) : esc(value);
}

function splitMatchKey(matchKey) {
    const parts = String(matchKey).split(' vs ');
    return [normalizeTeamName(parts[0]), normalizeTeamName(parts[1])];
}

function scoreFromValue(score) {
    const parts = String(score || '').replace(':', '-').split('-').map(v => parseInt(v.trim(), 10));
    return {
        home: Number.isFinite(parts[0]) ? parts[0] : 0,
        away: Number.isFinite(parts[1]) ? parts[1] : 0
    };
}

function mergeMatch(map, matchKey, payload) {
    const [home, away] = splitMatchKey(matchKey);
    if (!home || !away) return;
    const key = `${home} vs ${away}`;
    if (!map[key]) {
        map[key] = {
            key,
            home,
            away,
            score: null,
            date: null,
            venue: null,
            id: null,
            verified: false,
            stats: {}
        };
    }

    const match = map[key];
    Object.assign(match, {
        score: payload.score || match.score,
        date: payload.date || match.date,
        venue: payload.venue || match.venue,
        id: payload.id || match.id,
        verified: match.verified || !!payload.verified
    });

    Object.entries(payload.stats || {}).forEach(([team, stats]) => {
        match.stats[normalizeTeamName(team)] = {
            ...(match.stats[normalizeTeamName(team)] || {}),
            ...statsArrayToObject(stats)
        };
    });
}

function createEmptyTeam(name) {
    return {
        name,
        played: 0,
        wins: 0,
        draws: 0,
        losses: 0,
        goalsFor: 0,
        goalsAgainst: 0,
        points: 0,
        cleanSheets: 0,
        failedToScore: 0,
        shots: 0,
        shotsOnTarget: 0,
        corners: 0,
        fouls: 0,
        yellows: 0,
        reds: 0,
        assists: 0,
        keyPasses: 0,
        passes: 0,
        accuratePasses: 0,
        possessionTotal: 0,
        possessionSamples: 0,
        matches: []
    };
}

function applyTeamMatch(team, opponent, gf, ga, stats, match) {
    team.played += 1;
    team.goalsFor += gf;
    team.goalsAgainst += ga;
    team.points += gf > ga ? 3 : gf === ga ? 1 : 0;
    if (gf > ga) team.wins += 1;
    if (gf === ga) team.draws += 1;
    if (gf < ga) team.losses += 1;
    if (ga === 0) team.cleanSheets += 1;
    if (gf === 0) team.failedToScore += 1;

    team.shots += parseStatValue(stats.totalShots);
    team.shotsOnTarget += parseStatValue(stats.shotsOnTarget);
    team.corners += parseStatValue(stats.wonCorners);
    team.fouls += parseStatValue(stats.foulsCommitted);
    team.yellows += parseStatValue(stats.yellowCards);
    team.reds += parseStatValue(stats.redCards);
    team.assists += parseStatValue(stats.goalAssists);
    team.keyPasses += parseStatValue(stats.shotAssists);
    team.passes += parseStatValue(stats.totalPasses);
    team.accuratePasses += parseStatValue(stats.accuratePasses);

    const possession = parseStatValue(stats.possessionPct);
    if (possession) {
        team.possessionTotal += possession;
        team.possessionSamples += 1;
    }

    team.matches.push({
        opponent,
        gf,
        ga,
        result: gf > ga ? 'W' : gf === ga ? 'D' : 'L',
        score: `${gf} : ${ga}`,
        date: match.date,
        venue: match.venue,
        key: match.key,
        stats
    });
}

function buildTournamentAnalytics() {
    const matches = {};

    if (window.MATCH_STATS && window.MATCH_STATS.stats) {
        Object.entries(window.MATCH_STATS.stats).forEach(([key, info]) => {
            // Источник курируется вручную и содержит только реально сыгранные матчи.
            mergeMatch(matches, key, { score: info.score, date: info.date, verified: true });
        });
    }

    if (window.FULL_MATCH_STATS) {
        Object.entries(window.FULL_MATCH_STATS).forEach(([key, stats]) => {
            // loader_stats.py отбирает только завершённые (completed) события ESPN.
            mergeMatch(matches, key, { stats, verified: true });
        });
    }

    if (window.PRO_TOURNAMENT_DATA && window.PRO_TOURNAMENT_DATA.matches) {
        Object.entries(window.PRO_TOURNAMENT_DATA.matches).forEach(([key, info]) => {
            // Этот источник исторически отдавал и ещё не сыгранные матчи с
            // "превью"-статистикой (0:0 + 4 общих поля) — не считаем его доверенным
            // автоматически, ниже отфильтруем такие записи по реальной телеметрии.
            mergeMatch(matches, key, {
                id: info.id,
                score: info.score,
                venue: info.venue,
                stats: info.statistics
            });
        });
    }

    Object.values(matches).forEach(match => {
        if (!match.score) {
            const result = RESULTS && RESULTS[match.key] ? RESULTS[match.key] : null;
            if (result) match.score = `${result.home} - ${result.away}`;
        }
    });

    // Отбрасываем "фантомные" записи: матчи, которые не подтверждены доверенным
    // источником и не содержат реальной посматчевой статистики (т.е. ещё не сыграны,
    // а в данных оказался лишь предматчевый превью-блок). Иначе такие матчи
    // ошибочно засчитываются командам как сыгранная ничья 0:0.
    Object.keys(matches).forEach(key => {
        const match = matches[key];
        const homeStats = match.stats[match.home];
        const awayStats = match.stats[match.away];
        if (!match.verified && !hasMatchTelemetry(homeStats) && !hasMatchTelemetry(awayStats)) {
            delete matches[key];
        }
    });

    const teams = {};
    Object.values(matches).forEach(match => {
        const score = scoreFromValue(match.score);
        if (!teams[match.home]) teams[match.home] = createEmptyTeam(match.home);
        if (!teams[match.away]) teams[match.away] = createEmptyTeam(match.away);
        applyTeamMatch(teams[match.home], match.away, score.home, score.away, match.stats[match.home] || {}, match);
        applyTeamMatch(teams[match.away], match.home, score.away, score.home, match.stats[match.away] || {}, match);
    });

    Object.values(teams).forEach(team => {
        team.goalDifference = team.goalsFor - team.goalsAgainst;
        team.avgGoals = team.played ? team.goalsFor / team.played : 0;
        team.avgConceded = team.played ? team.goalsAgainst / team.played : 0;
        team.avgPossession = team.possessionSamples ? team.possessionTotal / team.possessionSamples : 0;
        team.shotAccuracy = team.shots ? team.shotsOnTarget / team.shots * 100 : 0;
        team.passAccuracy = team.passes ? team.accuratePasses / team.passes * 100 : 0;
        team.attackIndex = team.goalsFor * 5 + team.shotsOnTarget * 1.5 + team.shots * 0.35 + team.corners * 0.25;
        team.disciplineIndex = team.fouls + team.yellows * 3 + team.reds * 8;
    });

    const matchList = Object.values(matches).sort((a, b) => (a.date || '').localeCompare(b.date || '') || a.key.localeCompare(b.key));
    const teamList = Object.values(teams).sort((a, b) => b.points - a.points || b.goalDifference - a.goalDifference || b.goalsFor - a.goalsFor);
    const totals = {
        matches: matchList.length,
        goals: matchList.reduce((sum, m) => {
            const score = scoreFromValue(m.score);
            return sum + score.home + score.away;
        }, 0),
        shots: teamList.reduce((sum, t) => sum + t.shots, 0),
        shotsOnTarget: teamList.reduce((sum, t) => sum + t.shotsOnTarget, 0),
        cards: teamList.reduce((sum, t) => sum + t.yellows + t.reds, 0)
    };
    totals.avgGoals = totals.matches ? totals.goals / totals.matches : 0;
    totals.shotAccuracy = totals.shots ? totals.shotsOnTarget / totals.shots * 100 : 0;

    return { matches, matchList, teams, teamList, totals };
}

function getAnalytics() {
    return buildTournamentAnalytics();
}

function getTeamAnalytics(teamName) {
    const analytics = getAnalytics();
    const name = normalizeTeamName(teamName);
    return analytics.teams[name] || null;
}

function getMatchAnalytics(matchKey) {
    const analytics = getAnalytics();
    const [home, away] = splitMatchKey(matchKey);
    return analytics.matches[`${home} vs ${away}`] || null;
}

function renderMetricCard(value, label, sub = '') {
    return `
        <div class="overview-card">
            <div class="overview-val">${esc(value)}</div>
            <div class="overview-label">${esc(label)}</div>
            ${sub ? `<div class="overview-sub">${esc(sub)}</div>` : ''}
        </div>
    `;
}

function renderLeaderboard(title, teams, metric, formatter = value => value.toFixed ? value.toFixed(1) : value) {
    const max = Math.max(...teams.map(metric), 1);
    return `
        <div class="stats-panel">
            <div class="panel-title">${esc(title)}</div>
            <div class="top-teams-chart">
                ${teams.map(team => {
                    const value = metric(team);
                    return `
                        <div class="chart-row" onclick="showView('team-details', {teamName: ${jsArg(team.name)}})">
                            <div class="chart-team">${teamLogo(team.name, 28)}<span>${esc(team.name)}</span></div>
                            <div class="chart-bar-wrap"><div class="chart-bar" style="width:${Math.max(value / max * 100, 4)}%"></div></div>
                            <div class="chart-val">${esc(formatter(value))}</div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;
}

function renderTeamTable(teams) {
    return `
        <div class="stats-panel full-span">
            <div class="panel-title">Командный рейтинг</div>
            <div class="stats-table-wrap">
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Команда</th><th>И</th><th>В</th><th>Н</th><th>П</th><th>Голы</th><th>Удары</th><th>В створ</th><th>Влад.</th><th>Индекс</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${teams.map(team => `
                            <tr onclick="showView('team-details', {teamName: ${jsArg(team.name)}})">
                                <td class="table-team">${teamLogo(team.name, 24)}<span>${esc(team.name)}</span></td>
                                <td>${team.played}</td>
                                <td>${team.wins}</td>
                                <td>${team.draws}</td>
                                <td>${team.losses}</td>
                                <td>${team.goalsFor}:${team.goalsAgainst}</td>
                                <td>${team.shots}</td>
                                <td>${team.shotsOnTarget}</td>
                                <td>${team.avgPossession ? team.avgPossession.toFixed(1) + '%' : '-'}</td>
                                <td>${team.attackIndex.toFixed(1)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

function renderMatchCards(matches) {
    return `
        <div class="stats-panel full-span">
            <div class="panel-title">Матчи с расширенной статистикой</div>
            <div class="matches-detailed-list">
                ${matches.map(match => {
                    const h = match.stats[match.home] || {};
                    const a = match.stats[match.away] || {};
                    const score = scoreFromValue(match.score);
                    const totalShots = parseStatValue(h.totalShots) + parseStatValue(a.totalShots);
                    const totalSot = parseStatValue(h.shotsOnTarget) + parseStatValue(a.shotsOnTarget);
                    return `
                        <div class="detailed-card match-stat-card" onclick="showView('match-details', {matchKey: ${jsArg(match.key)}})">
                            <div class="dc-header">
                                <div class="dc-team">${teamLogo(match.home, 30)}<span>${esc(match.home)}</span></div>
                                <div class="dc-score">${score.home} : ${score.away}</div>
                                <div class="dc-team right"><span>${esc(match.away)}</span>${teamLogo(match.away, 30)}</div>
                            </div>
                            <div class="match-card-kpis">
                                <span>${totalShots} ударов</span>
                                <span>${totalSot} в створ</span>
                                <span>${match.venue ? esc(match.venue) : 'стадион уточняется'}</span>
                            </div>
                            <div class="spark-row">
                                <span>${formatPct(h.possessionPct)}</span>
                                <div class="spark-track"><i style="width:${parseStatValue(h.possessionPct) || 50}%"></i></div>
                                <span>${formatPct(a.possessionPct)}</span>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;
}

function buildStats() {
    const container = document.getElementById('stats-container');
    if (!container) return;

    const analytics = getAnalytics();
    if (!analytics.matchList.length) {
        container.innerHTML = '<div class="placeholder-msg">Статистика не загружена.</div>';
        return;
    }

    const teamsByGoals = [...analytics.teamList].sort((a, b) => b.goalsFor - a.goalsFor).slice(0, 8);
    const teamsByAttack = [...analytics.teamList].sort((a, b) => b.attackIndex - a.attackIndex).slice(0, 8);
    const teamsByPossession = [...analytics.teamList].filter(t => t.avgPossession).sort((a, b) => b.avgPossession - a.avgPossession).slice(0, 8);
    const topMatches = [...analytics.matchList].sort((a, b) => {
        const bs = scoreFromValue(b.score), as = scoreFromValue(a.score);
        return (bs.home + bs.away) - (as.home + as.away);
    }).slice(0, 12);

    container.innerHTML = `
        <div class="stats-hero">
            <div>
                <div class="eyebrow">ESPN data hub</div>
                <h2>Мега-статистика турнира</h2>
                <p>Матчи, командные рейтинги, форма, атакующие метрики, владение, передачи и дисциплина собраны в один живой дашборд.</p>
            </div>
            <div class="stats-hero-badge">${analytics.totals.matches} матчей</div>
        </div>

        <div class="stats-overview">
            ${renderMetricCard(analytics.totals.goals, 'Голов всего', `${analytics.totals.avgGoals.toFixed(2)} за матч`)}
            ${renderMetricCard(analytics.totals.shots, 'Ударов', `${analytics.totals.shotAccuracy.toFixed(1)}% в створ`)}
            ${renderMetricCard(analytics.teamList.length, 'Команд с данными', `${analytics.totals.cards} карточек`)}
        </div>

        <div class="analytics-grid">
            ${renderLeaderboard('Самые результативные', teamsByGoals, t => t.goalsFor, v => v.toFixed(0))}
            ${renderLeaderboard('Атакующий индекс', teamsByAttack, t => t.attackIndex, v => v.toFixed(1))}
            ${renderLeaderboard('Контроль мяча', teamsByPossession, t => t.avgPossession, v => `${v.toFixed(1)}%`)}
            ${renderTeamTable(analytics.teamList)}
            ${renderMatchCards(topMatches)}
        </div>
    `;
}
