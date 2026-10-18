function getPlayerImage(player, size = 60) {
    const fallback = `data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22%23b8c0cc%22%3E%3Cpath d=%22M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z%22/%3E%3C/svg%3E`;
    if (player.headshot && player.headshot.href) {
        return `<img src="${esc(player.headshot.href)}" onerror="this.onerror=null; this.src='${fallback}';" class="player-photo" style="width:${size}px;height:${size}px;">`;
    }
    return `<img src="${fallback}" class="player-photo" style="width:${size}px;height:${size}px;">`;
}

function getTeamData(teamName) {
    const enName = window.RU_TO_EN && window.RU_TO_EN[teamName] ? window.RU_TO_EN[teamName] : teamName;
    const varName = 'TEAM_DATA_' + enName.replace(/-/g, '_').replace(/ /g, '_');
    return window[varName] || null;
}

function summarizeRoster(teamData) {
    const athletes = teamData && Array.isArray(teamData.athletes) ? teamData.athletes : [];
    const validHeights = athletes.map(p => parseStatValue(p.height) * 2.54).filter(Boolean);
    const validWeights = athletes.map(p => parseStatValue(p.weight) * 0.453592).filter(Boolean);
    const validAges = athletes.map(p => parseStatValue(p.age)).filter(Boolean);
    const avg = arr => arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
    return {
        total: athletes.length,
        avgAge: avg(validAges),
        avgHeight: avg(validHeights),
        avgWeight: avg(validWeights),
        tallest: [...athletes].sort((a, b) => parseStatValue(b.height) - parseStatValue(a.height))[0],
        youngest: [...athletes].sort((a, b) => parseStatValue(a.age) - parseStatValue(b.age))[0]
    };
}

function renderFormPills(matches) {
    return (matches || []).slice(-5).map(m => `<span class="form-pill ${m.result.toLowerCase()}">${m.result}</span>`).join('') || '<span class="muted">Нет матчей</span>';
}

function renderTeamMetric(value, label, sub = '') {
    return `
        <div class="team-metric">
            <b>${esc(value)}</b>
            <span>${esc(label)}</span>
            ${sub ? `<small>${esc(sub)}</small>` : ''}
        </div>
    `;
}

function renderRecentMatches(teamStats) {
    if (!teamStats || !teamStats.matches.length) {
        return '<div class="placeholder-msg">Матчи команды с детальной статистикой пока не найдены.</div>';
    }
    return `
        <div class="team-match-list">
            ${teamStats.matches.slice().reverse().map(match => `
                <div class="team-match-row" onclick="showView('match-details', {matchKey: ${jsArg(match.key)}})">
                    <span class="form-pill ${match.result.toLowerCase()}">${match.result}</span>
                    <div>
                        <b>${esc(match.score)}</b>
                        <span>vs ${esc(match.opponent)}</span>
                    </div>
                    <em>${match.date ? esc(match.date) : 'дата уточняется'}</em>
                </div>
            `).join('')}
        </div>
    `;
}

function renderStyleProfile(teamStats) {
    if (!teamStats) return '';
    const values = [
        { label: 'Атака', value: Math.min(teamStats.attackIndex, 100) },
        { label: 'Точность', value: teamStats.shotAccuracy },
        { label: 'Контроль', value: teamStats.avgPossession },
        { label: 'Пас', value: teamStats.passAccuracy },
        { label: 'Дисциплина', value: Math.max(0, 100 - teamStats.disciplineIndex / Math.max(teamStats.played, 1) * 4) }
    ];
    return `
        <div class="style-profile">
            ${values.map(item => `
                <div class="style-row">
                    <span>${esc(item.label)}</span>
                    <div><i style="width:${Math.max(0, Math.min(item.value, 100))}%"></i></div>
                    <b>${item.value ? item.value.toFixed(0) : '0'}</b>
                </div>
            `).join('')}
        </div>
    `;
}

function renderRosterSection(roster) {
    return Object.entries(roster).map(([pos, players]) => players.length > 0 ? `
        <div class="stats-panel">
            <div class="panel-title">${esc(pos)}</div>
            <div class="roster-grid">
                ${players.map(p => `
                    <div class="player-card" onclick="showPlayerModal(${jsArg(JSON.stringify(p))})">
                        ${getPlayerImage(p)}
                        <div class="player-name">${esc(p.displayName)}</div>
                        <div class="player-number">#${esc(p.jersey || '-')} · ${esc(p.position && p.position.displayName ? p.position.displayName : '')}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    ` : '').join('');
}

function renderTeamDetails(teamName) {
    const container = document.getElementById('view-team-details');
    if (!container) return;

    const displayName = normalizeTeamName(teamName);
    const teamData = getTeamData(displayName);
    const teamStats = typeof getTeamAnalytics === 'function' ? getTeamAnalytics(displayName) : null;

    if (!teamData) {
        container.innerHTML = `<div class="placeholder-msg">Данные о составе не найдены для команды ${esc(displayName)}.</div>`;
        return;
    }

    const coach = teamData.coach && teamData.coach[0] ? `${teamData.coach[0].firstName} ${teamData.coach[0].lastName}` : 'не указан';
    const rosterSummary = summarizeRoster(teamData);
    const positions = {
        Goalkeeper: 'Вратари',
        Defender: 'Защитники',
        Midfielder: 'Полузащитники',
        Forward: 'Нападающие'
    };
    const roster = {};
    Object.values(positions).forEach(pos => roster[pos] = []);
    (teamData.athletes || []).forEach(player => {
        const pos = positions[player.position && player.position.displayName] || 'Другие';
        if (!roster[pos]) roster[pos] = [];
        roster[pos].push(player);
    });

    container.innerHTML = `
        <div class="detail-header">
            <button class="back-btn" onclick="goBack()">← Назад</button>
            <div class="team-title-row">
                <img src="${esc(teamData.team.logo)}" style="width:64px;height:64px;object-fit:contain;">
                <div>
                    <h2 style="font-size: 2rem">${esc(displayName)}</h2>
                    <div class="team-subtitle">Главный тренер: ${esc(coach)}</div>
                </div>
            </div>
        </div>

        <div class="team-dashboard">
            <div class="stats-panel team-command-center">
                <div class="panel-title">Командная статистика</div>
                <div class="team-metrics-grid">
                    ${renderTeamMetric(teamStats ? `${teamStats.wins}-${teamStats.draws}-${teamStats.losses}` : '-', 'Форма', teamStats ? `${teamStats.points} очков` : '')}
                    ${renderTeamMetric(teamStats ? `${teamStats.goalsFor}:${teamStats.goalsAgainst}` : '-', 'Голы', teamStats ? `разница ${teamStats.goalDifference > 0 ? '+' : ''}${teamStats.goalDifference}` : '')}
                    ${renderTeamMetric(teamStats ? teamStats.shots : '-', 'Удары')}
                    ${renderTeamMetric(teamStats ? `${teamStats.shotAccuracy.toFixed(1)}%` : '-', 'Точность ударов')}
                    ${renderTeamMetric(teamStats && teamStats.avgPossession ? `${teamStats.avgPossession.toFixed(1)}%` : '-', 'Среднее владение')}
                    ${renderTeamMetric(teamStats && teamStats.passAccuracy ? `${teamStats.passAccuracy.toFixed(1)}%` : '-', 'Точность передач')}
                </div>
                <div class="form-strip">${renderFormPills(teamStats ? teamStats.matches : [])}</div>
            </div>

            <div class="stats-panel">
                <div class="panel-title">Профиль стиля</div>
                ${renderStyleProfile(teamStats)}
            </div>

            <div class="stats-panel">
                <div class="panel-title">Последние матчи</div>
                ${renderRecentMatches(teamStats)}
            </div>

            <div class="stats-panel">
                <div class="panel-title">Состав в цифрах</div>
                <div class="team-metrics-grid compact">
                    ${renderTeamMetric(rosterSummary.total, 'Игроков')}
                    ${renderTeamMetric(rosterSummary.avgAge ? rosterSummary.avgAge.toFixed(1) : '-', 'Средний возраст')}
                    ${renderTeamMetric(rosterSummary.avgHeight ? `${rosterSummary.avgHeight.toFixed(0)} см` : '-', 'Средний рост')}
                    ${renderTeamMetric(rosterSummary.avgWeight ? `${rosterSummary.avgWeight.toFixed(0)} кг` : '-', 'Средний вес')}
                </div>
                <div class="squad-highlights">
                    ${rosterSummary.tallest ? `<span>Самый высокий: <b>${esc(rosterSummary.tallest.displayName)}</b></span>` : ''}
                    ${rosterSummary.youngest ? `<span>Самый молодой: <b>${esc(rosterSummary.youngest.displayName)}</b></span>` : ''}
                </div>
            </div>
        </div>

        ${renderRosterSection(roster)}
    `;
}

window.showPlayerModal = function(playerJson) {
    const p = JSON.parse(playerJson);
    const modal = document.getElementById('modal-overlay');
    const titleEl = document.getElementById('modal-title');
    const msgEl = document.getElementById('modal-msg');
    const confirmBtn = document.getElementById('confirm-btn');
    const cancelBtn = document.querySelector('.modal-btn.secondary');

    titleEl.innerText = p.fullName || p.displayName;
    const heightCm = p.height ? Math.round(parseStatValue(p.height) * 2.54) : null;
    const weightKg = p.weight ? Math.round(parseStatValue(p.weight) * 0.453592) : null;

    msgEl.innerHTML = `
        ${getPlayerImage(p, 120)}
        <div class="player-modal-grid">
            <p><strong>Позиция:</strong> ${esc(p.position && p.position.displayName ? p.position.displayName : '-')}</p>
            <p><strong>Возраст:</strong> ${esc(p.age || '-')}</p>
            <p><strong>Рост:</strong> ${heightCm ? `${heightCm} см` : '-'}</p>
            <p><strong>Вес:</strong> ${weightKg ? `${weightKg} кг` : '-'}</p>
            <p><strong>Номер:</strong> #${esc(p.jersey || '-')}</p>
        </div>
    `;

    modal.style.display = 'flex';
    cancelBtn.style.display = 'none';
    confirmBtn.innerText = 'Закрыть';
    confirmBtn.onclick = closeModal;
};
