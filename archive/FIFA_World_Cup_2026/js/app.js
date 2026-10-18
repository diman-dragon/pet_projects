// Состояние приложения
if (typeof initialized === 'undefined') {
    var initialized = { schedule: false, groups: false, playoff: false, stats: false };
}

var currentView = 'schedule';
var activeFilter = 'all';
var viewHistory = [];
var lastParams = {};

// Инициализация событий табов
document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
        viewHistory = []; // Сбрасываем историю при переключении табов
        showView(btn.dataset.view);
    });
});

function showView(name, params = {}, addToHistory = true) {
    if (addToHistory && currentView !== name) {
        viewHistory.push({ view: currentView, params: lastParams });
    }
    
    currentView = name;
    lastParams = params;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    
    // Пытаемся найти или создать вью
    let targetView = document.getElementById(`view-${name}`);
    if (!targetView) {
        // Если вью динамическая (например, детали матча), создаем контейнер
        targetView = document.createElement('div');
        targetView.id = `view-${name}`;
        targetView.className = 'view';
        document.querySelector('.main').appendChild(targetView);
    }
    targetView.classList.add('active');
    
    // Подсветка табов (только для основных разделов)
    document.querySelectorAll('.tab').forEach(t =>
        t.classList.toggle('active', t.dataset.view === name)
    );

    renderView(name, params);
}

function goBack() {
    if (viewHistory.length > 0) {
        const last = viewHistory.pop();
        showView(last.view, last.params, false);
    } else {
        showView('schedule', {}, false);
    }
}

function renderView(name, params = {}) {
    const container = document.getElementById(`view-${name}`);
    if (!container) return;

    switch(name) {
        case 'schedule': buildSchedule(activeFilter || 'all'); break;
        case 'groups':   buildGroups(); break;
        case 'master-table': buildMasterTable(); break;
        case 'playoff':  buildPlayoff(); break;
        case 'stats':    buildStats(); break;
        case 'match-details': renderMatchDetails(params.matchKey); break;
        case 'group-details': renderGroupDetails(params.groupLetter); break;
        case 'team-details':  renderTeamDetails(params.teamName); break;
    }
}

window.closeModal = function() {
    document.getElementById('modal-overlay').style.display = 'none';
};

// MATCH_STATS (js/data/match_stats.js) — это лёгкий датасет "счёт + дата",
// от которого напрямую зависят группы, турнирная таблица и счёт в расписании
// (см. js/components/groups.js и js/components/master-table.js).
// Он генерируется отдельно от FULL_MATCH_STATS / PRO_TOURNAMENT_DATA и может
// отставать на несколько последних сыгранных матчей даже после обновления
// данных — из-за этого свежие матчи "пропадают" из групп/таблицы/расписания,
// хотя в карточке матча (которая читает PRO_TOURNAMENT_DATA напрямую) данные
// уже есть. Здесь мы дособираем недостающие записи MATCH_STATS на лету из
// PRO_TOURNAMENT_DATA, чтобы все разделы приложения видели одни и те же матчи,
// даже если скрипты обновления данных запускались не по порядку.
function enrichMatchStats() {
    if (!window.MATCH_STATS) window.MATCH_STATS = { stats: {} };
    if (!window.MATCH_STATS.stats) window.MATCH_STATS.stats = {};

    const knownStats = window.MATCH_STATS.stats;

    // Дата матча в PRO_TOURNAMENT_DATA не хранится, поэтому берём её из
    // расписания, нормализуя названия команд под тот же формат ключей.
    const scheduleDates = {};
    if (typeof SCHEDULE !== 'undefined' && typeof normalizeTeamName === 'function') {
        SCHEDULE.forEach(m => {
            if (!m.home || !m.away || !m.date) return;
            const key = `${normalizeTeamName(m.home)} vs ${normalizeTeamName(m.away)}`;
            scheduleDates[key] = m.date;
        });
    }

    if (window.PRO_TOURNAMENT_DATA && window.PRO_TOURNAMENT_DATA.matches) {
        Object.entries(window.PRO_TOURNAMENT_DATA.matches).forEach(([key, info]) => {
            if (knownStats[key] || !info.score) return;
            const [home, away] = key.split(' vs ');
            const reverseKey = away !== undefined ? `${away} vs ${home}` : key;
            knownStats[key] = {
                score: info.score,
                date: scheduleDates[key] || scheduleDates[reverseKey] || null
            };
        });
    }
}

function syncStaticResults() {
    if (window.MATCH_STATS && window.MATCH_STATS.stats) {
        for (const [match, info] of Object.entries(window.MATCH_STATS.stats)) {
            const scoreParts = info.score.split(' - ');
            const hScore = parseInt(scoreParts[0]) || 0, aScore = parseInt(scoreParts[1]) || 0;
            RESULTS[match] = { score: `${hScore} : ${aScore}`, home: hScore, away: aScore, date: info.date };
        }
    }
}

// Старт
window.addEventListener('DOMContentLoaded', () => {
    enrichMatchStats();
    syncStaticResults();
    showView('schedule', {}, false);
});
