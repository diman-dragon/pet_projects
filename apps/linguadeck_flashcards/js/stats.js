// stats.js — статистика обучения из SQL-журнала study_events.
window.App = window.App || {};

App.stats = (function () {
  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>\"]/g, function (c) {
      return ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c];
    });
  }

  function queryAll(sql, params) { return App.db.query(sql, params || []); }

  function localDayKey(date) {
    return date.getFullYear() + "-" + String(date.getMonth() + 1).padStart(2, "0") + "-" + String(date.getDate()).padStart(2, "0");
  }

  function localDayStart(daysAgo) {
    var d = new Date();
    d.setHours(0, 0, 0, 0);
    d.setDate(d.getDate() - daysAgo);
    return d.getTime();
  }

  function summary(deckId) {
    var where = deckId ? " WHERE deck_id = ?" : "";
    var params = deckId ? [deckId] : [];
    var card = queryAll(
      "SELECT COUNT(*) AS total, " +
      "COALESCE(SUM(CASE WHEN learned = 1 THEN 1 ELSE 0 END), 0) AS learned, " +
      "COALESCE(SUM(show_count), 0) AS shows, " +
      "COALESCE(SUM(CASE WHEN due <= ? THEN 1 ELSE 0 END), 0) AS due " +
      "FROM cards" + where,
      [App.srs.today()].concat(params)
    )[0] || {};

    var evWhere = deckId ? " WHERE deck_id = ? AND event_type = 'grade'" : " WHERE event_type = 'grade'";
    var events = queryAll(
      "SELECT COUNT(*) AS n, COALESCE(SUM(CASE WHEN rating IN ('good','easy') THEN 1 ELSE 0 END),0) AS success FROM study_events" + evWhere,
      params
    )[0] || {};

    var rows = queryAll("SELECT DISTINCT date(created_at / 1000, 'unixepoch', 'localtime') AS day FROM study_events WHERE event_type IN ('grade','skip')" + (deckId ? " AND deck_id = ?" : "") + " ORDER BY day DESC;", deckId ? [deckId] : []);
    var days = {};
    rows.forEach(function (r) { if (r.day) days[r.day] = true; });
    var streak = 0;
    var cursor = new Date(); cursor.setHours(0, 0, 0, 0);
    while (days[localDayKey(cursor)]) { streak++; cursor.setDate(cursor.getDate() - 1); }

    var n = Number(events.n || 0);
    return {
      total: Number(card.total || 0), learned: Number(card.learned || 0), shows: Number(card.shows || 0),
      due: Number(card.due || 0), reviews: n, success: n ? Math.round(Number(events.success || 0) / n * 100) : 0, streak: streak
    };
  }

  function daily(deckId, days) {
    var since = localDayStart(days - 1);
    var sql = "SELECT date(created_at / 1000, 'unixepoch', 'localtime') AS day, " +
      "SUM(CASE WHEN event_type = 'grade' THEN 1 ELSE 0 END) AS reviews, " +
      "SUM(CASE WHEN event_type = 'show' THEN 1 ELSE 0 END) AS shows " +
      "FROM study_events WHERE created_at >= ?" + (deckId ? " AND deck_id = ?" : "") + " GROUP BY day ORDER BY day;";
    var rows = queryAll(sql, deckId ? [since, deckId] : [since]);
    var map = {};
    rows.forEach(function (r) { if (r.day) map[r.day] = r; });
    var out = [];
    for (var i = days - 1; i >= 0; i--) {
      var d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - i);
      var key = localDayKey(d), r = map[key] || {};
      out.push({
        label: d.toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" }),
        reviews: Number(r.reviews || 0), shows: Number(r.shows || 0)
      });
    }
    return out;
  }

  function ratings(deckId) {
    var sql = "SELECT rating, COUNT(*) AS n FROM study_events WHERE event_type = 'grade'" + (deckId ? " AND deck_id = ?" : "") + " GROUP BY rating;";
    var rows = queryAll(sql, deckId ? [deckId] : []);
    var out = { again: 0, hard: 0, good: 0, easy: 0 };
    rows.forEach(function (r) { if (out[r.rating] != null) out[r.rating] = Number(r.n); });
    return out;
  }

  function decks(deckId) {
    var rows = queryAll(
      "SELECT d.name AS deck, COUNT(c.id) AS total, " +
      "COALESCE(SUM(CASE WHEN c.learned = 1 THEN 1 ELSE 0 END),0) AS learned, " +
      "COALESCE(SUM(c.show_count),0) AS shows " +
      "FROM decks d LEFT JOIN cards c ON c.deck_id = d.id " +
      (deckId ? "WHERE d.id = ? " : "") + "GROUP BY d.id ORDER BY learned DESC, d.position;",
      deckId ? [deckId] : []
    );
    return rows.map(function (r) { return { deck: r.deck, total: Number(r.total || 0), learned: Number(r.learned || 0), shows: Number(r.shows || 0) }; });
  }

  function topCards(deckId) {
    return queryAll(
      "SELECT c.front, c.back, c.show_count, d.name AS deck FROM cards c JOIN decks d ON d.id = c.deck_id " +
      (deckId ? "WHERE c.deck_id = ? " : "") + "ORDER BY c.show_count DESC, c.position LIMIT 8;",
      deckId ? [deckId] : []
    );
  }

  function svgBar(data, valueKey, label, options) {
    options = options || {};
    var w = 760, h = options.height || 240, padL = 40, padR = 12, padT = 18, padB = 38;
    var innerW = w - padL - padR, innerH = h - padT - padB;
    var max = Math.max(1, data.reduce(function (m, d) { return Math.max(m, Number(d[valueKey] || 0)); }, 0));
    var step = data.length ? innerW / data.length : innerW;
    var barW = Math.max(4, step * 0.58);
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" class="stats-chart" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(label)+'">';
    svg += '<line x1="'+padL+'" y1="'+(padT+innerH)+'" x2="'+(w-padR)+'" y2="'+(padT+innerH)+'" class="chart-axis"/>';
    svg += '<text x="'+(padL-7)+'" y="'+(padT+5)+'" text-anchor="end" class="chart-value">'+max+'</text>';
    data.forEach(function (d, i) {
      var v = Number(d[valueKey] || 0), bh = v / max * innerH;
      var x = padL + i * step + (step - barW) / 2, y = padT + innerH - bh;
      svg += '<rect x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" width="'+barW.toFixed(1)+'" height="'+Math.max(0,bh).toFixed(1)+'" rx="4" class="chart-bar"><title>'+esc(d.label)+': '+v+'</title></rect>';
      if (data.length <= 14 || i % 2 === 0) svg += '<text x="'+(x+barW/2).toFixed(1)+'" y="'+(h-13)+'" text-anchor="middle" class="chart-label">'+esc(d.label)+'</text>';
    });
    if (!data.length || data.every(function(d){ return !Number(d[valueKey] || 0); })) {
      svg += '<text x="'+(w/2)+'" y="'+(h/2)+'" text-anchor="middle" class="chart-empty">Пока нет данных</text>';
    }
    svg += '</svg>';
    return svg;
  }

  function svgLine(data, valueKey, label) {
    var w=760,h=240,padL=40,padR=12,padT=18,padB=38,iw=w-padL-padR,ih=h-padT-padB;
    var max=Math.max(1,data.reduce(function(m,d){return Math.max(m,Number(d[valueKey]||0));},0));
    var points=data.map(function(d,i){var x=padL+(data.length===1?iw/2:i*iw/(data.length-1));var y=padT+ih-(Number(d[valueKey]||0)/max)*ih;return {x:x,y:y,d:d};});
    var path=points.map(function(p,i){return (i?'L':'M')+p.x.toFixed(1)+' '+p.y.toFixed(1);}).join(' ');
    var svg='<svg xmlns="http://www.w3.org/2000/svg" class="stats-chart" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(label)+'">';
    svg+='<line x1="'+padL+'" y1="'+(padT+ih)+'" x2="'+(w-padR)+'" y2="'+(padT+ih)+'" class="chart-axis"/>';
    svg+='<text x="'+(padL-7)+'" y="'+(padT+5)+'" text-anchor="end" class="chart-value">'+max+'</text>';
    if(points.length) { svg+='<path d="'+path+'" class="chart-line" fill="none"/>'; points.forEach(function(p,i){svg+='<circle cx="'+p.x.toFixed(1)+'" cy="'+p.y.toFixed(1)+'" r="3.5" class="chart-dot"><title>'+esc(p.d.label)+': '+Number(p.d[valueKey]||0)+'</title></circle>'; if(data.length<=14||i%2===0) svg+='<text x="'+p.x.toFixed(1)+'" y="'+(h-13)+'" text-anchor="middle" class="chart-label">'+esc(p.d.label)+'</text>';}); }
    if (!data.length || data.every(function(d){return !Number(d[valueKey]||0);})){svg+='<text x="'+(w/2)+'" y="'+(h/2)+'" text-anchor="middle" class="chart-empty">Пока нет данных</text>';}
    return svg+'</svg>';
  }

  function svgDonut(values, label) {
    var total = values.reduce(function(s,v){return s+v.value;},0), cx=120,cy=105,r=70,circ=2*Math.PI*r, offset=0;
    var svg='<svg xmlns="http://www.w3.org/2000/svg" class="stats-donut" viewBox="0 0 240 210" role="img" aria-label="'+esc(label)+'">';
    svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" class="donut-bg"/>';
    if(total){ values.forEach(function(v){var len=circ*v.value/total;svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" class="donut-segment '+v.key+'" stroke-dasharray="'+len+' '+(circ-len)+'" stroke-dashoffset="'+(-offset)+'"/>';offset+=len;}); }
    svg+='<text x="'+cx+'" y="'+(cy-2)+'" text-anchor="middle" class="donut-total">'+total+'</text><text x="'+cx+'" y="'+(cy+17)+'" text-anchor="middle" class="donut-caption">оценок</text>';
    values.forEach(function(v,i){var y=28+i*34;svg+='<rect x="190" y="'+(y-10)+'" width="12" height="12" rx="3" class="legend-'+v.key+'"/><text x="208" y="'+y+'" class="legend-label">'+esc(v.label)+' '+v.value+'</text>';});
    return svg+'</svg>';
  }

  function render(deckId) {
    var dialog = document.getElementById("stats-dialog");
    if (!dialog) return;
    try {
      var s=summary(deckId), dd=daily(deckId,14), rr=ratings(deckId), ds=decks(deckId), top=topCards(deckId);
      document.getElementById("stats-total").textContent=s.total;
      document.getElementById("stats-learned").textContent=s.learned;
      document.getElementById("stats-due").textContent=s.due;
      document.getElementById("stats-shows").textContent=s.shows;
      document.getElementById("stats-reviews").textContent=s.reviews;
      document.getElementById("stats-success").textContent=s.success+"%";
      document.getElementById("stats-streak").textContent=s.streak;
      document.getElementById("stats-daily-chart").innerHTML=svgLine(dd,"reviews","Повторения за последние 14 дней");
      document.getElementById("stats-shows-chart").innerHTML=svgBar(dd,"shows","Показы за последние 14 дней");
      document.getElementById("stats-decks-chart").innerHTML=svgBar(ds.slice(0,10).map(function(x){return {label:x.deck,value:x.learned};}),"value","Выученные карточки по колодам");
      document.getElementById("stats-ratings").innerHTML=svgDonut([
        {key:"again",label:"Снова",value:rr.again},{key:"hard",label:"Трудно",value:rr.hard},{key:"good",label:"Хорошо",value:rr.good},{key:"easy",label:"Легко",value:rr.easy}
      ],"Распределение оценок");
      document.getElementById("stats-top-cards").innerHTML=top.length?top.map(function(c){return '<div class="stats-card-row"><div><strong>'+esc(c.front)+'</strong><span>'+esc(c.back)+'</span></div><b>'+Number(c.show_count||0)+' показ.</b></div>';}).join(""):'<p class="stats-empty">Пока нет карточек.</p>';
    } catch(err) {
      console.error("[stats] render failed",err);
      document.getElementById("stats-daily-chart").innerHTML='<p class="stats-error">Не удалось построить статистику: '+esc(err.message||err)+'</p>';
    }
  }

  function open(deckId) {
    var dialog=document.getElementById("stats-dialog"); if(!dialog) return;
    var select=document.getElementById("stats-scope");
    select.innerHTML='<option value="">Все направления и колоды</option>';
    var dirs=App.deckManager.getDirections();
    dirs.forEach(function(dir){
      var group=document.createElement("optgroup"); group.label=dir.name;
      App.deckManager.getDecks().filter(function(d){return d.directionId===dir.id;}).forEach(function(d){var o=document.createElement("option");o.value=d.id;o.textContent=d.name;group.appendChild(o);});
      select.appendChild(group);
    });
    select.value=deckId||"";
    if(!dialog.open) dialog.showModal();
    // Один кадр после showModal() — гарантирует, что на всех движках графики
    // перерисуются уже внутри открытого диалога.
    requestAnimationFrame(function(){render(select.value||null);});
  }

  function init() {
    document.getElementById("stats-close").addEventListener("click",function(){document.getElementById("stats-dialog").close();});
    document.getElementById("stats-close-2").addEventListener("click",function(){document.getElementById("stats-dialog").close();});
    document.getElementById("stats-refresh").addEventListener("click",function(){render(document.getElementById("stats-scope").value||null);});
    document.getElementById("stats-scope").addEventListener("change",function(){render(this.value||null);});
  }
  return {init:init,open:open,render:render};
})();
