/**
 * Russians names for all 48 teams
 */
var TEAM_NAMES_RU = {
  "Mexico": "Мексика",
  "South Africa": "ЮАР",
  "South Korea": "Южная Корея",
  "Czechia": "Чехия",
  "Canada": "Канада",
  "Bosnia and Herzegovina": "Босния и Герцеговина",
  "USA": "США",
  "Paraguay": "Парагвай",
  "Qatar": "Катар",
  "Switzerland": "Швейцария",
  "Brazil": "Бразилия",
  "Morocco": "Марокко",
  "Haiti": "Гаити",
  "Scotland": "Шотландия",
  "Australia": "Австралия",
  "Türkiye": "Турция",
  "Germany": "Германия",
  "Curacao": "Кюрасао",
  "Curaçao": "Кюрасао",
  "Netherlands": "Нидерланды",
  "Japan": "Япония",
  "Ivory Coast": "Кот-д'Ивуар",
  "Ecuador": "Эквадор",
  "Sweden": "Швеция",
  "Tunisia": "Тунис",
  "Spain": "Испания",
  "Cape Verde": "Кабо-Верде",
  "Belgium": "Бельгия",
  "Egypt": "Египет",
  "Saudi Arabia": "Саудовская Аравия",
  "Uruguay": "Уругвай",
  "Iran": "Иран",
  "New Zealand": "Новая Зеландия",
  "France": "Франция",
  "Senegal": "Сенегал",
  "Iraq": "Ирак",
  "Norway": "Норвегия",
  "Argentina": "Аргентина",
  "Algeria": "Алжир",
  "Austria": "Австрия",
  "Jordan": "Иордания",
  "Portugal": "Португалия",
  "DR Congo": "ДР Конго",
  "England": "Англия",
  "Croatia": "Хорватия",
  "Ghana": "Гана",
  "Panama": "Панама",
  "Uzbekistan": "Узбекистан",
  "Colombia": "Колумбия"
};

var TEAM_SHORT = {
  "Мексика":"МЕК","ЮАР":"ЮАР","Южная Корея":"КОР","Чехия":"ЧЕХ",
  "Канада":"КАН","Босния и Герцеговина":"БОС","США":"США","Парагвай":"ПАР",
  "Катар":"КАТ","Швейцария":"ШВА","Бразилия":"БРА","Марокко":"МАР",
  "Гаити":"ГАИ","Шотландия":"ШОТ","Австралия":"АВС","Турция":"ТУР",
  "Германия":"ГЕР","Кюрасао":"КЮР","Нидерланды":"НИД","Япония":"ЯПО",
  "Кот-д'Ивуар":"КИВ","Эквадор":"ЭКВ","Швеция":"ШВЕ","Тунис":"ТУН",
  "Испания":"ИСП","Кабо-Верде":"КАБ","Бельгия":"БЕЛ","Египет":"ЕГИ",
  "Саудовская Аравия":"САУ","Уругвай":"УРУ","Иран":"ИРА","Новая Зеландия":"НЗЛ",
  "Франция":"ФРА","Сенегал":"СЕН","Ирак":"ИРА","Норвегия":"НОР",
  "Аргентина":"АРГ","Алжир":"АЛЖ","Австрия":"АВТ","Иордания":"ИОР",
  "Португалия":"ПОР","ДР Конго":"ДРК","Англия":"АНГ","Хорватия":"ХОР",
  "Гана":"ГАН","Панама":"ПАН","Узбекистан":"УЗБ","Колумбия":"КОЛ"
};

function t(name) {
    return TEAM_NAMES_RU[name] || name;
}

// Global mapping for logo lookups
window.RU_TO_EN = {};
Object.entries(TEAM_NAMES_RU).forEach(([en, ru]) => { window.RU_TO_EN[ru] = en; });

// ESPN logo URLs for all 48 teams
var LOGOS = {
  "Mexico":                "https://a.espncdn.com/i/teamlogos/countries/500/mex.png",
  "South Africa":          "https://a.espncdn.com/i/teamlogos/countries/500/rsa.png",
  "South Korea":           "https://a.espncdn.com/i/teamlogos/countries/500/kor.png",
  "Czechia":               "https://a.espncdn.com/i/teamlogos/countries/500/cze.png",
  "Canada":                "https://a.espncdn.com/i/teamlogos/countries/500/can.png",
  "Bosnia and Herzegovina":"https://a.espncdn.com/i/teamlogos/countries/500/bih.png",
  "USA":                   "https://a.espncdn.com/i/teamlogos/countries/500/usa.png",
  "Paraguay":              "https://a.espncdn.com/i/teamlogos/countries/500/par.png",
  "Qatar":                 "https://a.espncdn.com/i/teamlogos/countries/500/qat.png",
  "Switzerland":           "https://a.espncdn.com/i/teamlogos/countries/500/sui.png",
  "Brazil":                "https://a.espncdn.com/i/teamlogos/countries/500/bra.png",
  "Morocco":               "https://a.espncdn.com/i/teamlogos/countries/500/mar.png",
  "Haiti":                 "https://a.espncdn.com/i/teamlogos/countries/500/hai.png",
  "Scotland":              "https://a.espncdn.com/i/teamlogos/countries/500/sco.png",
  "Australia":             "https://a.espncdn.com/i/teamlogos/countries/500/aus.png",
  "Türkiye":               "https://a.espncdn.com/i/teamlogos/countries/500/tur.png",
  "Germany":               "https://a.espncdn.com/i/teamlogos/countries/500/ger.png",
  "Curacao":               "https://a.espncdn.com/i/teamlogos/countries/500/cuw.png",
  "Curaçao":               "https://a.espncdn.com/i/teamlogos/countries/500/cuw.png",
  "Netherlands":           "https://a.espncdn.com/i/teamlogos/countries/500/ned.png",
  "Japan":                 "https://a.espncdn.com/i/teamlogos/countries/500/jpn.png",
  "Ivory Coast":           "https://a.espncdn.com/i/teamlogos/countries/500/civ.png",
  "Ecuador":               "https://a.espncdn.com/i/teamlogos/countries/500/ecu.png",
  "Sweden":                "https://a.espncdn.com/i/teamlogos/countries/500/swe.png",
  "Tunisia":               "https://a.espncdn.com/i/teamlogos/countries/500/tun.png",
  "Spain":                 "https://a.espncdn.com/i/teamlogos/countries/500/esp.png",
  "Cape Verde":            "https://a.espncdn.com/i/teamlogos/countries/500/cpv.png",
  "Belgium":               "https://a.espncdn.com/i/teamlogos/countries/500/bel.png",
  "Egypt":                 "https://a.espncdn.com/i/teamlogos/countries/500/egy.png",
  "Saudi Arabia":          "https://a.espncdn.com/i/teamlogos/countries/500/ksa.png",
  "Uruguay":               "https://a.espncdn.com/i/teamlogos/countries/500/uru.png",
  "Iran":                  "https://a.espncdn.com/i/teamlogos/countries/500/irn.png",
  "New Zealand":           "https://a.espncdn.com/i/teamlogos/countries/500/nzl.png",
  "France":                "https://a.espncdn.com/i/teamlogos/countries/500/fra.png",
  "Senegal":               "https://a.espncdn.com/i/teamlogos/countries/500/sen.png",
  "Iraq":                  "https://a.espncdn.com/i/teamlogos/countries/500/irq.png",
  "Norway":                "https://a.espncdn.com/i/teamlogos/countries/500/nor.png",
  "Argentina":             "https://a.espncdn.com/i/teamlogos/countries/500/arg.png",
  "Algeria":               "https://a.espncdn.com/i/teamlogos/countries/500/alg.png",
  "Austria":               "https://a.espncdn.com/i/teamlogos/countries/500/aut.png",
  "Jordan":                "https://a.espncdn.com/i/teamlogos/countries/500/jor.png",
  "Portugal":              "https://a.espncdn.com/i/teamlogos/countries/500/por.png",
  "DR Congo":              "https://a.espncdn.com/i/teamlogos/countries/500/cod.png",
  "England":               "https://a.espncdn.com/i/teamlogos/countries/500/eng.png",
  "Croatia":               "https://a.espncdn.com/i/teamlogos/countries/500/cro.png",
  "Ghana":                 "https://a.espncdn.com/i/teamlogos/countries/500/gha.png",
  "Panama":                "https://a.espncdn.com/i/teamlogos/countries/500/pan.png",
  "Uzbekistan":            "https://a.espncdn.com/i/teamlogos/countries/500/uzb.png",
  "Colombia":              "https://a.espncdn.com/i/teamlogos/countries/500/col.png",
};

// Emoji fallbacks (still used in some compact spots)
var FLAGS = {
  "Mexico":"🇲🇽","South Africa":"🇿🇦","South Korea":"🇰🇷","Czechia":"🇨🇿",
  "Canada":"🇨🇦","Bosnia and Herzegovina":"🇧🇦","USA":"🇺🇸","Paraguay":"🇵🇾",
  "Qatar":"🇶🇦","Switzerland":"🇨🇭","Brazil":"🇧🇷","Morocco":"🇲🇦",
  "Haiti":"🇭🇹","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Australia":"🇦🇺","Türkiye":"🇹🇷",
  "Germany":"🇩🇪","Curacao":"🇨🇼","Curaçao":"🇨🇼","Netherlands":"🇳🇱","Japan":"🇯🇵",
  "Ivory Coast":"🇨🇮","Ecuador":"🇪🇨","Sweden":"🇸🇪","Tunisia":"🇹🇳",
  "Spain":"🇪🇸","Cape Verde":"🇨🇻","Belgium":"🇧🇪","Egypt":"🇪🇬",
  "Saudi Arabia":"🇸🇦","Uruguay":"🇺🇾","Iran":"🇮🇷","New Zealand":"🇳🇿",
  "France":"🇫🇷","Senegal":"🇸🇳","Iraq":"🇮🇶","Norway":"🇳🇴",
  "Argentina":"🇦🇷","Algeria":"🇩🇿","Austria":"🇦🇹","Jordan":"🇯🇴",
  "Portugal":"🇵🇹","DR Congo":"🇨🇩","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Croatia":"🇭🇷",
  "Ghana":"🇬🇭","Panama":"🇵🇦","Uzbekistan":"🇺🇿","Colombia":"🇨🇴",
};

var DATE_LABELS = {
  "2026-06-11": "11 июня 2026 года, четверг",
  "2026-06-12": "12 июня 2026 года, пятница",
  "2026-06-13": "13 июня 2026 года, суббота",
  "2026-06-14": "14 июня 2026 года, воскресенье",
  "2026-06-15": "15 июня 2026 года, понедельник",
  "2026-06-16": "16 июня 2026 года, вторник",
  "2026-06-17": "17 июня 2026 года, среда",
  "2026-06-18": "18 июня 2026 года, четверг",
  "2026-06-19": "19 июня 2026 года, пятница",
  "2026-06-20": "20 июня 2026 года, суббота",
  "2026-06-21": "21 июня 2026 года, воскресенье",
  "2026-06-22": "22 июня 2026 года, понедельник",
  "2026-06-23": "23 июня 2026 года, вторник",
  "2026-06-24": "24 июня 2026 года, среда",
  "2026-06-25": "25 июня 2026 года, четверг",
  "2026-06-26": "26 июня 2026 года, пятница",
  "2026-06-27": "27 июня 2026 года, суббота",
  "2026-06-28": "28 июня 2026 года, воскресенье",
  "2026-06-29": "29 июня 2026 года, понедельник",
  "2026-06-30": "30 июня 2026 года, вторник",
  "2026-07-01": "01 июля 2026 года, среда",
  "2026-07-02": "02 июля 2026 года, четверг",
  "2026-07-03": "03 июля 2026 года, пятница",
  "2026-07-04": "04 июля 2026 года, суббота",
  "2026-07-05": "05 июля 2026 года, воскресенье",
  "2026-07-06": "06 июля 2026 года, понедельник",
  "2026-07-07": "07 июля 2026 года, вторник",
  "2026-07-09": "09 июля 2026 года, четверг",
  "2026-07-10": "10 июля 2026 года, пятница",
  "2026-07-11": "11 июля 2026 года, суббота",
  "2026-07-12": "12 июля 2026 года, воскресенье",
  "2026-07-14": "14 июля 2026 года, вторник",
  "2026-07-15": "15 июля 2026 года, среда",
  "2026-07-18": "18 июля 2026 года, суббота",
  "2026-07-19": "19 июля 2026 года, воскресенье",
};

// Returns <img> logo element for team
function teamLogo(team, size = 28) {
  const enName = window.RU_TO_EN[team] || team;
  const src = LOGOS[enName];
  const emoji = FLAGS[team] || FLAGS[enName] || '🏳️';
  if (!src) return `<span style="font-size:${size * 0.8}px">${emoji}</span>`;
  return `<img src="${src}" alt="${team}" width="${size}" height="${size}" 
    onerror="this.style.display='none'; this.insertAdjacentHTML('afterend', '<span style=\\'font-size:${size * 0.8}px\\'>${emoji}</span>')" 
    style="border-radius:3px;object-fit:contain;vertical-align:middle;display:inline-block">`;
}

// Returns emoji flag (small contexts)
function flag(team) { return FLAGS[team] || '🏳️'; }
