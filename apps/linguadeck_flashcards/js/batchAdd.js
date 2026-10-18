// batchAdd.js
// Пакетное добавление из текста: одна строка = одна карточка, поля через разделитель:
//
//   front ; back ; транскрипция ; пример 1, пример 2, пример 3
//
// Медиа в пакетном режиме не поддерживается (добавляется руками через addCard).
// parse() — чистая функция без DOM и без базы, поэтому легко проверяется отдельно.
//
// Что понимает разбор:
//  • разделитель: авто (; Tab |) или заданный явно; пробелы вокруг полей не важны
//  • переводы строк \r\n, \n, \r; BOM в начале; пустые строки и строки из одних разделителей — пропускаются
//  • пустые поля в любом месте: "слово;перевод;;пример" (нет транскрипции), "слово;перевод;;" (хвостовые ;)
//  • меньше 4 полей — нормально; больше 4 — все лишние поля тоже считаются примерами
//  • поле в "кавычках" может содержать разделитель: "нарочитый; обдуманный"; кавычка внутри — ""
//    (случайная одиночная кавычка в тексте разбор не ломает)
//  • примеры внутри поля — через запятую; запятая внутри примера экранируется: \,
//  • необязательно: пропуск строки-заголовка, пропуск дубликатов

window.App = window.App || {};

App.batchAdd = (function () {
  var DELIMITERS = [";", "\t", "|"]; // порядок = приоритет при равенстве
  var DETECT_SAMPLE_LINES = 50;
  var MAX_FILE_BYTES = 5 * 1024 * 1024;

  function trim(s) { return s.trim(); } // trim() режет и NBSP

  function splitLines(text) {
    return String(text == null ? "" : text)
      .replace(/^\uFEFF/, "")
      .split(/\r\n|\r|\n|\u2028|\u2029/);
  }

  // Берём тот разделитель, который встречается в наибольшем числе строк.
  function detectDelimiter(lines) {
    var counts = {};
    DELIMITERS.forEach(function (d) { counts[d] = 0; });
    var seen = 0;
    for (var i = 0; i < lines.length && seen < DETECT_SAMPLE_LINES; i++) {
      if (!lines[i].trim()) continue;
      seen++;
      DELIMITERS.forEach(function (d) {
        if (lines[i].indexOf(d) !== -1) counts[d]++;
      });
    }
    var best = DELIMITERS[0];
    DELIMITERS.forEach(function (d) {
      if (counts[d] > counts[best]) best = d;
    });
    return best;
  }

  // Читает "…" начиная с символа ПОСЛЕ открывающей кавычки. Закрывающей считается
  // кавычка, после которой (с точностью до пробелов) идёт разделитель или конец строки;
  // "" — экранированная кавычка; любая другая кавычка — обычный символ.
  // null — закрывающей нет: тогда открывающая кавычка тоже обычный символ.
  function readQuoted(line, start, delim) {
    var value = "";
    var n = line.length;
    var i = start;
    while (i < n) {
      var ch = line[i];
      if (ch === '"') {
        if (line[i + 1] === '"') { value += '"'; i += 2; continue; }
        var j = i + 1;
        while (j < n && line[j] !== delim && /\s/.test(line[j])) j++;
        if (j >= n || line[j] === delim) return { value: value, end: j };
      }
      value += ch;
      i++;
    }
    return null;
  }

  function splitLine(line, delim) {
    var fields = [];
    var n = line.length;
    var i = 0;
    for (;;) {
      var j = i;
      while (j < n && line[j] !== delim && /\s/.test(line[j])) j++; // пробелы перед возможной кавычкой
      if (line[j] === '"') {
        var q = readQuoted(line, j + 1, delim);
        if (q) {
          fields.push(q.value);
          if (q.end >= n) return fields;
          i = q.end + 1;
          continue;
        }
      }
      var k = line.indexOf(delim, i);
      if (k === -1) {
        fields.push(line.slice(i));
        return fields;
      }
      fields.push(line.slice(i, k));
      i = k + 1;
    }
  }

  // "a, b\, c, ,d" → ["a", "b, c", "d"]
  function splitExamples(raw) {
    var out = [];
    var cur = "";
    for (var i = 0; i < raw.length; i++) {
      var ch = raw[i];
      if (ch === "\\" && raw[i + 1] === ",") { cur += ","; i++; }
      else if (ch === ",") { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur);
    return out.map(trim).filter(Boolean);
  }

  function dupKey(front, back) {
    return (front + "\u0000" + back).toLowerCase();
  }

  // options: { delimiter: "auto"|";"|"\t"|"|", skipHeader, skipDuplicates, allowEmptyBack,
  //            existing: [{front, back}] — карточки колоды для поиска дубликатов }
  // → { rows: [{line, front, back, transcription?, examples[]}],
  //     errors: [{line, raw, reason}], delimiter, headerLine }
  function parse(text, options) {
    options = options || {};
    var lines = splitLines(text);
    var delim = (options.delimiter && options.delimiter !== "auto")
      ? options.delimiter
      : detectDelimiter(lines);

    var rows = [];
    var errors = [];
    var headerLine = null;
    var headerPending = !!options.skipHeader;
    var seen = {};
    if (options.skipDuplicates) {
      (options.existing || []).forEach(function (c) { seen[dupKey(trim(c.front), trim(c.back))] = "deck"; });
    }

    lines.forEach(function (raw, idx) {
      var lineNo = idx + 1;
      if (!raw.trim()) return;

      if (headerPending) {
        headerPending = false;
        headerLine = lineNo;
        return;
      }

      var fields = splitLine(raw, delim).map(trim);
      if (fields.every(function (f) { return !f; })) return; // ";;;" — считаем пустой строкой

      var front = fields[0];
      var back = fields[1] || "";
      if (!front) {
        errors.push({ line: lineNo, raw: raw, reason: "нет лицевой стороны" });
        return;
      }
      if (!back && !options.allowEmptyBack) {
        errors.push({
          line: lineNo,
          raw: raw,
          reason: fields.length < 2 ? "нет обратной стороны (не найден разделитель?)" : "пустая обратная сторона"
        });
        return;
      }

      if (options.skipDuplicates) {
        var key = dupKey(front, back);
        if (seen[key]) {
          errors.push({
            line: lineNo,
            raw: raw,
            reason: seen[key] === "deck" ? "уже есть в колоде" : "повтор в тексте",
            duplicate: true
          });
          return;
        }
        seen[key] = "text";
      }

      var examples = [];
      for (var k = 3; k < fields.length; k++) {
        examples = examples.concat(splitExamples(fields[k]));
      }

      rows.push({
        line: lineNo,
        front: front,
        back: back,
        transcription: fields[2] || undefined,
        examples: examples
      });
    });

    return { rows: rows, errors: errors, delimiter: delim, headerLine: headerLine };
  }

  // Читает загруженный текстовый файл. UTF-8 (с BOM или без), UTF-16 (Блокнот «Юникод»),
  // а если в UTF-8 не читается — windows-1251 (старые русские .txt).
  async function readFile(file) {
    if (file.size > MAX_FILE_BYTES) {
      throw new Error("Файл слишком большой (максимум " + (MAX_FILE_BYTES >> 20) + " МБ).");
    }
    var buf = await file.arrayBuffer();
    var b = new Uint8Array(buf);
    if (b[0] === 0xFF && b[1] === 0xFE) return new TextDecoder("utf-16le").decode(buf);
    if (b[0] === 0xFE && b[1] === 0xFF) return new TextDecoder("utf-16be").decode(buf);
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(buf);
    } catch (e) {
      return new TextDecoder("windows-1251").decode(buf);
    }
  }

  // rows — из parse(). Все карточки в одной транзакции: либо все, либо ни одной.
  async function add(deckId, rows) {
    if (!deckId) throw new Error("Не выбрана колода.");
    if (!rows || !rows.length) throw new Error("Нет карточек для добавления.");
    var ids = await App.cardManager.addCards(deckId, rows.map(function (r) {
      return { front: r.front, back: r.back, transcription: r.transcription, examples: r.examples };
    }));
    return { ids: ids, count: ids.length };
  }

  return {
    parse: parse,
    readFile: readFile,
    add: add,
    DELIMITERS: DELIMITERS
  };
})();
