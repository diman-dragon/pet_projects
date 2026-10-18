// render.js
// Отвечает только за отрисовку. Не хранит состояние и не решает бизнес-логику.
// Карточка гибкая: любое поле (картинка/транскрипция/примеры/аудио/видео) может отсутствовать.

window.App = window.App || {};

App.render = (function () {
  function setText(el, value) {
    el.textContent = value || "";
  }

  function setOptionalText(el, value) {
    if (value) {
      el.textContent = value;
      el.hidden = false;
    } else {
      el.textContent = "";
      el.hidden = true;
    }
  }

  function renderList(listEl, items) {
    listEl.innerHTML = "";
    if (!items || !items.length) {
      listEl.hidden = true;
      return;
    }
    items.forEach(function (item) {
      var li = document.createElement("li");
      li.textContent = item;
      listEl.appendChild(li);
    });
    listEl.hidden = false;
  }

  function renderImage(el, url) {
    if (url) {
      el.style.backgroundImage = "url('" + url + "')";
      el.classList.add("has-image");
    } else {
      el.style.backgroundImage = "";
      el.classList.remove("has-image");
    }
  }

  function renderMediaButton(btnEl, url) {
    btnEl.disabled = !url;
    btnEl.hidden = !url;
    btnEl.dataset.url = url || "";
  }

  // els — объект со ссылками на все узлы карточки (см. app.js)
  function renderCard(els, card) {
    if (!card) {
      setText(els.term, "Нет карточек");
      setText(els.description, "Нажмите «+», чтобы добавить карточки, или «⋯» → «Новая колода»");
      els.transcription.hidden = true;
      els.examples.hidden = true;
      renderImage(els.image, null);
      renderMediaButton(els.audioBtn, null);
      renderMediaButton(els.videoBtn, null);
      if (els.showCount) { els.showCount.textContent = ""; els.showCount.hidden = true; }
      return;
    }

    // Лицевая сторона
    setText(els.term, card.front);
    setOptionalText(els.transcription, card.transcription);
    renderImage(els.image, card.image);

    // Обратная сторона
    setText(els.description, card.back);
    renderList(els.examples, card.examples);

    // Медиа — общие для карточки, лежат поверх обеих сторон
    if (els.showCount) {
      els.showCount.textContent = "Показов: " + (card.showCount || 0);
      els.showCount.hidden = false;
    }
    renderMediaButton(els.audioBtn, card.audio);
    renderMediaButton(els.videoBtn, card.video);
  }

  function renderProgress(counterEl, index, total) {
    counterEl.textContent = total === 0 ? "0 / 0" : (index + 1) + " / " + total;
  }


  function renderDirectionOptions(selectEl, directions, currentDirectionId) {
    selectEl.innerHTML = "";
    directions.forEach(function (direction) {
      var opt = document.createElement("option");
      opt.value = direction.id;
      opt.textContent = direction.name;
      opt.selected = direction.id === currentDirectionId;
      selectEl.appendChild(opt);
    });
    selectEl.disabled = !directions.length;
  }

  function renderDeckOptions(selectEl, decks, currentDeckId) {
    selectEl.innerHTML = "";
    decks.forEach(function (deck) {
      var opt = document.createElement("option");
      opt.value = deck.id;
      opt.textContent = deck.name + " (" + deck.cards.length + ")";
      opt.selected = deck.id === currentDeckId;
      selectEl.appendChild(opt);
    });
    selectEl.disabled = !decks.length;
  }

  return {
    renderCard: renderCard,
    renderProgress: renderProgress,
    renderDirectionOptions: renderDirectionOptions,
    renderDeckOptions: renderDeckOptions
  };
})();
