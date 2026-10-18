// seed.js
// Начальные данные: направление → колода → карточки.

window.App = window.App || {};

App.seed = (function () {
  var DIRECTIONS = [
    { id: "english", name: "Английский", position: 0 },
    { id: "history", name: "История", position: 1 },
    { id: "geography", name: "География", position: 2 }
  ];

  var DECKS = [
    {
      id: "english-b1", direction_id: "english", name: "English B1",
      cards: [
        { front: "ubiquitous", back: "повсеместный, вездесущий", transcription: "/juːˈbɪkwɪtəs/", image: "media/english-b1/ubiquitous.jpg", audio: "media/english-b1/ubiquitous.mp3", examples: ["Smartphones have become ubiquitous in modern life.", "Wi-Fi is now ubiquitous in cafes and offices."] },
        { front: "reluctant", back: "неохотный, несклонный", transcription: "/rɪˈlʌktənt/" },
        { front: "deliberate", back: "нарочитый; обдуманный" },
        { front: "resilient", back: "устойчивый, жизнестойкий" },
        { front: "ambiguous", back: "неоднозначный" }
      ]
    },
    {
      id: "history-dates", direction_id: "history", name: "Даты — история",
      cards: [
        { front: "1945", back: "Окончание Второй мировой войны" },
        { front: "1969", back: "Первая высадка на Луну" },
        { front: "1991", back: "Распад СССР" }
      ]
    },
    {
      id: "capitals", direction_id: "geography", name: "Столицы",
      cards: [
        { front: "Австралия", back: "Канберра", image: "media/capitals/australia.jpg" },
        { front: "Канада", back: "Оттава", audio: "media/capitals/canada.mp3" },
        { front: "Бразилия", back: "Бразилиа" },
        { front: "Швейцария", back: "Берн" }
      ]
    }
  ];

  function insertInto(dbInstance) {
    var directionStmt = dbInstance.prepare("INSERT INTO directions (id, name, position) VALUES (?, ?, ?);");
    var deckStmt = dbInstance.prepare("INSERT INTO decks (id, direction_id, name, position) VALUES (?, ?, ?, ?);");
    var cardStmt = dbInstance.prepare(
      "INSERT INTO cards (id, deck_id, position, front, back, transcription, image, audio, video, examples) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
    );

    DIRECTIONS.forEach(function (d) { directionStmt.run([d.id, d.name, d.position]); });
    DECKS.forEach(function (deck, deckIndex) {
      deckStmt.run([deck.id, deck.direction_id, deck.name, deckIndex]);
      deck.cards.forEach(function (card, cardIndex) {
        cardStmt.run([
          deck.id + "-" + cardIndex, deck.id, cardIndex, card.front, card.back,
          card.transcription || null, card.image || null, card.audio || null,
          card.video || null, card.examples ? JSON.stringify(card.examples) : null
        ]);
      });
    });

    directionStmt.free();
    deckStmt.free();
    cardStmt.free();
  }

  return { insertInto: insertInto };
})();
