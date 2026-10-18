// ESPN match data — Australia vs Türkiye (760421)
// Add more matches here as they come in, keyed by "Home vs Away"
const ESPN_DATA = {
  "Australia vs Türkiye": {
    espnId: "760421",
    home: "Australia",
    away: "Türkiye",
    score: "2 : 0",
    homeScore: 2,
    awayScore: 0,
    venue: "BC Place, Vancouver",
    attendance: 52497,
    date: "13.06.2026",
    time: "20:00",
    status: "FT",
    recap: "Nestory Irankunda and Connor Metcalfe scored and Australia spoiled Türkiye's return to the World Cup for the first time in 24 years with a 2-0 victory.",
    homeStats: { possession: 28.4, shots: 9, shotsOnTarget: 4, corners: 5, fouls: 12, assists: 1 },
    awayStats: { possession: 71.6, shots: 30, shotsOnTarget: 8, corners: 8, fouls: 4, assists: 0 },
    events: [
      { minute: "27'", type: "goal", team: "Australia", player: "Nestory Irankunda", jersey: "17" },
      { minute: "75'", type: "goal", team: "Australia", player: "Connor Metcalfe",   jersey: "8"  },
      { minute: "86'", type: "yellow", team: "Türkiye",  player: "Yunus Akgün",       jersey: "19" },
    ],
  },
};
