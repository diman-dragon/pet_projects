/**
 * Returns the result object for a given home/away pair, or null.
 */
function getResult(home, away) {
  return RESULTS[`${home} vs ${away}`] || null;
}

/**
 * Returns the flag emoji for a team name.
 */
function flag(team) {
  return FLAGS[team] || '🏳️';
}
