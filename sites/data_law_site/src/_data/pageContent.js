const fs = require("fs");
const path = require("path");

const pagesDir = path.join(__dirname, "..", "content", "pages");

const files = fs
  .readdirSync(pagesDir)
  .filter((file) => file.endsWith(".json"))
  .sort((a, b) => a.localeCompare(b));

const bySlug = {};
const list = files.map((file) => {
  const fullPath = path.join(pagesDir, file);
  const page = JSON.parse(fs.readFileSync(fullPath, "utf8"));

  bySlug[page.slug] = page;
  return page;
});

module.exports = {
  ...bySlug,
  bySlug,
  list,
};
