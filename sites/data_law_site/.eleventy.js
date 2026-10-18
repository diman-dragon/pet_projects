const Image = require("@11ty/eleventy-img");

module.exports = function(eleventyConfig) {
  
  // --- 0. IMAGE OPTIMIZATION ---
  eleventyConfig.addAsyncShortcode("image", async function(src, alt) {
    if (!alt) throw new Error(`Missing \`alt\` on image: ${src}`);
    let stats = await Image(src, {
      widths: [300, 600, 1200],
      formats: ["webp", "jpeg"],
      urlPath: "/img/",
      outputDir: "./public/img/"
    });
    return Image.generateHTML(stats, {
      alt,
      loading: "lazy",
      decoding: "async",
      sizes: "(min-width: 30em) 50vw, 100vw"
    });
  });

  // --- 1. ПРЯМОЕ КОПИРОВАНИЕ СТАТИКИ ---
  // src/css/* → css/* (покрывает tokens, global, components, pages)
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });
  eleventyConfig.addPassthroughCopy({ "src/css": "css" });
  eleventyConfig.addPassthroughCopy({ "src/js": "js" });

  // --- 2. КОМПОНЕНТЫ: HEADER ---
  eleventyConfig.addPassthroughCopy({ 
    "src/_includes/components/header/header.css": "css/header.css" 
  });

  // --- 3. КОМПОНЕНТЫ: FOOTER ---
  eleventyConfig.addPassthroughCopy({ 
    "src/_includes/components/footer/footer.css": "css/footer.css" 
  });

  // --- 4. КОМПОНЕНТЫ: LAYLA WIDGET ---
  eleventyConfig.addPassthroughCopy({ 
    "src/_includes/components/layla-widget/layla-widget.css": "css/layla-widget.css",
    "src/_includes/components/layla-widget/layla-core.css": "css/layla-core.css",
    "src/_includes/components/layla-widget/layla-theme.css": "css/layla-theme.css",
    "src/_includes/components/layla-widget/layla-window.css": "css/layla-window.css",
    "src/_includes/components/layla-widget/layla-messages.css": "css/layla-messages.css",
    "src/_includes/components/layla-widget/layla-input.css": "css/layla-input.css"
  });
  eleventyConfig.addPassthroughCopy({ 
    "src/_includes/components/layla-widget/layla-widget.js": "js/layla-widget.js" 
  });

  // --- 5. ДАННЫЕ ДЛЯ БРАУЗЕРА ---
  eleventyConfig.addPassthroughCopy({ "src/_data/layla-db.json": "data/layla-db.json" });
  eleventyConfig.addPassthroughCopy({ "src/_data/echr_decisions.json": "data/echr_decisions.json" });
  eleventyConfig.addPassthroughCopy({ "src/_data/legal_library.json": "data/legal_library.json" });

  // --- 6. СЛЕЖЕНИЕ ЗА ИЗМЕНЕНИЯМИ ---
  eleventyConfig.addWatchTarget("./src/css/");
  eleventyConfig.addWatchTarget("./src/js/");
  eleventyConfig.addWatchTarget("./src/_includes/components/");
  eleventyConfig.addWatchTarget("./src/_includes/ui/");
  eleventyConfig.addWatchTarget("./src/_includes/macros/");
  eleventyConfig.addWatchTarget("./src/_data/");

  // --- 7. НАСТРОЙКИ ДВИЖКА ---
  return {
    pathPrefix: "/data_law_site/",   // ← GitHub Pages: github.io/data_law_site/
    dir: {
      input: "src",
      output: "public",
      includes: "_includes",
      data: "_data"
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    dataTemplateEngine: "njk"
  };
};
