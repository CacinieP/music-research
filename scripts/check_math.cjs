// Render every extracted formula with the same pinned MathJax used by the site.
const fs = require('node:fs');
const formulas = JSON.parse(fs.readFileSync(process.argv[2] || 'site/math-formulas.json', 'utf8'));
if (!formulas.length) throw new Error('No formulas found in this mathematics site');
require('mathjax').init({loader: {load: ['input/tex', 'output/svg']}}).then(async MathJax => {
  const failures = [];
  for (const formula of formulas) {
    const node = await MathJax.tex2svgPromise(formula.tex, {display: formula.display});
    const output = MathJax.startup.adaptor.outerHTML(node);
    if (output.includes('data-mjx-error')) {
      failures.push({source: formula.source, tex: formula.tex, errors: [...output.matchAll(/data-mjx-error="([^"]*)"/g)].map(match => match[1])});
    }
  }
  console.log(`Rendered ${formulas.length} formulas; ${failures.length} errors`);
  if (failures.length) {
    console.error(JSON.stringify(failures, null, 2));
    process.exitCode = 1;
  }
}).catch(error => { console.error(error.message); process.exitCode = 1; });
