const { chromium } = require('/home/saurab/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:'/opt/google/chrome/chrome',args:['--no-sandbox','--disable-dev-shm-usage']});
const context=await browser.newContext({offline:true,viewport:{width:1440,height:1000}});const page=await context.newPage();const requests=[],errors=[];page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});page.on('pageerror',e=>errors.push(e.message));
await page.goto('file:///home/saurab/Documents/Stochastic-Proj/stochastic-pll-project/report/stochastic_pll_complete_report.html',{waitUntil:'load'});
await page.evaluate(async()=>{if(window.MathJax?.startup?.promise)await window.MathJax.startup.promise;});
console.log(JSON.stringify(await page.evaluate(()=>({title:document.title,sections:document.querySelectorAll('main>section').length,math:document.querySelectorAll('mjx-container').length,matherrors:[...document.querySelectorAll('[data-mjx-error]')].map(x=>x.textContent),overflow:document.documentElement.scrollWidth>innerWidth+1,body:document.body.innerText.slice(0,250)})),null,2));
console.log(JSON.stringify({requests,errors}));await page.screenshot({path:'/home/saurab/Documents/Stochastic-Proj/stochastic-pll-project/results/run_metadata/html_desktop_initial.png'});await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
