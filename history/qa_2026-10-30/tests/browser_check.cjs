let pw;try {pw=require(process.env.PLAYWRIGHT_MODULE||'playwright')} catch(e) {pw=require('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')}
const {chromium}=pw;
const fs=require('fs');const path=require('path');
(async()=>{
 const out=path.resolve(process.argv[2]);fs.mkdirSync(out,{recursive:true});
 const b=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await b.newPage({viewport:{width:1440,height:1250},acceptDownloads:true});
 await page.goto('http://127.0.0.1:8517');
 await page.getByRole('button',{name:'Оценить риск',exact:true}).waitFor({timeout:60000});
 await page.getByRole('button',{name:'Оценить риск',exact:true}).click();
 await page.getByText('Оценка вероятности отмены',{exact:true}).waitFor();
 await page.screenshot({path:path.join(out,'01_result.png'),fullPage:true});
 const dl=page.waitForEvent('download');await page.getByRole('button',{name:'Сохранить результат JSON'}).click();
 const download=await dl;await download.saveAs(path.join(out,'prediction.json'));
 const result=JSON.parse(fs.readFileSync(path.join(out,'prediction.json'),'utf8'));
 if(Math.abs(result.probability-0.2773894823554712)>1e-12||result.risk!=='Средний')throw Error('Export mismatch');
 for(const label of ['Взрослых','Детей','Младенцев'])await page.getByRole('spinbutton',{name:label,exact:true}).fill('0');
 await page.getByRole('button',{name:'Оценить риск',exact:true}).click();
 await page.getByText('Общее число гостей должно быть от 1 до 20.',{exact:true}).waitFor();
 await page.screenshot({path:path.join(out,'02_invalid_input.png'),fullPage:true});
 if(await page.getByText('Оценка вероятности отмены',{exact:true}).count())throw Error('Old result not cleared');
 await page.getByRole('tab',{name:'Качество и ограничения'}).click();
 await page.getByText('Фактическое сравнение на отложенной выборке',{exact:true}).waitFor();
 await page.screenshot({path:path.join(out,'03_quality.png'),fullPage:true});
 fs.writeFileSync(path.join(out,'browser_results.json'),JSON.stringify({date:new Date().toISOString(),browser:await b.version(),method:'Chrome headless, Playwright, real Streamlit HTTP server',checks:[{id:'B-01',status:'PASS',description:'Page loaded and example 2 result displayed'},{id:'B-02',status:'PASS',description:'Downloaded JSON matches example 2 prediction',result},{id:'B-03',status:'PASS',description:'Zero guests error displayed and old metrics removed'},{id:'B-04',status:'PASS',description:'Quality tab accessible'}]},null,2));
 await b.close();console.log('4 browser checks passed');
})().catch(e=>{console.error(e);process.exit(1)});
