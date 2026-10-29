import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

const escape = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replace(/[A-Za-z][A-Za-z0-9+./-]*(?: +[A-Za-z][A-Za-z0-9+./-]*)*|[+−-]?[۰-۹0-9]+(?:[٫٬.,][۰-۹0-9]+)*(?:٪)?/g, value=>`<bdi dir="ltr">${value}</bdi>`);
const fa = value => String(value).replace(/\d/g, c=>'۰۱۲۳۴۵۶۷۸۹'[Number(c)]);
const file = value => pathToFileURL(value).href;

function table(model) {
  const values=[model.headers,...model.rows];
  // Source columns are already ordered visually left-to-right with the subject at right.
  return '<table dir="ltr">'+values.map((row,index)=>'<tr>'+row.map(cell=>`<${index===0?'th':'td'} dir="rtl">${escape(cell)}</${index===0?'th':'td'}>`).join('')+'</tr>').join('')+'</table>';
}

function chart(model) {
  const max=model.max||Math.max(...model.series.flatMap(s=>s.values))*1.15;
  const colors=['#117C83','#DA9351'];
  const groups=model.categories.map((category,index)=>`<div class="bar-group"><div class="bars">${model.series.map((series,s)=>`<div class="bar-unit"><span class="value">${model.format?fa(series.values[index].toFixed(3)):fa(series.values[index])}</span><div class="bar" style="height:${330*series.values[index]/max}px;background:${colors[s]}"></div></div>`).join('')}</div><div class="category">${escape(category)}</div></div>`).join('');
  const legend=model.series.length>1?`<div class="legend">${model.series.map((series,index)=>`<span><i style="background:${colors[index]}"></i>${escape(series.name)}</span>`).join('')}</div>`:'';
  return `<div class="chart" dir="ltr">${groups}</div>${legend}`;
}

export async function exportPdf(deck,root,build,out,version) {
  const font=file('C:/Users/HAMI-PARS/AppData/Local/Microsoft/Windows/Fonts/B-NAZANIN.TTF');
  const pages=deck.slides.map((model,index)=>{
    if(model.kind==='cover') return `<section class="cover" dir="rtl"><div class="eyebrow">${escape(deck.label)}</div><h1>${escape(model.title)}</h1><div class="subtitle">${escape(model.subtitle).replaceAll('\n','<br>')}</div></section>`;
    let body='';
    if(model.kind==='table') body=table(model);
    else if(model.kind==='chart') body=chart(model);
    else if(model.kind==='image') body=`<div class="image-content${model.wide?' wide':''}"><img src="${file(path.join(root,model.image))}">${model.wide?'':`<div class="image-lines">${model.lines.map(line=>`<p>${escape(line)}</p>`).join('')}</div>`}</div><div class="caption">${escape(model.caption)}</div>`;
    else body=`<div class="lines">${model.lines.map(line=>`<p>${escape(line)}</p>`).join('')}</div>`;
    return `<section dir="rtl"><h2>${escape(model.title)}</h2><main>${body}</main>${model.note?`<aside>${escape(model.note)}</aside>`:''}<footer>${escape(deck.range)} &nbsp;&nbsp;&nbsp; اسلاید ${fa(index+1)} از ۱۲</footer></section>`;
  }).join('');
  const html=`<!doctype html><html lang="fa"><head><meta charset="utf-8"><style>
  @font-face{font-family:Nazanin;src:url('${font}')}*{box-sizing:border-box}html,body{margin:0;padding:0;font-family:Nazanin,'Times New Roman',serif;color:#152B40;background:#FBFCFA} @page{size:1280px 720px;margin:0}section{position:relative;width:1280px;height:720px;break-after:page;padding:44px 72px 70px;background:#FBFCFA;overflow:hidden}section:last-child{break-after:auto}h2{font-size:48px;font-weight:bold;margin:0;height:92px;display:flex;align-items:center}main{margin-top:28px;height:428px}p{margin:0}footer{position:absolute;bottom:17px;right:72px;left:72px;color:#4C6270;font-size:21px;height:28px}aside{position:absolute;top:611px;left:72px;right:72px;height:52px;font-size:26px;line-height:1.3;display:flex;align-items:center;color:#117C83}.cover{background:#152B40;padding:80px 88px}.eyebrow{font-size:34px;color:#A8D8DA}.cover h1{font-size:64px;color:white;margin:70px 0 28px;line-height:1.4;min-height:160px}.subtitle{font-size:36px;color:#D8E8E8;line-height:1.7}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:32px}th,td{border:1px solid #D1DDDE;padding:13px 16px;text-align:right;height:76px;line-height:1.2;vertical-align:middle}th{background:#152B40;color:white;font-weight:bold}tr:nth-child(odd) td{background:#F0F5F3}.lines p{font-size:39px;height:126px;display:flex;align-items:center}.image-content{display:flex;direction:ltr;height:420px;gap:55px}.image-content img{width:600px;height:420px;object-fit:contain}.image-lines{direction:rtl;width:475px;padding-top:20px}.image-lines p{font-size:32px;line-height:1.5;height:126px;display:flex;align-items:center}.caption{font-size:24px;color:#4C6270;margin-top:5px;line-height:1.2}.chart{display:flex;height:370px;gap:40px;align-items:flex-end;border-bottom:1px solid #B5C5C6}.bar-group{flex:1;min-width:0;height:370px;position:relative}.bars{height:330px;display:flex;gap:12px;align-items:flex-end;justify-content:center}.bar-unit{width:62px;display:flex;flex-direction:column;align-items:center}.bar{width:62px}.value{font-size:26px;white-space:nowrap;line-height:30px}.category{font-size:27px;text-align:center;direction:rtl;white-space:nowrap;margin-top:7px}.legend{display:flex;justify-content:center;gap:45px;font-size:26px;margin-top:22px}.legend i{display:inline-block;width:18px;height:18px;margin-left:10px}
  h2,.lines p,.image-lines p,aside{display:block;align-content:center}.wide img{width:1136px}.wide{gap:0}
  </style></head><body>${pages}</body></html>`;
  const htmlPath=path.join(build,deck.id,`slides_${version}.html`);
  const pdfPath=path.join(out,`${deck.id}_${version}.pdf`);
  await fs.writeFile(htmlPath,html);
  execFileSync('C:/Program Files/Google/Chrome/Application/chrome.exe',[
    '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--no-pdf-header-footer',
    `--user-data-dir=${path.join(build,'chrome-print-profile')}`,`--print-to-pdf=${pdfPath}`,file(htmlPath)
  ],{timeout:60000,stdio:'pipe'});
  return pdfPath;
}
