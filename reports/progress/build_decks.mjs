import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { decks } from './slides.mjs';
import { exportPdf } from './html_pdf.mjs';

const ROOT = path.resolve(import.meta.dirname, '../..');
const BUILD = path.join(ROOT, 'reports/progress/build');
const OUT = path.join(BUILD, 'output');
const SKILL = 'C:/Users/HAMI-PARS/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations';
const PYTHON = 'C:/Users/HAMI-PARS/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = 'C:/Users/HAMI-PARS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require = createRequire(path.join(BUILD, 'runtime.mjs'));
const { Presentation, PresentationFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const FONT = 'B Nazanin';
const INK = '#152B40';
const MUTED = '#4C6270';
const TEAL = '#117C83';
const PAPER = '#FBFCFA';
const rtl = text => '\u200f' + text.replace(/[A-Za-z][A-Za-z0-9+./-]*(?: +[A-Za-z][A-Za-z0-9+./-]*)*|[+−-]?[۰-۹0-9]+(?:[٫٬.,][۰-۹0-9]+)*(?:٪)?/g, value=>'\u2066'+value+'\u2069');
const fa = number => String(number).replace(/\d/g, c => '۰۱۲۳۴۵۶۷۸۹'[Number(c)]);
await fs.mkdir(OUT, { recursive: true });

function text(slide, content, x, y, w, h, size = 34, bold = false, color = INK, align = 'right') {
  const shape = slide.shapes.add({geometry:'textbox', position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text = rtl(content);
  shape.text.style = {fontSize:size,bold,color,typeface:FONT,alignment:align,verticalAlignment:'middle',insets:0,wrap:'square',autoFit:'none'};
  return shape;
}

function frame(slide, model, index, deck) {
  slide.background.fill = PAPER;
  text(slide, model.title, 68, 44, 1144, 92, 48, true);
  text(slide, `${deck.range}     اسلاید ${fa(index + 1)} از ۱۲`, 68, 675, 1144, 28, 21, false, MUTED);
  if (model.note) text(slide, model.note, 72, 611, 1136, 52, 26, false, TEAL);
  slide.speakerNotes.text = model.notes + '\n\n' + model.sources.map(p => 'شاهد: ' + p).join('\n');
}

function table(slide, model) {
  const values = [model.headers, ...model.rows];
  const height = Math.min(428, values.length * 76);
  const t = slide.tables.add({rows:values.length,columns:values[0].length,left:72,top:164,width:1136,height,values:values.map(row => row.map(rtl))});
  t.styleOptions = {headerRow:false,bandedRows:false};
  t.borders.assign({style:'solid',fill:'#D1DDDE',width:.7});
  for (let row=0; row<values.length; row++) {
    for (let col=0; col<values[0].length; col++) {
      const c=t.getCell(row,col);
      c.fill = row===0 ? INK : row%2===0 ? '#F0F5F3' : '#FFFFFF';
      c.text.style = {typeface:FONT,fontSize:32,bold:row===0,color:row===0?'#FFFFFF':INK,alignment:'right',verticalAlignment:'middle',insets:14};
    }
  }
}

async function image(slide, model) {
  const bytes = await fs.readFile(path.join(ROOT, model.image));
  slide.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:model.caption,fit:'contain',position:{left:72,top:156,width:model.wide?1136:600,height:420}});
  if(!model.wide) model.lines.forEach((line,i)=>text(slide,line,728,183+i*126,475,100,32));
  text(slide,model.caption,72,574,1136,50,24,false,MUTED);
}

function chart(slide, model) {
  const c=slide.charts.add('bar',{
    position:{left:72,top:164,width:1136,height:423},
    categories:model.categories,
    series:model.series.map((s,i)=>({...s,fill:i===0?TEAL:'#DA9351'})),
    barOptions:{direction:'column',grouping:'clustered',gapWidth:110},
    hasLegend:model.series.length>1,
    legend:{position:'bottom',textStyle:{fontSize:28,fill:INK}},
    xAxis:{textStyle:{fontSize:27,fill:INK},line:{fill:'#B5C5C6',width:1}},
    yAxis:{min:0,max:model.max,numberFormatCode:model.format||'0',textStyle:{fontSize:24,fill:MUTED},majorGridlines:{fill:'#DCE5E4',width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:model.format||'0',textStyle:{fontSize:26,fill:INK}},
    chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0}
  });
  applyPresentationChartFont(c,{fontFamily:FONT});
}

async function makeDeck(deck) {
  const presentation=Presentation.create({slideSize:{width:1280,height:720}});
  presentation.theme.defaultFont=FONT;
  for (const [index,model] of deck.slides.entries()) {
    const slide=presentation.slides.add();
    if(model.kind==='cover') {
      slide.background.fill=INK;
      text(slide,deck.label,88,80,1100,62,34,false,'#A8D8DA');
      text(slide,model.title,88,205,1100,160,64,true,'#FFFFFF');
      text(slide,model.subtitle,88,398,1100,150,36,false,'#D8E8E8');
      slide.speakerNotes.text=model.notes+'\n\n'+model.sources.map(p=>'شاهد: '+p).join('\n');
    } else {
      frame(slide,model,index,deck);
      if(model.kind==='table') table(slide,model);
      else if(model.kind==='image') await image(slide,model);
      else if(model.kind==='chart') chart(slide,model);
      else model.lines.forEach((line,i)=>text(slide,line,110,177+i*126,1096,106,39));
    }
  }
  const own=path.join(BUILD,deck.id);
  await fs.mkdir(own,{recursive:true});
  const candidate=path.join(own,'candidate.pptx');
  await (await PresentationFile.exportPptx(presentation)).save(candidate);
  execFileSync(PYTHON,[path.join(ROOT,'reports/progress/rtl_pptx.py'),candidate]);
  const version=process.env.REVISION||'v1';
  const final=path.join(OUT,`${deck.id}_${version}.pptx`);
  const tables=deck.slides.flatMap((m,i)=>m.kind==='table'?[i+1]:[]);
  const charts=deck.slides.flatMap((m,i)=>m.kind==='chart'?[i+1]:[]);
  await finalizePresentation({
    workspaceDir:BUILD,candidatePath:candidate,finalPath:final,pythonExecutable:PYTHON,
    integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],
    explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:charts,
    materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},
    verifyArtifactToolImport:true,receiptPath:path.join(own,`validation_${version}.json`)
  });
  console.log('FINALIZED',final);
  await exportPdf(deck,ROOT,BUILD,OUT,version);
  console.log('RENDERED',deck.id);
}

for(const deck of decks.filter(d=>!process.env.ONLY||d.id===process.env.ONLY)) await makeDeck(deck);
