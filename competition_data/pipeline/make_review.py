"""Build review/index.html: one row per candidate with thumbnails, an inline player limited to the
candidate's time range, and fields to fill. Answers autosave in the browser and export to CSV."""
import csv, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
import sys
CAND = sys.argv[1] if len(sys.argv) > 1 else "candidates.csv"
PAGE = sys.argv[2] if len(sys.argv) > 2 else "index.html"
KEYV = sys.argv[3] if len(sys.argv) > 3 else "bigair_review_v1"
rows = list(csv.DictReader(open(HERE / CAND)))
data = json.dumps([{k: r[k] for k in ("clip_id", "video", "competition", "start", "end", "dur", "replay_flag", "chapter")} for r in rows])
html = r"""<!doctype html><html><head><meta charset="utf-8"><title>Jump review</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#6b7280;--line:#e5e7eb;--acc:#2563eb;--warn:#b45309}
body{margin:0;font:14px/1.4 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:5;background:#111827;color:#fff;padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header select,header button{font:inherit;padding:5px 9px;border-radius:6px;border:0}
header button{background:var(--acc);color:#fff;cursor:pointer}
#stats{margin-left:auto;color:#d1d5db}
.help{padding:10px 16px;color:var(--mute);max-width:1100px}
.row{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:10px 16px;padding:10px;display:grid;grid-template-columns:1fr 330px;gap:12px}
.row.done{border-left:5px solid #16a34a}.row.drop{opacity:.45;border-left:5px solid #9ca3af}
.row img{width:100%;border-radius:6px;display:block}
.meta{font-size:12px;color:var(--mute);margin-bottom:4px}.flag{color:var(--warn);font-weight:600}
summary{cursor:pointer;color:var(--acc);margin-top:6px;font-size:13px}video{width:100%;border-radius:6px;background:#000;margin-top:6px}
.f{display:grid;grid-template-columns:auto 1fr;gap:6px 8px;align-items:center}
.f input[type=text],.f input[type=number]{width:100%;padding:5px;border:1px solid var(--line);border-radius:6px;font:inherit;box-sizing:border-box}
.btns{display:flex;gap:6px;grid-column:1/-1}.btns button{flex:1;padding:6px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer;font:inherit}
.btns button.on{background:var(--ink);color:#fff}
@media(max-width:900px){.row{grid-template-columns:1fr}}
</style></head><body>
<header><b>Big Air jump review</b><select id="comp"></select>
<label><input type="checkbox" id="todo"> only unreviewed</label>
<button id="exp">Export CSV</button><span id="stats"></span></header>
<p class="help">For each candidate: press <b>Live jump</b> if it is a real run (not a replay, slow-mo, crowd or interview), otherwise <b>Drop</b>.
For live jumps fill athlete + attempt (from the name overlay, last image), tick <b>fall</b> if he/she falls, and add the official score if you have it
(X Games / Olympics results pages list scores per athlete and attempt). Player plays only the candidate's time range. Answers autosave in this browser; press <b>Export CSV</b> when done and save it as <code>long_cuts/review_labels.csv</code>.</p>
<div id="list"></div>
<script>
const C=__DATA__;const KEY='__KEY__';let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){};stats()};
const comps=[...new Set(C.map(c=>c.competition))];const sel=document.getElementById('comp');
sel.innerHTML='<option value="">all competitions</option>'+comps.map(c=>`<option>${c}</option>`).join('');
function stats(){const n=C.length,d=C.filter(c=>S[c.clip_id]&&S[c.clip_id].status).length,k=C.filter(c=>S[c.clip_id]&&S[c.clip_id].status=='keep').length;
document.getElementById('stats').textContent=`${d}/${n} reviewed · ${k} live jumps`}
function render(){const L=document.getElementById('list');L.innerHTML='';const comp=sel.value,todo=document.getElementById('todo').checked;
for(const c of C){if(comp&&c.competition!=comp)continue;const s=S[c.clip_id]||(S[c.clip_id]={});if(todo&&s.status)continue;
const d=document.createElement('div');d.className='row '+(s.status=='keep'?'done':s.status=='drop'?'drop':'');
d.innerHTML=`<div><div class="meta">${c.clip_id} · ${c.start}–${c.end}s (${c.dur}s) · ${c.chapter||''} ${c.replay_flag=='1'?'<span class=flag>· maybe replay</span>':''}</div>
<img loading="lazy" src="thumbs/${c.clip_id}.jpg"><details><summary>▶ play this run</summary><video preload="none" controls src="../../long_videos/${c.video}.mp4#t=${c.start},${c.end}"></video></details></div>
<div class="f"><div class="btns"><button data-v="keep">Live jump</button><button data-v="drop">Drop</button></div>
<label>athlete</label><input type="text" data-k="athlete"><label>attempt</label><input type="number" min="1" max="4" data-k="attempt">
<label>fall</label><input type="checkbox" data-k="fall"><label>score</label><input type="number" step="0.01" min="0" max="100" data-k="score">
<label>note</label><input type="text" data-k="note"></div>`;
d.querySelectorAll('[data-k]').forEach(i=>{const k=i.dataset.k;if(i.type=='checkbox')i.checked=!!s[k];else i.value=s[k]??'';
i.oninput=()=>{s[k]=i.type=='checkbox'?i.checked:i.value;save()}});
d.querySelectorAll('[data-v]').forEach(b=>{b.classList.toggle('on',s.status==b.dataset.v);b.onclick=()=>{s.status=s.status==b.dataset.v?'':b.dataset.v;save();
d.className='row '+(s.status=='keep'?'done':s.status=='drop'?'drop':'');d.querySelectorAll('[data-v]').forEach(x=>x.classList.toggle('on',s.status==x.dataset.v))}});
L.appendChild(d)}stats()}
sel.onchange=render;document.getElementById('todo').onchange=render;
document.getElementById('exp').onclick=()=>{const h=['clip_id','video','competition','start','end','status','athlete','attempt','fall','score','note'];
const q=v=>{v=(v??'').toString();return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v};
const lines=[h.join(',')].concat(C.map(c=>{const s=S[c.clip_id]||{};return h.map(k=>k=='fall'?(s.fall?1:0):q(k in c?c[k]:s[k])).join(',')}));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/csv'}));a.download='review_labels.csv';a.click()};
render();
</script></body></html>"""
(HERE / "review" / PAGE).write_text(html.replace("__DATA__", data).replace("__KEY__", KEYV).replace("review_labels.csv", PAGE.replace(".html", "") + "_labels.csv" if PAGE != "index.html" else "review_labels.csv"))
print("wrote review/index.html with", len(rows), "candidates")
