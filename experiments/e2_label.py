"""E2 step 2 — local labelling tool (Python standard library only, opens in your browser).

    python experiments/e2_label.py --annotator siqi --pilot     # calibration round (30 items)
    python experiments/e2_label.py --annotator siqi             # full round
    python experiments/e2_label.py --annotator alex --dir path/to/e2   # a teammate with the zipped folder

Then open http://127.0.0.1:8765 . Answers are appended to <dir>/labels/<annotator>[_pilot].jsonl after every
item, so you can close the window at any time and continue later. Items are shown in a per-annotator random
order: all single-image items first, then the pairs.

Keys:  1 / 2 / 3  answer the highlighted question (yes / no / unsure;  A / B / same for pairs; 1-5 for the
overall rating) ·  Enter  save and next ·  ←  previous item ·  z  zoom the edited image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>E2 标注</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#ddd;--acc:#2f6fdb;--sel:#2f6fdb;--hi:#fff6d6}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
header{display:flex;gap:16px;align-items:center;padding:8px 16px;background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}
header b{font-size:15px}.bar{flex:1;height:6px;background:#e7e7e3;border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;padding:16px;max-width:1700px;margin:auto}
.imgs{display:flex;gap:10px;align-items:flex-start}.imgs figure{margin:0;flex:1;min-width:0}.imgs figure.src{flex:0 0 22%}
.imgs img{width:100%;max-height:78vh;object-fit:contain;background:#222;border-radius:6px;cursor:zoom-in}
figcaption{font-size:12px;color:var(--muted);margin:2px 0 4px}.big figcaption{font-weight:600;color:var(--ink);font-size:14px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;align-self:start;position:sticky;top:52px;max-height:calc(100vh - 64px);overflow:auto}
.task{background:#eef3fc;border-radius:6px;padding:8px 10px;margin-bottom:10px}
.q{padding:5px 8px;border-radius:6px;margin-bottom:2px}.q.cur{background:var(--hi)}.q p{margin:0 0 3px}.q small{color:var(--muted);display:block;margin:-2px 0 4px;font-size:12px;line-height:1.3}
.opts{display:flex;gap:6px;flex-wrap:wrap}.opts button{flex:1;min-width:44px;padding:4px;border:1px solid var(--line);background:#fff;border-radius:5px;cursor:pointer;font:inherit}
.opts button.on{background:var(--sel);color:#fff;border-color:var(--sel)}
textarea{width:100%;height:36px;font:inherit;border:1px solid var(--line);border-radius:5px;padding:4px}
.nav{display:flex;gap:8px;margin-top:8px}.nav button{flex:1;padding:8px;border-radius:6px;border:1px solid var(--line);background:#fff;cursor:pointer;font:inherit}
.nav button.go{background:var(--acc);color:#fff;border-color:var(--acc)}.nav button:disabled{opacity:.4}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;z-index:9;overflow:auto;cursor:zoom-out}#zoom img{display:block;margin:auto;max-width:none}
.done{max-width:600px;margin:80px auto;text-align:center}
@media(max-width:1000px){main{grid-template-columns:1fr}.panel{position:static}}
</style></head><body>
<header><b>E2 标注 · <span id="who"></span></b><div class="bar"><i id="prog"></i></div><span id="count"></span>
<button onclick="jumpFirst()">跳到第一张未标注</button></header>
<main id="main"></main><div id="zoom" onclick="this.style.display='none'"><img id="zimg"></div>
<script>
const YN=[["1","是"],["2","否"],["3","不确定"]], AB=[["1","A 更好"],["2","B 更好"],["3","差不多"]];
const SINGLE=[
 {k:"bg",t:it=>`背景是否换成了「${it.target}」，并且原来的「${it.source}」基本看不到了？`,h:"只是整体变色/变风格、原场景还在 → 否",o:YN},
 {k:"keep",t:it=>`主体（${it.subject==="dog"?"狗":"人"}）是否还是同一个，姿态和动作没有变？`,h:"随新场景变化的光照、色温、阴影不算改变；换了姿势、少了/多了肢体、换了衣服/毛色 → 否",o:YN},
 {k:"support",t:it=>`主体是否被一个合理的表面支撑着（不悬空、不陷进去、坐/站的位置说得通）？`,h:"例如坐在空中、脚在地面以下、坐在不可能坐的东西上 → 否",o:YN},
 {k:"shadow",t:it=>`接触处的阴影是否合理？`,h:"脚/身体与地面接触处应有阴影，方向和软硬大致符合场景光照；完全没有或明显错误 → 否",o:YN},
 {k:"light",t:it=>`主体的光照（亮度、色温、亮面方向）是否和新场景一致？`,h:"像是从另一张照片里剪下来贴上去的 → 否",o:YN},
 {k:"edge",t:it=>`主体边缘是否干净？`,h:"有光晕、残留旧背景（水、石头）、或有奇怪的物体贴着主体轮廓 → 否",o:YN},
 {k:"overall",t:it=>`作为一张「背景替换」结果，整体有多可信？`,h:"1 = 完全失败  2 = 明显问题  3 = 有瑕疵但能用  4 = 小瑕疵  5 = 看不出是编辑的",o:[["1","1"],["2","2"],["3","3"],["4","4"],["5","5"]]}];
const PAIR=[
 {k:"pref_overall",t:it=>`整体上哪一张是更好的「换到${it.target}」的结果？`,h:"同时考虑：背景是否换对、主体是否保留、是否融入新场景",o:AB},
 {k:"pref_keep",t:it=>`哪一张的主体保留得更好（同一个、姿态不变）？`,h:"随场景变化的光照/色调不算改变",o:AB},
 {k:"pref_physics",t:it=>`哪一张的主体和新场景在物理上更融合（支撑、阴影、光照、边缘）？`,h:"",o:AB}];
let S, idx=0, ans={}, cur=0, t0=Date.now();
async function init(){S=await (await fetch("api/state")).json();document.getElementById("who").textContent=S.annotator+(S.pilot?"（pilot）":"");
 idx=S.items.findIndex(it=>!S.answers[it.id]); if(idx<0) idx=S.items.length; render();}
function qs(it){return it.kind==="pair"?PAIR:SINGLE}
function progress(){const n=S.items.filter(it=>S.answers[it.id]).length;document.getElementById("count").textContent=`${n} / ${S.items.length}`;document.getElementById("prog").style.width=(100*n/S.items.length)+"%"}
function jumpFirst(){const i=S.items.findIndex(it=>!S.answers[it.id]);idx=i<0?S.items.length:i;render()}
function render(){progress();const m=document.getElementById("main");
 if(idx>=S.items.length){m.innerHTML=`<div class="done"><h2>全部完成，谢谢！</h2><p>标注已保存在 ${S.out}</p><p><button onclick="idx=0;render()">回到第一张检查</button></p></div>`;return}
 const it=S.items[idx];ans=Object.assign({},(S.answers[it.id]||{}).answers||{});cur=qs(it).findIndex(q=>!ans[q.k]);if(cur<0)cur=qs(it).length;t0=Date.now();
 const sub=it.subject==="dog"?"狗":"人";
 let imgs=`<figure class="src"><figcaption>原图（参照）</figcaption><img src="${it.before}" onclick="zoom(this.src)"></figure>`;
 if(it.kind==="pair") imgs+=`<figure class="big"><figcaption>A</figcaption><img src="${it.A}" onclick="zoom(this.src)"></figure><figure class="big"><figcaption>B</figcaption><img src="${it.B}" onclick="zoom(this.src)"></figure>`;
 else imgs+=`<figure class="big"><figcaption>编辑结果</figcaption><img id="main_img" src="${it.after}" onclick="zoom(this.src)"></figure>`;
 let html=`<div class="imgs">${imgs}</div><div class="panel"><div class="task"><b>${idx+1}/${S.items.length}</b>　指令：把${sub}（${it.pose||""}）的背景换成 <b>${it.target}</b>。原背景：${it.source}。</div>`;
 qs(it).forEach((q,i)=>{html+=`<div class="q ${i===cur?"cur":""}" id="q${i}" onclick="cur=${i};mark()"><p>${i+1}. ${q.t(it)}</p>${q.h?`<small>${q.h}</small>`:""}<div class="opts">`+
   q.o.map(([v,l])=>`<button class="${ans[q.k]===v?"on":""}" onclick="event.stopPropagation();setA(${i},'${v}')">${l}</button>`).join("")+`</div></div>`});
 html+=`<textarea id="note" placeholder="备注（可选：奇怪的地方、拿不准的原因）">${ans.note||""}</textarea>
  <div class="nav"><button onclick="prev()" ${idx===0?"disabled":""}>← 上一张</button><button class="go" id="next" onclick="save()">保存并下一张 ⏎</button></div></div>`;
 m.innerHTML=html;check();}
function mark(){document.querySelectorAll(".q").forEach((e,i)=>e.classList.toggle("cur",i===cur))}
function setA(i,v){const it=S.items[idx],q=qs(it)[i];ans[q.k]=v;
 document.querySelectorAll(`#q${i} button`).forEach(b=>b.classList.toggle("on",b.textContent===q.o.find(o=>o[0]===v)[1]));
 const n=qs(it).findIndex(q=>!ans[q.k]);cur=n<0?qs(it).length:n;mark();check()}
function check(){const it=S.items[idx];document.getElementById("next").disabled=!qs(it).every(q=>ans[q.k])}
async function save(){const it=S.items[idx];if(!qs(it).every(q=>ans[q.k]))return;ans.note=document.getElementById("note").value.trim();
 const rec={id:it.id,answers:Object.assign({},ans),ms:Date.now()-t0};
 const r=await fetch("api/answer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(rec)});
 if(!r.ok){alert("保存失败，请检查终端");return}S.answers[it.id]=rec;idx++;render();window.scrollTo(0,0)}
function prev(){if(idx>0){idx--;render()}}
function zoom(src){document.getElementById("zimg").src=src;document.getElementById("zoom").style.display="block"}
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA"){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();save()}return}
 const z=document.getElementById("zoom");if(z.style.display==="block"){if(e.key==="Escape"||e.key==="z")z.style.display="none";return}
 if(idx>=S.items.length)return;const it=S.items[idx],Q=qs(it);
 if(e.key==="Enter"){e.preventDefault();save();return}if(e.key==="ArrowLeft"){prev();return}
 if(e.key==="z"){const im=document.querySelector(".big img");if(im)zoom(im.src);return}
 if(e.key==="ArrowUp"){cur=Math.max(0,cur-1);mark();return}if(e.key==="ArrowDown"){cur=Math.min(Q.length-1,cur+1);mark();return}
 if(cur<Q.length&&Q[cur].o.some(o=>o[0]===e.key))setA(cur,e.key)});
init();
</script></body></html>"""


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if not ctype.startswith("image") else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "/api/state":
                body = {"annotator": state["annotator"], "pilot": state["pilot"], "items": state["items"],
                        "answers": state["answers"], "out": str(state["out"])}
                return self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")
            if p.startswith("/images/"):
                f = (state["dir"] / p.lstrip("/")).resolve()
                if f.parent == (state["dir"] / "images").resolve() and f.is_file():
                    return self._send(200, f.read_bytes(), "image/jpeg")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/answer":
                return self._send(404, b"", "text/plain")
            rec = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if rec.get("id") not in state["ids"]:
                return self._send(400, b"unknown item", "text/plain")
            rec.update(annotator=state["annotator"], pilot=state["pilot"], t=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(state["out"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["answers"][rec["id"]] = rec
            self._send(200, b"{}", "application/json")

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", required=True, help="short name, e.g. siqi")
    ap.add_argument("--dir", default=str(ROOT / "annotation" / "e2"), help="folder with manifest.json + images/")
    ap.add_argument("--pilot", action="store_true", help="only the calibration subset")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    singles, pairs = list(man["single"]), list(man["pair"])
    if args.pilot:
        keep = set(man["pilot"])
        singles = [s for s in singles if s["id"] in keep]
        pairs = [p for p in pairs if p["id"] in keep]
    # per-annotator order: single block first (absolute judgements without a comparison anchor), then pairs
    rng = random.Random(int(hashlib.md5(f"{args.annotator}|{args.pilot}".encode()).hexdigest()[:8], 16))
    rng.shuffle(singles)
    rng.shuffle(pairs)
    items = singles + pairs

    out = d / "labels" / f"{args.annotator}{'_pilot' if args.pilot else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    answers = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                answers[r["id"]] = r  # last answer wins
    state = {"annotator": args.annotator, "pilot": args.pilot, "items": items, "ids": {i["id"] for i in items},
             "answers": answers, "out": out, "dir": d}
    url = f"http://127.0.0.1:{args.port}"
    print(f"{len(items)} items ({len(singles)} single, {len(pairs)} pair), {len(answers)} already done")
    print(f"open {url}   (Ctrl+C to stop; answers are saved after every item to {out})")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
