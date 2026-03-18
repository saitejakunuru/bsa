"""
Bank Statement Analyzer — API + Web UI
========================================
Single-file FastAPI server. Keeps things simple for a PoC.

Endpoints:
  POST /api/analyze   — Full pipeline: extract → classify → score
  GET  /api/health    — Health check
  GET  /              — Web UI

WHY FastAPI:
  - Async by default, automatic OpenAPI docs at /docs
  - Pydantic integration (same models for validation + serialization)
  - Minimal boilerplate vs Flask/Django
  Alternative: Flask — simpler, but no auto-docs or async.
"""

from __future__ import annotations
import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from extractor import extract_from_file
from classifier import classify_transactions
from scorer import compute_risk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bsa")

app = FastAPI(title="Bank Statement Analyzer", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm_configured": bool(config.GROQ_API_KEY)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Full pipeline: ingest → extract → classify → score."""
    logger.info(f"Analyzing: {file.filename}")

    suffix = Path(file.filename or "upload.pdf").suffix
    if suffix.lower() not in (".pdf", ".png", ".jpg", ".jpeg"):
        raise HTTPException(400, "Supported formats: PDF, PNG, JPG")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 1. Extract
        logger.info("Step 1/3: Extraction")
        account, transactions, page_count, warnings = extract_from_file(tmp_path)

        # 2. Classify
        logger.info("Step 2/3: Classification")
        transactions = classify_transactions(transactions)

        # 3. Score
        logger.info("Step 3/3: Risk Scoring")
        risk = compute_risk(transactions, account.account_holder, account.account_number)

        result = {
            "account": account.model_dump(mode="json"),
            "transactions": [t.model_dump(mode="json") for t in transactions],
            "risk_summary": risk.model_dump(mode="json"),
            "warnings": warnings,
            "page_count": page_count,
        }

        # Save sample output
        output_path = Path(__file__).parent / "output" / "sample_output.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_UI


# ═══════════════════════════════════════════════════════════════════════
# INLINE HTML UI
# Simple, functional, no build step.
# WHY inline: For a PoC, a separate templates/ dir is overkill.
# ═══════════════════════════════════════════════════════════════════════

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BSA — Bank Statement Analyzer</title>
<style>
  :root { --bg:#f8f9fa; --card:#fff; --border:#e0e0e0; --text:#1a1a1a;
          --dim:#666; --accent:#2563eb; --green:#16a34a; --red:#dc2626; --yellow:#d97706; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }
  .wrap { max-width:1100px; margin:0 auto; padding:24px; }
  h1 { font-size:22px; margin-bottom:4px; }
  .sub { color:var(--dim); font-size:14px; margin-bottom:24px; }

  /* Upload */
  .upload { background:var(--card); border:2px dashed var(--border); border-radius:12px;
            padding:40px; text-align:center; margin-bottom:24px; }
  .upload.dragover { border-color:var(--accent); background:#f0f6ff; }
  .btn { background:var(--accent); color:#fff; border:none; padding:10px 28px; border-radius:8px;
         font-size:14px; font-weight:600; cursor:pointer; }
  .btn:hover { opacity:0.9; }
  .btn:disabled { opacity:0.4; cursor:default; }
  #file-input { display:none; }
  .fname { margin-top:10px; font-size:13px; color:var(--accent); }

  /* Progress */
  .progress { display:none; margin-bottom:24px; }
  .pbar-bg { height:4px; background:#e5e7eb; border-radius:2px; }
  .pbar { height:100%; background:var(--accent); border-radius:2px; transition:width .5s; }
  .plabel { font-size:13px; color:var(--dim); margin-top:6px; }

  /* Results */
  .results { display:none; }

  /* Score banner */
  .score-banner { background:var(--card); border-radius:12px; padding:28px; text-align:center;
                  border:1px solid var(--border); margin-bottom:20px; }
  .score-num { font-size:56px; font-weight:800; }
  .score-of { font-size:16px; color:var(--dim); }
  .rating { font-size:20px; font-weight:700; margin:6px 0; }
  .decision { display:inline-block; padding:6px 20px; border-radius:6px; font-size:13px; font-weight:700; }

  /* Stats row */
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:16px 0; }
  .stat { background:var(--card); border-radius:10px; padding:14px; text-align:center; border:1px solid var(--border); }
  .stat .v { font-size:20px; font-weight:700; }
  .stat .l { font-size:12px; color:var(--dim); }

  /* Component cards */
  .comps { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-bottom:20px; }
  .comp { background:var(--card); border-radius:10px; padding:16px; border:1px solid var(--border); }
  .comp-hdr { display:flex; justify-content:space-between; margin-bottom:8px; }
  .comp-name { font-size:13px; color:var(--dim); }
  .comp-score { font-size:13px; font-weight:700; padding:2px 8px; border-radius:4px; }
  .comp-row { display:flex; justify-content:space-between; font-size:13px; padding:4px 0;
              border-bottom:1px solid #f0f0f0; }
  .comp-row:last-child { border:none; }

  /* Table */
  .tbl-wrap { background:var(--card); border-radius:10px; border:1px solid var(--border);
              padding:16px; margin-bottom:20px; overflow-x:auto; }
  .tbl-wrap h3 { font-size:15px; margin-bottom:12px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th { text-align:left; padding:8px; color:var(--dim); font-size:11px; text-transform:uppercase;
       border-bottom:1px solid var(--border); }
  td { padding:8px; border-bottom:1px solid #f5f5f5; }
  .cr { color:var(--green); } .dr { color:var(--red); }
  .tag { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px; font-weight:600; }
  .tag-cr { background:#dcfce7; color:var(--green); }
  .tag-dr { background:#fef2f2; color:var(--red); }
  .tag-cat { background:#eff6ff; color:var(--accent); }
  .tag-rev { background:#fef9c3; color:var(--yellow); }

  .rationale { margin-top:12px; }
  .rationale-item { font-size:13px; padding:8px 12px; background:#f8f9fa; border-radius:6px;
                    margin-bottom:4px; border-left:3px solid var(--accent); }
  .warn { font-size:13px; color:var(--yellow); padding:4px 0; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Bank Statement Analyzer</h1>
  <p class="sub">Upload a bank statement PDF to get a full credit risk assessment</p>

  <div class="upload" id="upload-area">
    <p style="font-size:15px;margin-bottom:8px">Drop a PDF here or click to browse</p>
    <p style="font-size:13px;color:var(--dim);margin-bottom:14px">Supports digital PDFs (SBI, HDFC, ICICI, etc.)</p>
    <input type="file" id="file-input" accept=".pdf,.png,.jpg,.jpeg">
    <button class="btn" onclick="document.getElementById('file-input').click()">Choose File</button>
    <div class="fname" id="fname"></div>
    <div style="margin-top:14px">
      <button class="btn" id="go-btn" onclick="run()" disabled>Analyze</button>
    </div>
  </div>

  <div class="progress" id="progress">
    <div class="pbar-bg"><div class="pbar" id="pbar" style="width:0%"></div></div>
    <div class="plabel" id="plabel">Starting...</div>
  </div>

  <div class="results" id="results"></div>
</div>

<script>
let file=null;
const ua=document.getElementById('upload-area');
ua.addEventListener('dragover',e=>{e.preventDefault();ua.classList.add('dragover')});
ua.addEventListener('dragleave',()=>ua.classList.remove('dragover'));
ua.addEventListener('drop',e=>{e.preventDefault();ua.classList.remove('dragover');if(e.dataTransfer.files.length)pick(e.dataTransfer.files[0])});
document.getElementById('file-input').addEventListener('change',e=>{if(e.target.files.length)pick(e.target.files[0])});

function pick(f){file=f;document.getElementById('fname').textContent=f.name+' ('+Math.round(f.size/1024)+'KB)';document.getElementById('go-btn').disabled=false}

async function run(){
  if(!file)return;
  const pb=document.getElementById('pbar'),pl=document.getElementById('plabel');
  document.getElementById('progress').style.display='block';
  document.getElementById('results').style.display='none';
  document.getElementById('go-btn').disabled=true;

  const steps=[[15,'Extracting text from PDF...'],[45,'Parsing transactions with AI...'],[75,'Classifying transactions...'],[90,'Computing risk score...']];
  let si=0;const iv=setInterval(()=>{if(si<steps.length){pb.style.width=steps[si][0]+'%';pl.textContent=steps[si][1];si++}},3000);

  try{
    const fd=new FormData();fd.append('file',file);
    const r=await fetch('/api/analyze',{method:'POST',body:fd});
    clearInterval(iv);
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Failed')}
    pb.style.width='100%';pl.textContent='Done!';
    render(await r.json());
  }catch(e){clearInterval(iv);pb.style.width='100%';pb.style.background='var(--red)';pl.textContent='Error: '+e.message}
  document.getElementById('go-btn').disabled=false;
}

function render(d){
  const el=document.getElementById('results');el.style.display='block';
  const rs=d.risk_summary, a=d.account, tx=d.transactions;
  const sc=rs.composite_score, col=sc>=700?'var(--green)':sc>=500?'var(--yellow)':'var(--red)';
  const dc={'APPROVE':'background:#dcfce7;color:var(--green)','APPROVE_WITH_CONDITIONS':'background:#fef9c3;color:var(--yellow)','REFER':'background:#fed7aa;color:#c2410c','DECLINE':'background:#fef2f2;color:var(--red)'}[rs.decision]||'';

  el.innerHTML=`
    <div class="score-banner">
      <div style="font-size:13px;color:var(--dim)">${a.account_holder||'Customer'} &bull; ${a.account_number} &bull; ${a.bank_name||''}</div>
      <div style="font-size:12px;color:var(--dim);margin-bottom:12px">${rs.analysis_period}</div>
      <div class="score-num" style="color:${col}">${sc}</div>
      <div class="score-of">/ 1000</div>
      <div class="rating" style="color:${col}">${rs.rating_band}</div>
      <div class="decision" style="${dc}">${rs.decision.replace(/_/g,' ')}</div>
      <div class="stats">
        <div class="stat"><div class="v cr">₹${fmt(rs.total_credits)}</div><div class="l">Credits</div></div>
        <div class="stat"><div class="v dr">₹${fmt(rs.total_debits)}</div><div class="l">Debits</div></div>
        <div class="stat"><div class="v" style="color:${rs.net_cashflow>=0?'var(--green)':'var(--red)'}">₹${fmt(rs.net_cashflow)}</div><div class="l">Net Flow</div></div>
        <div class="stat"><div class="v">${rs.transaction_count}</div><div class="l">Transactions</div></div>
      </div>
      <div class="rationale">${rs.rationale.map(r=>'<div class="rationale-item">'+r+'</div>').join('')}</div>
    </div>

    <div class="comps">
      ${rs.components.map(c=>{
        const s=c.score,cl=s>=700?'var(--green)':s>=500?'var(--yellow)':'var(--red)';
        const bg=s>=700?'#dcfce7':s>=500?'#fef9c3':'#fef2f2';
        return '<div class="comp"><div class="comp-hdr"><span class="comp-name">'+c.name+' ('+Math.round(c.weight*100)+'%)</span><span class="comp-score" style="background:'+bg+';color:'+cl+'">'+c.score+'</span></div>'+
          Object.entries(c.details).map(([k,v])=>'<div class="comp-row"><span>'+k.replace(/_/g,' ')+'</span><span style="font-weight:600">'+(typeof v==='number'?fmt(v):v)+'</span></div>').join('')+'</div>'
      }).join('')}
    </div>

    <div class="tbl-wrap">
      <h3>Transactions (${tx.length})</h3>
      <div style="max-height:420px;overflow-y:auto">
        <table><thead><tr><th>Date</th><th>Description</th><th>Counterparty</th><th>Channel</th><th>Debit</th><th>Credit</th><th>Balance</th><th>Category</th></tr></thead>
        <tbody>${tx.map(t=>'<tr><td>'+t.txn_date+'</td><td style="white-space:normal;max-width:200px;font-size:11px">'+t.description.substring(0,70)+'</td><td>'+t.counterparty+'</td><td>'+t.payment_channel+'</td><td class="dr">'+(t.debit?'₹'+fmt(t.debit):'')+'</td><td class="cr">'+(t.credit?'₹'+fmt(t.credit):'')+'</td><td>₹'+fmt(t.balance)+'</td><td><span class="tag '+(t.level1==='CREDIT'?'tag-cr':'tag-dr')+'">'+t.level1+'</span> <span class="tag tag-cat">'+t.level2+'</span>'+(t.needs_review?' <span class="tag tag-rev">REVIEW</span>':'')+'</td></tr>').join('')}</tbody></table>
      </div>
    </div>

    ${d.warnings.length?'<div>'+d.warnings.map(w=>'<div class="warn">⚠️ '+w+'</div>').join('')+'</div>':''}
  `;
}

function fmt(n){if(n==null)return'—';return Number(n).toLocaleString('en-IN',{maximumFractionDigits:2})}
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
