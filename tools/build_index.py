#!/usr/bin/env python3
"""Build the static product-photo index (data/idx/*.json) from Open Food/Beauty/Products Facts exports.
Usage: python3 tools/build_index.py <workdir> <outdir>   (downloads exports into workdir if missing)"""
import sys,os,re,json,gzip,csv,unicodedata,collections,subprocess
import pyarrow.parquet as pq
W,OUT=sys.argv[1],sys.argv[2]; os.makedirs(W,exist_ok=True)
HF="https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/"
SRC={"food.parquet":HF+"food.parquet","beauty.parquet":HF+"beauty.parquet",
     "opf.csv.gz":"https://static.openproductsfacts.org/data/en.openproductsfacts.org.products.csv.gz"}
for f,u in SRC.items():
    p=os.path.join(W,f)
    if not os.path.exists(p) or os.path.getsize(p)<1000: subprocess.check_call(["curl","-sSL","--retry","5","-o",p,u])
PRIO={"en:"+c for c in "united-arab-emirates saudi-arabia kuwait qatar bahrain oman jordan lebanon iraq egypt morocco united-kingdom ireland india pakistan bangladesh philippines malaysia".split()}
MAX_TOTAL=int(os.environ.get("IDX_MAX","420000")); PER_BRAND=40; TOP_BRANDS=4000
def norm(s): s=unicodedata.normalize("NFKD",s or "").encode("ascii","ignore").decode().lower(); return re.sub(r"[^a-z0-9]+"," ",s).strip()
def pname(lst):
    if not lst: return "",""
    d={x["lang"]:x["text"] for x in lst if x and x.get("text")}
    for k in ("en","main"):
        if d.get(k): return d[k].strip(),k
    for k,v in d.items(): return v.strip(),k
    return "",""
def front(imgs,lang):
    best=None
    for im in imgs or []:
        k=im.get("key") or ""
        if not k.startswith("front") or im.get("rev") is None: continue
        if not (im.get("sizes") or {}).get("400"): continue
        l=k[6:] if k.startswith("front_") else ""
        r=(0 if l=="en" else 1 if l==lang else 2)
        if best is None or r<best[0]: best=(r,f"{l}.{im['rev']}")
    return best[1] if best else None
cat=lambda t:(t[-1].split(":",1)[-1].replace("-"," ")[:24] if t else "")
COLS=["code","product_name","brands","images","lang","countries_tags","quantity","categories_tags","unique_scans_n","obsolete"]
cands=[]   # (score, entry, brandkey)
def take(path,src):
    f=pq.ParquetFile(path)
    for b in f.iter_batches(batch_size=50000,columns=COLS):
        for r in b.to_pylist():
            if r.get("obsolete"): continue
            n,_=pname(r["product_name"]); code=(r["code"] or "").strip()
            if not n or not code.isdigit(): continue
            im=front(r["images"],r["lang"])
            if not im: continue
            br=(r["brands"] or "").strip()[:60]; bk=norm(br.split(",")[0]) if br else ""
            prio=bool(PRIO.intersection(r["countries_tags"] or []))
            cands.append(((r["unique_scans_n"] or 0)+(1e6 if prio else 0),[code,n[:90],br,(r["quantity"] or "")[:20],cat(r["categories_tags"]),im,src],bk,prio))
take(os.path.join(W,"food.parquet"),"f"); take(os.path.join(W,"beauty.parquet"),"b")
csv.field_size_limit(10**9)
with gzip.open(os.path.join(W,"opf.csv.gz"),"rt",encoding="utf-8",errors="ignore") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        u=r.get("image_url") or ""; m=re.search(r"/front_?([a-z]*)\.(\d+)\.\d+\.jpg",u); n=(r.get("product_name") or "").strip(); code=(r.get("code") or "").strip()
        if not (m and n and code.isdigit()): continue
        br=(r.get("brands") or "").strip()[:60]; prio=bool(PRIO.intersection((r.get("countries_tags") or "").split(",")))
        cands.append(((int(r.get("unique_scans_n") or 0) if (r.get("unique_scans_n") or "").isdigit() else 0)+(1e6 if prio else 0),
            [code,n[:90],br,(r.get("quantity") or "")[:20],cat((r.get("categories_tags") or "").split(",")) if r.get("categories_tags") else "",f"{m.group(1)}.{m.group(2)}","p"],norm(br.split(",")[0]) if br else "",prio))
print("candidates",len(cands))
bc=collections.Counter(c[2] for c in cands if c[2]); big={b for b,_ in bc.most_common(TOP_BRANDS)}
cands.sort(key=lambda c:-c[0]); per=collections.Counter(); keep=[]; seen=set()
for sc,e,bk,prio in cands:
    if e[0] in seen: continue
    if prio or (bk in big and per[bk]<PER_BRAND): keep.append(e); seen.add(e[0]); per[bk]+=1
    if len(keep)>=MAX_TOTAL: break
print("kept",len(keep))
shards=collections.defaultdict(list)
for e in keep:
    ks=set()
    for s in (e[2].split(",")[0],e[1]):
        t=norm(s).split()
        if t: ks.add(t[0][:2].ljust(2,"_"))
    for k in ks: shards[k].append(e)
os.makedirs(OUT,exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".json"): os.remove(os.path.join(OUT,f))
tot=0
for k,v in shards.items():
    s=json.dumps(v,ensure_ascii=False,separators=(",",":")); tot+=len(s)
    open(os.path.join(OUT,k+".json"),"w").write(s)
json.dump({"v":1,"n":len(keep),"shards":sorted(shards),"built":__import__("time").strftime("%Y-%m-%d")},open(os.path.join(OUT,"meta.json"),"w"))
print("shards",len(shards),"raw MB",round(tot/1e6,1))
