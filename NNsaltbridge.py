#!/usr/bin/env python
"""
saltbridge_predictor.py  ·  last updated 2025-07-11
---------------------------------------------------
build          – create feature CSV of existing Asp/Glu ↔ Arg/Lys pairs
train          – train feed-forward NN (early-stopping)
predict        – score existing salt bridges in one structure
design         – suggest NEW bridges (intra + inter)
design_inter   – suggest NEW bridges **inter-chain only**
"""

from __future__ import annotations
import argparse, csv, itertools as it, logging, math, os, signal, tempfile, time
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from pathlib import Path

import gemmi, numpy as np, pandas as pd
import torch, torch.nn as nn
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s: %(message)s")

# ─── global constants ──────────────────────────────────
RADIUS   = 8.0      # CB–CB neighbour shell
CUTOFF   = 4      # charged-atom dist ≤ CUTOFF ⇒ positive label
MAX_RES  = 30_000   # skip larger structures
TIMEOUT  = 1000       # sec/file

ACIDS, BASES = {'ASP','GLU'}, {'ARG','LYS'}
NEG_ATOMS = {'ASP':['OD1','OD2'], 'GLU':['OE1','OE2']}
POS_ATOMS = {'ARG':['NE','NH1','NH2'], 'LYS':['NZ']}
OFFSET = {'ASP':3.3, 'GLU':3.3, 'LYS':3.7, 'ARG':3.4}  # fake charge offset Å

HEADER = [
    'file','chain1','res1','chain2','res2',
    'ab_dist','ca_dist','cb_dist','seq_sep','same_chain','ori',
    'phi1','psi1','phi_bin1','psi_bin1','asa1','b1',
    'phi2','psi2','phi_bin2','psi_bin2','asa2','b2','label'
]
NUMERIC = [
    'ab_dist','ca_dist','cb_dist','seq_sep','same_chain','ori',
    'phi1','psi1','phi_bin1','psi_bin1','asa1','b1',
    'phi2','psi2','phi_bin2','psi_bin2','asa2','b2'
]

# ─── geometry helpers ───────────────────────────────────
def _find(res,nm):
    for fn in (lambda: res.find_atom(nm,''),                    # ≥ 0.6
               lambda: res.find_atom(nm,'',gemmi.Element('X')),
               lambda: res.find_atom(nm)):                      # older
        try: return fn()
        except Exception: pass
    for a in res:
        if a.name.strip()==nm.strip(): return a
    return None

_pos  = lambda r,n: (_find(r,n).pos if _find(r,n) else None)
_CA   = lambda r: _pos(r,'CA')
def _CB(r):                                    # gly → use CA
    p=_pos(r,'CB')
    return p if p or r.name!='GLY' else _pos(r,'CA')
_N, _C = lambda r: _pos(r,'N'), lambda r: _pos(r,'C')
_dist  = lambda a,b: None if a is None or b is None else a.dist(b)

def _tors(a,b,c,d):
    if None in (a,b,c,d): return None
    return math.degrees(gemmi.calculate_dihedral(a,b,c,d))
_phi = lambda p,c: _tors(_C(p),_N(c),_CA(c),_C(c)) if p else None
_psi = lambda c,n: _tors(_N(c),_CA(c),_C(c),_N(n)) if n else None
_bin = lambda a: 3 if a is None else (0 if a<-30 else 2 if a<50 else 1 if a<180 else 3)

def _is_atom(at): return at.element.is_atom() if hasattr(at.element,'is_atom') else getattr(at.element,'number',0)!=0
def _asa_map(model):
    if hasattr(gemmi,'ShrakeRupley'):
        sr=gemmi.ShrakeRupley(); sr.set_b_factors(True); sr.compute_b_factors(model)
        return {(c.name,r.seqid.num): sum(a.b_iso for a in r if _is_atom(a))
                for c in model for r in c}
    return {}
_bfac=lambda r: np.mean([a.b_iso for a in r if _is_atom(a)]) if any(_is_atom(a) for a in r) else 0.0

def charged_atoms(r):
    return [_find(r,n) for n in NEG_ATOMS.get(r.name,[])+POS_ATOMS.get(r.name,[]) if _find(r,n)]

def min_charged(r1,r2):
    d=None
    for a in charged_atoms(r1):
        for b in charged_atoms(r2):
            d0=a.pos.dist(b.pos); d=d0 if d is None or d0<d else d
    return d
min_charged_dist = min_charged    # alias for legacy

def orientation(r1,r2):
    cb1,cb2=_CB(r1),_CB(r2)
    if cb1 is None or cb2 is None: return 0.0
    ch1,ch2=charged_atoms(r1),charged_atoms(r2)
    if not ch1 or not ch2: return 0.0
    v1=np.array([ch1[0].pos.x-cb1.x,ch1[0].pos.y-cb1.y,ch1[0].pos.z-cb1.z])
    v2=np.array([ch2[0].pos.x-cb2.x,ch2[0].pos.y-cb2.y,ch2[0].pos.z-cb2.z])
    if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0: return 0.0
    return float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def fake_charge(cb,ca,rtype):
    if ca is None or cb is None: return None
    v=np.array([cb.x-ca.x,cb.y-ca.y,cb.z-ca.z]); n=np.linalg.norm(v)
    if n==0: return None
    v/=n; off=OFFSET[rtype]
    return gemmi.Position(cb.x+off*v[0], cb.y+off*v[1], cb.z+off*v[2])

# ─── timeout context ────────────────────────────────────
@contextmanager
def time_limit(sec,path):
    def _h(s,f): raise TimeoutError
    signal.signal(signal.SIGALRM,_h); signal.alarm(sec)
    try: yield
    except TimeoutError: logging.warning(f'{path.name} – timeout'); yield 'TIMEOUT'
    finally: signal.alarm(0)

# ─── per-file parser (for build) ────────────────────────
def _parse_one(path:Path):
    try:
        with time_limit(TIMEOUT,path):
            st=gemmi.read_structure(str(path))
            if st=='TIMEOUT': return []
    except Exception as e:
        logging.warning(f'{path.name} – parse error: {e}'); return []
    mdl=st[0]
    if sum(len(c) for c in mdl)>MAX_RES: return []

    acids,bases={},{}
    for c in mdl:
        for r in c:
            k=(c.name,r.seqid.num)
            if r.name in ACIDS: acids[k]=r
            elif r.name in BASES: bases[k]=r
    if not acids or not bases: return []

    asa=_asa_map(mdl)
    geom={}
    for c in mdl:
        rl=list(c)
        for i,r in enumerate(rl):
            prev=rl[i-1] if i else None; nxt=rl[i+1] if i<len(rl)-1 else None
            phi=(_phi(prev,r) or 0.0); psi=(_psi(r,nxt) or 0.0)
            geom[(c.name,r.seqid.num)]=(phi,psi,_bin(phi),_bin(psi),
                                        asa.get((c.name,r.seqid.num),0.0), _bfac(r))

    try:
        cell=getattr(mdl,'cell',None) or st.cell
        ns=gemmi.NeighborSearch(mdl,cell,RADIUS).populate()
        ns_mode=True
    except Exception:
        ns_mode=False

    def hits(pos):
        if ns_mode:
            try: return ns.find_atoms(pos,radius=RADIUS)
            except TypeError: return ns.find_atoms(pos,'\0',0,RADIUS)
        return [(_CB(rB),kB,rB) for kB,rB in bases.items()
                if (cb:=_CB(rB)) and pos.dist(cb)<=RADIUS]

    rows=[]
    for kA,rA in acids.items():
        cbA=_CB(rA);  # skip if CB missing
        if cbA is None: continue
        for h in hits(cbA):
            if ns_mode:
                atom=getattr(h,'atom',None)
                if atom is None and hasattr(h,'to_atom'):
                    try: atom=h.to_atom()
                    except Exception: atom=None
                if atom is None: continue
                rB=atom.get_parent_residue(); chainB=atom.get_parent_chain().name
            else:
                _,kB,rB = h; chainB=kB[0]
            kB=(chainB,rB.seqid.num)

            abd=min_charged_dist(rA,rB)
            if abd is None or abd>RADIUS: continue
            cad=_dist(_CA(rA),_CA(rB)); cbd=_dist(_CB(rA),_CB(rB))
            if None in (cad,cbd): continue
            seq=abs(kA[1]-kB[1]); same=int(kA[0]==kB[0])
            (phi1,psi1,pb1,qb1,asa1,b1)=geom[kA]
            (phi2,psi2,pb2,qb2,asa2,b2)=geom[kB]
            rows.append([path.name,kA[0],kA[1],kB[0],kB[1],
                         abd,cad,cbd,seq,same,orientation(rA,rB),
                         phi1,psi1,pb1,qb1,asa1,b1,
                         phi2,psi2,pb2,qb2,asa2,b2,
                         1 if abd<=CUTOFF else 0])
    return rows

# ─── build (stream CSV) ─────────────────────────────────
def build(a):
    files=[p for p in Path(a.data_dir).rglob('*')
           if p.suffix.lower() in {'.cif','.mmcif','.pdb','.gz'}]
    tmp=tempfile.NamedTemporaryFile('w',delete=False,newline='')
    csv.writer(tmp).writerow(HEADER)
    with Pool(a.nproc or cpu_count()) as pool:
        for rows in tqdm(pool.imap_unordered(_parse_one,files,1),
                         total=len(files),desc='Parsing'):
            if rows: csv.writer(tmp).writerows(rows); tmp.flush()
    tmp.close(); os.replace(tmp.name,a.out_csv)
    print('saved →',a.out_csv)

# ─── NN & train (unchanged) ─────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.m=nn.Sequential(nn.Linear(len(NUMERIC),64),nn.ReLU(),
                             nn.Linear(64,32),nn.ReLU(),
                             nn.Linear(32,1))
    def forward(self,x): return self.m(x)

def train(a):
    df=pd.read_csv(a.dataset)
    if len(df)==0: raise ValueError('Dataset empty')
    for c in ['ab_dist','ca_dist','cb_dist','seq_sep']: df[c]=np.log1p(df[c])
    for c in ['phi1','psi1','phi2','psi2']: df[c]=df[c]/180.0
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=NUMERIC)
    X=df[NUMERIC].values.astype('float32'); y=df['label'].values.astype('float32')
    idx=np.random.default_rng(0).permutation(len(df)); sp=int(.8*len(df))
    Xtr,Xva=torch.tensor(X[idx[:sp]]),torch.tensor(X[idx[sp:]])
    ytr,yva=torch.tensor(y[idx[:sp]])[:,None],torch.tensor(y[idx[sp:]])[:,None]
    net=Net(); opt=torch.optim.Adam(net.parameters(),1e-3); lossF=nn.BCEWithLogitsLoss()
    best=np.inf; stall=0
    for ep in range(1,a.epochs+1):
        net.train(); opt.zero_grad()
        loss=lossF(net(Xtr),ytr); loss.backward(); opt.step()
        net.eval();
        with torch.no_grad(): val=lossF(net(Xva),yva).item()
        if best-val>5e-5: best=val; stall=0; torch.save(net.state_dict(),a.model)
        else: stall+=1
        if ep==1 or ep%5==0: print(f'Epoch {ep:4d}  train {loss.item():.4f}  val {val:.4f}')
        if stall>=25: print('⏸️  early stop'); break
    print('best val',best)

# ─── shared cache for predict/design ────────────────────
def _cache(model):
    asa=_asa_map(model); d={}
    for ch in model:
        rl=list(ch)
        for i,r in enumerate(rl):
            prev=rl[i-1] if i else None; nxt=rl[i+1] if i<len(rl)-1 else None
            phi=(_phi(prev,r) or 0.0); psi=(_psi(r,nxt) or 0.0)
            d[(ch.name,r.seqid.num)]=(phi,psi,_bin(phi),_bin(psi),
                                      asa.get((ch.name,r.seqid.num),0.0), _bfac(r))
    return d

# ─── predict (unchanged from previous) ──────────────────
def predict(a):
    net=Net(); net.load_state_dict(torch.load(a.model,map_location='cpu')); net.eval()
    mdl=gemmi.read_structure(str(a.structure))[0]; geom=_cache(mdl)
    acids=[((c.name,r.seqid.num),r) for c in mdl for r in c if r.name in ACIDS]
    bases=[((c.name,r.seqid.num),r) for c in mdl for r in c if r.name in BASES]

    ns=None
    try:
        ns=gemmi.NeighborSearch(mdl,getattr(mdl,'cell',None) or mdl.get_unit_cell(),RADIUS).populate()
        ns_mode=True
    except Exception:
        try: ns=gemmi.NeighborSearch(gemmi.Structure(mdl),RADIUS).populate(); ns_mode=True
        except Exception: ns_mode=False

    rows=[]
    for kA,rA in acids:
        cbA=_CB(rA);  # skip degenerate
        if cbA is None: continue

        hits=(ns.find_atoms(cbA,radius=RADIUS) if ns_mode else
              [(_CB(rB),kB,rB) for kB,rB in bases if (cb:=_CB(rB)) and cbA.dist(cb)<=RADIUS])

        for h in hits:
            if ns_mode:
                atom=getattr(h,'atom',None)
                if atom is None and hasattr(h,'to_atom'):
                    try: atom=h.to_atom()
                    except Exception: atom=None
                if atom is None: continue
                rB=atom.get_parent_residue(); chainB=atom.get_parent_chain().name
            else:
                _,kB,rB=h; chainB=kB[0]
            kB=(chainB,rB.seqid.num)

            abd=min_charged_dist(rA,rB)
            if abd is None or abd>a.cutoff: continue
            cad=_dist(_CA(rA),_CA(rB)); cbd=_dist(_CB(rA),_CB(rB))
            if None in (cad,cbd): continue
            seq=abs(kA[1]-kB[1]); same=int(kA[0]==kB[0])
            (phi1,psi1,pb1,qb1,asa1,b1)=geom[kA]
            (phi2,psi2,pb2,qb2,asa2,b2)=geom[kB]
            feat=[np.log1p(abd),np.log1p(cad),np.log1p(cbd),
                  np.log1p(seq),same,orientation(rA,rB),
                  phi1/180,psi1/180,pb1,qb1,asa1,b1,
                  phi2/180,psi2/180,pb2,qb2,asa2,b2]
            with torch.no_grad():
                p=float(torch.sigmoid(net(torch.tensor(feat,dtype=torch.float32))))
            rows.append([kA[0],kA[1],kB[0],kB[1],p])

    rows.sort(key=lambda r:-r[4]); rows=rows if a.top_k<=0 else rows[:a.top_k]
    pd.DataFrame(rows,columns=['chain1','res1','chain2','res2','prob']).to_csv(a.out,index=False)
    print('saved →',a.out)

# ─── design helpers ─────────────────────────────────────
def _score_pair(net,geom,k1,k2,seq,same,abd,cad,cbd):
    (phi1,psi1,pb1,qb1,asa1,b1)=geom[k1]
    (phi2,psi2,pb2,qb2,asa2,b2)=geom[k2]
    feat=[np.log1p(abd),np.log1p(cad),np.log1p(cbd),
          np.log1p(seq),same,0.0,
          phi1/180,psi1/180,pb1,qb1,asa1,b1,
          phi2/180,psi2/180,pb2,qb2,asa2,b2]
    with torch.no_grad():
        return float(torch.sigmoid(net(torch.tensor(feat,dtype=torch.float32))))

def _design_generic(a, inter_only: bool):
    net=Net(); net.load_state_dict(torch.load(a.model,map_location='cpu')); net.eval()
    mdl=gemmi.read_structure(str(a.structure))[0]; geom=_cache(mdl)
    bb=[((c.name,r.seqid.num),r) for c in mdl for r in c]
    rows=[]
    for (k1,r1),(k2,r2) in it.combinations(bb,2):
        if inter_only and k1[0]==k2[0]: continue
        cb1,cb2=_CB(r1),_CB(r2)
        if cb1 is None or cb2 is None or cb1.dist(cb2)>RADIUS: continue
        seq=abs(k1[1]-k2[1]); same=int(k1[0]==k2[0])
        for mut1,mut2 in (('ASP','LYS'),('ASP','ARG'),('GLU','LYS'),('GLU','ARG')):
            ch1=fake_charge(cb1,_CA(r1),mut1); ch2=fake_charge(cb2,_CA(r2),mut2)
            if ch1 is None or ch2 is None: continue
            abd=ch1.dist(ch2)
            cad=_dist(_CA(r1),_CA(r2)); cbd=_dist(cb1,cb2)
            if None in (cad,cbd): continue
            prob=_score_pair(net,geom,k1,k2,seq,same,abd,cad,cbd)
            rows.append([k1[0],k1[1],mut1,k2[0],k2[1],mut2,prob])
    rows.sort(key=lambda r:-r[6]); rows=rows if a.top_k<=0 else rows[:a.top_k]
    pd.DataFrame(rows,columns=['chain1','res1','mut1','chain2','res2','mut2','prob']).to_csv(a.out,index=False)
    print('saved →',a.out)

def design(a):       _design_generic(a, inter_only=False)
def design_inter(a): _design_generic(a, inter_only=True)

# ─── CLI wiring ─────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(description='Salt-bridge toolkit')
    sp=ap.add_subparsers(dest='cmd',required=True)

    b=sp.add_parser('build');   b.add_argument('--data_dir',required=True); b.add_argument('--out_csv',required=True); b.add_argument('--nproc',type=int); b.set_defaults(func=build)
    t=sp.add_parser('train');   t.add_argument('--dataset',required=True); t.add_argument('--model',default='sb_model.pt'); t.add_argument('--epochs',type=int,default=1000); t.set_defaults(func=train)
    p=sp.add_parser('predict'); p.add_argument('--model',required=True); p.add_argument('--structure',required=True); p.add_argument('--cutoff',type=float,default=8); p.add_argument('--top_k',type=int,default=25); p.add_argument('--out',default='sb_preds.csv'); p.set_defaults(func=predict)
    d=sp.add_parser('design');  d.add_argument('--model',required=True); d.add_argument('--structure',required=True); d.add_argument('--top_k',type=int,default=50); d.add_argument('--out',default='sb_design.csv'); d.set_defaults(func=design)
    di=sp.add_parser('design_inter'); di.add_argument('--model',required=True); di.add_argument('--structure',required=True); di.add_argument('--top_k',type=int,default=50); di.add_argument('--out',default='sb_design_inter.csv'); di.set_defaults(func=design_inter)

    args=ap.parse_args(); args.func(args)

if __name__=='__main__':
    main()
