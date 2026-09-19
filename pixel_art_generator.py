#!/usr/bin/env python3
"""
Pixel Art AI Generator - MoE 64 experts + RAG + procedural
Autor: Valentin
"""
import os, sys, json, math, random, hashlib, argparse, gc
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

OUTPUT_DIR = Path("pixel_art_output")
MODEL_DIR = Path("models")
LAYERS_DIR = MODEL_DIR / "layers"
KNOWLEDGE_FILE = Path("knowledge_base.json")
MANIFEST_FILE = MODEL_DIR / "manifest.json"

PALETTE_NES = ["#7C7C7C","#0000FC","#0000BC","#4428BC","#940084","#A80020","#A81000","#881400","#503000","#007800","#006800","#005800","#004058","#000000","#BCBCBC","#0078F8","#0058F8","#6844FC","#D800CC","#E40058","#F83800","#E45C10","#AC7C00","#00B800","#00A800","#00A844","#008888","#F8F8F8","#3CBCFC","#6888FC","#9878F8","#F878F8","#F85898","#F87858","#FCA044","#F8B800","#B8F818","#58D854","#58F898","#00E8D8","#787878","#FCFCFC","#A4E4FC","#B8B8F8","#D8B8F8","#F8B8F8","#F8A4C0","#F0D0B0","#FCE0A8","#F8D878","#D8F878","#B8F8B8","#B8F8D8","#00FCFC"]
PALETTE_GAMEBOY = ["#0f380f","#306230","#8bac0f","#9bbc0e"]
PALETTE_CGA = ["#000000","#55FFFF","#FF55FF","#FFFFFF"]

BAYER_2x2 = np.array([[0,2],[3,1]],dtype=np.float32)/4.0
BAYER_4x4 = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]],dtype=np.float32)/16.0
BAYER_8x8 = np.array([[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],[12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],[3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],[15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]],dtype=np.float32)/64.0

EXPERT_NAMES = ["nes_8bit","snes_16bit","gameboy","gba","cga","atari","commodore64","amiga","modern_pixel","isometric","topdown","side_scroller","platformer","rpg","action","adventure","puzzle","shooter","fighting","racing","sports","strategy","simulation","horror_pixel","fantasy_pixel","scifi_pixel","medieval_pixel","steampunk_pixel","cyberpunk_pixel","post_apocalyptic","underwater_pixel","space_pixel","western_pixel","noir_pixel","cartoon_pixel","anime_pixel","realistic_pixel","abstract_pixel","minimal_pixel","detailed_pixel","monochrome","sepia_pixel","neon_pixel","pastel_pixel","dark_pixel","bright_pixel","character_sprite","enemy_sprite","item_sprite","tileset_terrain","tileset_dungeon","tileset_city","tileset_forest","background_sky","background_ground","ui_elements","icons","animations","portraits","logos","dithering_expert","outline_expert","cel_shading_expert","color_harmony_expert"]

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def make_palette(name="nes", n=16, seed=None):
    rng = np.random.RandomState(seed)
    if name == "nes": colors = PALETTE_NES
    elif name == "gameboy": colors = PALETTE_GAMEBOY
    elif name == "cga": colors = PALETTE_CGA
    elif name == "monochrome": colors = ["#000000","#555555","#AAAAAA","#FFFFFF"]
    elif name == "sepia": colors = ["#3B2414","#6B4423","#A67B5B","#D4A574","#E8C8A0","#F5E6D3","#F8F0E3","#FFFFFF"]
    elif name == "pastel": colors = ["#FFD1DC","#FFB7C5","#FDFD96","#B5EAD7","#C7CEEA","#E2F0CB","#FFDAC1","#B5D8EB","#F0E6FF","#FFE4E1","#DCD0FF","#C1E1C1","#FFEBCD","#E0FFFF","#F5DEB3","#FFFACD"]
    elif name == "neon": colors = ["#FF006E","#FF4DA6","#FF00FF","#CC00FF","#8B00FF","#00F0FF","#00FFFF","#39FF14","#CCFF00","#FFF200","#FF8C00","#FF1744","#D500F9","#651FFF","#00E5FF","#76FF03"]
    elif name == "dark": colors = ["#0A0A0A","#1A1A1A","#2C2C2C","#3D3D3D","#4F4F4F","#1A0000","#2A0000","#3A0000","#00001A","#00002A","#00003A","#1A001A","#2A002A","#3A003A","#0A1A0A","#1A2A1A"]
    elif name == "bright": colors = ["#FFFFFF","#FFFFAA","#AAFFFF","#FFAAFF","#AAFFAA","#FFAAAA","#AAAAFF","#FFCC00","#00CCFF","#FF00CC","#00FFCC","#CCFF00","#CC00FF","#00FF00","#FF0000","#0000FF"]
    else: colors = PALETTE_NES
    n = min(n, len(colors))
    selected = list(colors)
    if n < len(selected):
        rng.shuffle(selected)
        selected = selected[:n]
    return np.array([hex_to_rgb(c) for c in selected], dtype=np.uint8)

class PerlinNoise:
    def __init__(self, seed=None):
        rng = np.random.RandomState(seed)
        p = np.arange(256); rng.shuffle(p)
        self.perm = np.concatenate([p,p])
        self.grads = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]],dtype=np.float32)
    def fade(self,t): return t*t*t*(t*(t*6-15)+10)
    def lerp(self,a,b,t): return a+t*(b-a)
    def noise(self,x,y):
        X=int(math.floor(x))&255; Y=int(math.floor(y))&255
        xf=x-math.floor(x); yf=y-math.floor(y)
        u=self.fade(xf); v=self.fade(yf)
        aa=self.perm[self.perm[X]+Y]%8; ab=self.perm[self.perm[X]+Y+1]%8
        ba=self.perm[self.perm[X+1]+Y]%8; bb=self.perm[self.perm[X+1]+Y+1]%8
        g_aa=np.dot(self.grads[aa],[xf,yf]); g_ba=np.dot(self.grads[ba],[xf-1,yf])
        g_ab=np.dot(self.grads[ab],[xf,yf-1]); g_bb=np.dot(self.grads[bb],[xf-1,yf-1])
        return self.lerp(self.lerp(g_aa,g_ba,u), self.lerp(g_ab,g_bb,u), v)
    def fbm(self,x,y,octaves=4):
        total,freq,amp,maxv=0.0,1.0,1.0,0.0
        for _ in range(octaves):
            total+=self.noise(x*freq,y*freq)*amp; maxv+=amp
            amp*=0.5; freq*=2.0
        return total/maxv
    def generate(self,w,h,scale=8.0,octaves=4):
        out=np.zeros((h,w),dtype=np.float32)
        for y in range(h):
            for x in range(w):
                out[y,x]=self.fbm(x/scale,y/scale,octaves)
        return (out-out.min())/(out.max()-out.min()+1e-8)

def cellular_cave(w,h,fill=0.45,iter=5,seed=None):
    rng=np.random.RandomState(seed)
    grid=(rng.random((h,w))<fill).astype(np.uint8)
    grid[0,:]=grid[-1,:]=grid[:,0]=grid[:,-1]=1
    for _ in range(iter):
        new=grid.copy()
        for y in range(1,h-1):
            for x in range(1,w-1):
                new[y,x]=1 if grid[y-1:y+2,x-1:x+2].sum()>=5 else 0
        grid=new
    return grid

def quantize(img,pal):
    h,w,_=img.shape
    flat=img.reshape(-1,3).astype(np.float32)
    p=pal.astype(np.float32)
    dist=((flat[:,None,:]-p[None,:,:])**2).sum(axis=2)
    idx=dist.argmin(axis=1)
    return p[idx].astype(np.uint8).reshape(h,w,3), idx.reshape(h,w)

def dither_bayer(img,pal,matrix):
    h,w,_=img.shape; mh,mw=matrix.shape
    p=pal.astype(np.float32); out=np.zeros_like(img); idx_out=np.zeros((h,w),dtype=np.int32)
    imgf=img.astype(np.float32)
    for y in range(h):
        for x in range(w):
            pix=np.clip(imgf[y,x]+(matrix[y%mh,x%mw]-0.5)*64,0,255)
            d=((p-pix[None,:])**2).sum(axis=1)
            i=d.argmin(); out[y,x]=p[i].astype(np.uint8); idx_out[y,x]=i
    return out, idx_out

def dither_floyd(img,pal):
    h,w,_=img.shape; p=pal.astype(np.float32)
    img=img.astype(np.float32).copy(); out=np.zeros_like(img); idx_out=np.zeros((h,w),dtype=np.int32)
    for y in range(h):
        for x in range(w):
            old=img[y,x].copy()
            d=((p-old[None,:])**2).sum(axis=1); i=d.argmin(); new=p[i]
            out[y,x]=new.astype(np.uint8); idx_out[y,x]=i
            err=old-new
            if x+1<w: img[y,x+1]+=err*7/16
            if y+1<h:
                if x-1>=0: img[y+1,x-1]+=err*3/16
                img[y+1,x]+=err*5/16
                if x+1<w: img[y+1,x+1]+=err*1/16
    return np.clip(out,0,255).astype(np.uint8), idx_out

def dither_atkinson(img,pal):
    h,w,_=img.shape; p=pal.astype(np.float32)
    img=img.astype(np.float32).copy(); out=np.zeros_like(img); idx_out=np.zeros((h,w),dtype=np.int32)
    for y in range(h):
        for x in range(w):
            old=img[y,x].copy()
            d=((p-old[None,:])**2).sum(axis=1); i=d.argmin(); new=p[i]
            out[y,x]=new.astype(np.uint8); idx_out[y,x]=i
            err=(old-new)/8
            if x+1<w: img[y,x+1]+=err
            if x+2<w: img[y,x+2]+=err
            if y+1<h:
                if x-1>=0: img[y+1,x-1]+=err
                img[y+1,x]+=err
                if x+1<w: img[y+1,x+1]+=err
            if y+2<h: img[y+2,x]+=err
    return np.clip(out,0,255).astype(np.uint8), idx_out

def apply_dithering(img,pal,method="bayer4x4"):
    if method=="bayer2x2": return dither_bayer(img,pal,BAYER_2x2)
    if method=="bayer4x4": return dither_bayer(img,pal,BAYER_4x4)
    if method=="bayer8x8": return dither_bayer(img,pal,BAYER_8x8)
    if method=="floyd_steinberg": return dither_floyd(img,pal)
    if method=="atkinson": return dither_atkinson(img,pal)
    return quantize(img,pal)

def add_outline(img,color="dark"):
    h,w,_=img.shape; out=img.copy()
    ol=np.array([0,0,0] if color=="dark" else [255,255,255],dtype=np.uint8)
    bg=np.array([255,255,255],dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if np.array_equal(img[y,x],bg): continue
            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny,nx=y+dy,x+dx
                if 0<=ny<h and 0<=nx<w and np.array_equal(img[ny,nx],bg):
                    out[y,x]=ol; break
    return out

def resize_nearest(img,scale=4):
    pil=Image.fromarray(img)
    return np.array(pil.resize((pil.width*scale,pil.height*scale),Image.NEAREST))

def gen_symmetric_sprite(w,h,seed,pal,density=0.45):
    rng=np.random.RandomState(seed)
    img=np.full((h,w,3),255,dtype=np.uint8)
    half_w=(w+1)//2
    main_color=pal[rng.randint(0,len(pal))]
    second_color=pal[rng.randint(0,len(pal))]
    mask=np.zeros((h,half_w),dtype=bool)
    cy=rng.randint(h//4,3*h//4); cx=rng.randint(0,half_w)
    for _ in range(int(w*h*density)):
        if 0<=cy<h and 0<=cx<half_w: mask[cy,cx]=True
        cy+=rng.randint(-1,2); cx+=rng.randint(-1,2)
        cy=max(0,min(h-1,cy)); cx=max(0,min(half_w-1,cx))
    for y in range(h):
        for x in range(half_w):
            if mask[max(0,y-1):y+2,max(0,x-1):x+2].sum()>=3: mask[y,x]=True
    for y in range(h):
        for x in range(half_w):
            if mask[y,x]:
                img[y,x]=main_color
                mx=w-1-x
                if mx>=0: img[y,mx]=main_color
    for y in range(h):
        for x in range(half_w):
            if mask[y,x] and rng.random()<0.15:
                img[y,x]=second_color
                mx=w-1-x
                if mx>=0: img[y,mx]=second_color
    return img

def gen_terrain(w,h,seed,biome="grass"):
    perlin=PerlinNoise(seed)
    noise=perlin.generate(w,h,scale=max(4,w/4),octaves=4)
    img=np.zeros((h,w,3),dtype=np.uint8)
    if biome=="desert":
        img[...,0]=(200+noise*55).astype(np.uint8); img[...,1]=(170+noise*40).astype(np.uint8); img[...,2]=(100+noise*30).astype(np.uint8)
    elif biome=="water":
        img[...,0]=(30+noise*50).astype(np.uint8); img[...,1]=(80+noise*80).astype(np.uint8); img[...,2]=(150+noise*100).astype(np.uint8)
    elif biome=="mountain":
        img[...,0]=(80+noise*100).astype(np.uint8); img[...,1]=(80+noise*100).astype(np.uint8); img[...,2]=(85+noise*100).astype(np.uint8)
        img[noise>0.75]=[240,240,250]
    else:
        img[...,0]=(30+noise*60).astype(np.uint8); img[...,1]=(100+noise*120).astype(np.uint8); img[...,2]=(30+noise*50).astype(np.uint8)
    return img

def gen_space(w,h,seed):
    rng=np.random.RandomState(seed)
    img=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        img[y,:]=[int(10+y/h*20),0,int(30+y/h*40)]
    for _ in range(w*h//8):
        x,y=rng.randint(0,w),rng.randint(0,h); b=rng.randint(180,256)
        img[y,x]=[b,b,b]
    perlin=PerlinNoise(seed+1000)
    neb=perlin.generate(w,h,scale=max(4,w/3),octaves=3)
    for y in range(h):
        for x in range(w):
            v=neb[y,x]
            if v>0.55:
                r,g,b=img[y,x]
                img[y,x]=[min(255,int(r+v*100)),min(255,int(g+v*30)),min(255,int(b+v*80))]
    return img

def gen_city(w,h,seed):
    rng=np.random.RandomState(seed)
    img=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        t=y/h; img[y,:]=[int(20+t*30),int(10+t*20),int(60+t*40)]
    ground=int(h*0.75); x=0
    while x<w:
        bw=rng.randint(max(3,w//10),max(4,w//5)); bh=rng.randint(h//4,int(h*0.7)); by=ground-bh
        c=rng.randint(40,100); img[by:ground,x:min(w,x+bw)]=[c,c,c+20]
        for wy in range(by+2,ground-2,4):
            for wx in range(x+1,min(w-1,x+bw-1),3):
                if rng.random()>0.4: img[wy:wy+2,wx:wx+1]=[255,220,100]
        x+=bw+rng.randint(0,2)
    img[ground:h,:]=[30,30,40]
    return img

def gen_dungeon(w,h,seed):
    cave=cellular_cave(w,h,fill=0.45,iter=5,seed=seed)
    img=np.zeros((h,w,3),dtype=np.uint8)
    img[cave==1]=[40,30,25]; img[cave==0]=[80,70,60]
    rng=np.random.RandomState(seed)
    for _ in range(max(1,w*h//400)):
        tx,ty=rng.randint(1,w-1),rng.randint(1,h-1)
        if cave[ty,tx]==0: img[ty,tx]=[255,150,0]
    return img

def gen_tileset(w,h,seed,biome="grass"):
    rng=np.random.RandomState(seed)
    img=np.zeros((h,w,3),dtype=np.uint8)
    ts=16 if w>=64 else 8
    for ty in range(h//ts):
        for tx in range(w//ts):
            sub=gen_terrain(ts,ts,seed+ty*100+tx,biome)
            img[ty*ts:ty*ts+ts,tx*ts:tx*ts+ts]=sub
    return img

def generate_base(prompt,cfg,w,h,seed,pal):
    subject=cfg.get("subject"); scene=cfg.get("scene"); cat=cfg.get("category")
    if scene in ("cave","dungeon"): return gen_dungeon(w,h,seed)
    if scene=="space": return gen_space(w,h,seed)
    if scene=="city": return gen_city(w,h,seed)
    if scene in ("desert","beach"): return gen_terrain(w,h,seed,"desert")
    if scene in ("water","sea","underwater"): return gen_terrain(w,h,seed,"water")
    if scene=="mountain": return gen_terrain(w,h,seed,"mountain")
    if scene in ("forest","grass","terrain"): return gen_terrain(w,h,seed,"grass")
    if cat=="tileset": return gen_tileset(w,h,seed,scene or "grass")
    return gen_symmetric_sprite(w,h,seed,pal)

KEYWORDS = {
    "nes":{"style":"nes_8bit","palette":"nes","palette_size":16,"dithering":"bayer2x2"},
    "8-bit":{"style":"nes_8bit","palette":"nes","palette_size":16},
    "8bit":{"style":"nes_8bit","palette":"nes","palette_size":16},
    "snes":{"style":"snes_16bit","palette":"nes","palette_size":32},
    "16-bit":{"style":"snes_16bit","palette":"nes","palette_size":32},
    "gameboy":{"style":"gameboy","palette":"gameboy","palette_size":4},
    "game boy":{"style":"gameboy","palette":"gameboy","palette_size":4},
    "cga":{"style":"cga","palette":"cga","palette_size":4},
    "retrô":{"style":"nes_8bit"},"retro":{"style":"nes_8bit"},
    "isométrico":{"style":"isometric"},"isometric":{"style":"isometric"},
    "top-down":{"style":"topdown"},"cyberpunk":{"style":"cyberpunk_pixel","palette":"neon"},
    "steampunk":{"style":"steampunk_pixel","palette":"sepia"},
    "medieval":{"style":"medieval_pixel"},"anime":{"style":"anime_pixel"},
    "horror":{"style":"horror_pixel","palette":"dark"},
    "fantasia":{"style":"fantasy_pixel"},"espacial":{"style":"space_pixel"},
    "neon":{"palette":"neon","style":"neon_pixel"},"pastel":{"palette":"pastel"},
    "monocromático":{"palette":"monochrome","palette_size":4},
    "monocromatico":{"palette":"monochrome","palette_size":4},
    "escuro":{"palette":"dark"},"claro":{"palette":"bright"},
    "personagem":{"subject":"character"},"herói":{"subject":"hero"},"heroi":{"subject":"hero"},
    "vilão":{"subject":"villain"},"monstro":{"subject":"monster"},"inimigo":{"subject":"enemy"},
    "dragão":{"subject":"dragon"},"dragao":{"subject":"dragon"},
    "slime":{"subject":"slime"},"goblin":{"subject":"goblin"},"esqueleto":{"subject":"skeleton"},
    "gato":{"subject":"cat"},"robô":{"subject":"robot"},"robo":{"subject":"robot"},
    "espada":{"subject":"sword"},"escudo":{"subject":"shield"},"poção":{"subject":"potion"},
    "chave":{"subject":"key"},"moeda":{"subject":"coin"},"baú":{"subject":"chest"},
    "floresta":{"scene":"forest"},"dungeon":{"scene":"dungeon"},"caverna":{"scene":"cave"},
    "cidade":{"scene":"city"},"espaço":{"scene":"space"},"espaco":{"scene":"space"},
    "deserto":{"scene":"desert"},"montanha":{"scene":"mountain"},
    "subaquático":{"scene":"underwater"},"mar":{"scene":"water"},
    "tileset":{"category":"tileset"},"terreno":{"scene":"terrain"},"grama":{"scene":"grass"},
    "água":{"scene":"water"},"agua":{"scene":"water"},"noite":{"palette":"dark"},
}

def interpret_prompt(prompt):
    cfg={"style":"modern_pixel","palette":"nes","palette_size":16,"dithering":"bayer4x4",
         "outline":"none","subject":None,"scene":None,"category":None,"tags":[]}
    pl=prompt.lower()
    for key in sorted(KEYWORDS.keys(),key=len,reverse=True):
        if key in pl:
            for k,v in KEYWORDS[key].items():
                if v is not None: cfg[k]=v
            cfg["tags"].append(key)
    return cfg

class RAG:
    def __init__(self,path=KNOWLEDGE_FILE):
        self.entries=[]
        if path.exists():
            try:
                with open(path,"r",encoding="utf-8") as f:
                    self.entries=json.load(f).get("entries",[])
            except Exception: pass
    def retrieve(self,query,top_k=3):
        if not self.entries: return []
        qt=set(query.lower().replace(","," ").replace("-"," ").split())
        scored=[]
        for e in self.entries:
            s=0
            tags=set(t.lower() for t in e.get("tags",[]))
            s+=len(qt&tags)*3
            if e.get("style","").lower() in query.lower(): s+=5
            if s>0: scored.append((s,e))
        scored.sort(key=lambda x:-x[0])
        return [e for _,e in scored[:top_k]]

class MoE:
    def __init__(self,num_experts=64,hidden=512,blocks=12):
        self.num_experts=num_experts; self.hidden=hidden; self.blocks=blocks
        self.current=None; self.current_idx=-1
    def load_layer(self,idx):
        if idx==self.current_idx and self.current is not None: return self.current
        self.current=None; self.current_idx=-1; gc.collect()
        path=LAYERS_DIR/f"layer_{idx:03d}.npz"
        if path.exists():
            data=np.load(path); layer={}
            for k in data.files:
                arr=data[k]
                if arr.dtype==np.int8 and f"{k}_scale" in data.files:
                    layer[k]=arr.astype(np.float32)*float(data[f"{k}_scale"])
                else: layer[k]=arr.astype(np.float32)
            self.current=layer
        else:
            seed=int(hashlib.md5(f"layer_{idx}".encode()).hexdigest()[:8],16)%(2**31)
            rng=np.random.RandomState(seed)
            self.current={"router":rng.randn(self.hidden,self.num_experts).astype(np.float32)*0.1}
        self.current_idx=idx
        return self.current
    def route(self,features):
        layer=self.load_layer(0)
        router=layer.get("router")
        if router is None: return [0,1],[0.5,0.5]
        x=features[:self.hidden] if len(features)>=self.hidden else np.pad(features,(0,self.hidden-len(features)))
        logits=np.dot(x,router)
        e=np.exp(logits-logits.max()); probs=e/e.sum()
        top2=np.argsort(probs)[-2:][::-1]
        w=probs[top2]; w=w/w.sum()
        return top2.tolist(),w.tolist()

def get_next_number():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    nums=[int(f.stem) for f in OUTPUT_DIR.glob("*.png") if f.stem.isdigit()]
    return max(nums)+1 if nums else 1

def save(img,meta,scale=4):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    num=get_next_number()
    Image.fromarray(img).save(OUTPUT_DIR/f"{num}.png")
    if scale>1:
        Image.fromarray(resize_nearest(img,scale)).save(OUTPUT_DIR/f"{num}_x{scale}.png")
    with open(OUTPUT_DIR/f"{num}.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2,ensure_ascii=False)
    meta["number"]=num; meta["path"]=str(OUTPUT_DIR/f"{num}.png")
    return meta

def generate_pixel_art(prompt,width=64,height=64,style=None,palette_size=16,
                       dithering="bayer4x4",outline="none",scale=4,seed=None):
    print(f"\n🎨 Gerando: '{prompt}'")
    cfg=interpret_prompt(prompt)
    if style: cfg["style"]=style
    cfg["palette_size"]=palette_size; cfg["dithering"]=dithering; cfg["outline"]=outline

    rag=RAG(); matches=rag.retrieve(prompt,top_k=3)
    if matches:
        top=matches[0]
        if not style and top.get("style"): cfg["style"]=top["style"]
        if top.get("palette"): cfg["palette"]=top["palette"]
        print(f"   📚 RAG: {[m.get('style') for m in matches]}")

    if seed is None: seed=int.from_bytes(os.urandom(4),"little")^hash(prompt)
    rng=np.random.RandomState(seed)

    moe=MoE()
    feat=np.zeros(512,dtype=np.float32)
    feat[0]=(int(hashlib.md5(prompt.encode()).hexdigest()[:8],16)&0xFF)/255.0
    feat[1]=width/256.0; feat[2]=height/256.0; feat[3]=palette_size/256.0
    for i,c in enumerate(prompt[:100]): feat[10+i]=ord(c)/255.0
    experts,weights=moe.route(feat)
    expert_names=[EXPERT_NAMES[i] if i<len(EXPERT_NAMES) else f"expert_{i}" for i in experts]
    print(f"   🧠 MoE: {expert_names}")

    pal=make_palette(cfg["palette"],cfg["palette_size"],seed)
    base=generate_base(prompt,cfg,width,height,seed,pal)
    dithered,_=apply_dithering(base,pal,cfg["dithering"])
    if cfg["outline"] not in (None,"none"):
        dithered=add_outline(dithered,cfg["outline"])

    meta={"prompt":prompt,"config":{k:v for k,v in cfg.items() if k!="tags"},
          "tags":cfg["tags"],"rag_matches":[m.get("style") for m in matches],
          "moe_experts":expert_names,"moe_weights":weights,"seed":seed,
          "palette_name":cfg["palette"],"palette_size":int(cfg["palette_size"]),
          "dithering":cfg["dithering"],"outline":cfg["outline"],
          "timestamp":datetime.now().isoformat(),"resolution":[width,height],"scale":scale}
    saved=save(dithered,meta,scale)
    print(f"   ✅ Salvo: {saved['path']}")
    return saved

def batch_generate(prompt,count=1,**kw):
    results=[]
    for i in range(count):
        s=int.from_bytes(os.urandom(4),"little")^hash(prompt+str(i))
        results.append(generate_pixel_art(prompt,seed=s,**kw))
    return results

def main():
    p=argparse.ArgumentParser(description="🎨 Gerador de Pixel Art")
    p.add_argument("--batch",action="store_true")
    p.add_argument("--prompt",type=str)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--height",type=int,default=64)
    p.add_argument("--style",type=str,default=None)
    p.add_argument("--palette-size",type=int,default=16)
    p.add_argument("--dithering",type=str,default="bayer4x4")
    p.add_argument("--outline",type=str,default="none")
    p.add_argument("--scale",type=int,default=4)
    p.add_argument("--count",type=int,default=1)
    p.add_argument("--seed",type=int,default=None)
    args=p.parse_args()
    print("="*60); print("🎨 PIXEL ART AI GENERATOR"); print("="*60)
    if args.batch:
        if not args.prompt: print("❌ --prompt obrigatório"); sys.exit(1)
        batch_generate(args.prompt,count=args.count,width=args.width,height=args.height,
                       style=args.style,palette_size=args.palette_size,dithering=args.dithering,
                       outline=args.outline,scale=args.scale,seed=args.seed)
        print(f"\n✅ {args.count} pixel arts geradas em {OUTPUT_DIR}/")
        return
    while True:
        print("\n1.Gerar  2.Aleatório  3.Batch(5)  4.Listar  5.Sair")
        op=input("Opção: ").strip()
        if op=="1":
            pr=input("Prompt: ").strip()
            if pr: generate_pixel_art(pr)
        elif op=="2":
            generate_pixel_art(random.choice(["herói estilo NES","dragão 16-bit","floresta Game Boy","nave cyberpunk","dungeon escura"]))
        elif op=="3":
            pr=input("Prompt: ").strip()
            if pr: batch_generate(pr,count=5)
        elif op=="4":
            if OUTPUT_DIR.exists():
                for f in sorted(OUTPUT_DIR.glob("*.png"))[-10:]: print(f"  - {f.name}")
        elif op=="5": break

if __name__=="__main__": main()