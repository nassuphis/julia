import sys
sys.path.insert(0, "/Users/nicknassuphis/specparser")
import math
import numpy as np
from numba import njit, prange, types, complex128, int32, float64
import argparse
from specparser import specparser
from specparser import expandspec
import galaxy_raster
import json
import cv2

@njit
def f(seed):
    np.random.seed(seed)       # sets Numba's RNG
    out = np.empty(5)
    for i in range(5):
        out[i] = np.random.rand()
    return out

f(500)

@njit("complex128[:](int64,float64,complex128)",fastmath=True, cache=True)
def _points(N,w:float=1,center:complex=0+0j):
    re = -w + 2 * w * np.random.rand(N)
    im = -w + 2 * w * np.random.rand(N)
    return re + 1j*im + center


@njit("complex128(complex128, complex128, int32)", fastmath=True, cache=True)
def julia_equation(z: np.complex128, c:np.complex128, eqn:np.int32):
    z=z
    if eqn==0:
        return z*z*z*z*z*z - z*z*z*z + c
    elif eqn==1:
        return np.exp(1j*2*np.pi*np.abs(z)) + z + c
    elif eqn==2:
        return z*z + c
    elif eqn==3:
        return z*z*z + c
    elif eqn==4:
        return z*z*z*z + c
    elif eqn==5:
        return np.sin(z)*np.abs(z) + z*z + c
    elif eqn==6:
        return (z+1)/(z-1)+c
    elif eqn==7:
        return z*z + z/c
    elif eqn==8:
        return z*z + c/z
    elif eqn==9:
        return z*z + c/np.conj(z)
    elif eqn==10:
        return ( z*z -z + c )/( 3*z*z*z - z*z)
    elif eqn==11:
        return ( z*z + c )/( 3*z - np.exp(z) )
    elif eqn==12:
        return ( z*z*z + c )/( z*z*z*z*z + (2.5+1j)*z*z*z*z + (1.5-1j)*z*z*z + (-0.5+4j)*z*z + z - 1.0 + 3j )
    elif eqn==13:
        den1 = ( z*z*z*z*z + (2.5*c+1j)*z*z*z*z + (1.5-c*1j)*z*z*z + (-0.5+4j)*z*z + z - c + 3j )
        den2 = ( z*z*z*z*z + c*(-0.5+18j)*z*z*z*z + (12-1j)*z*z*z + (-0.5+4j)*z*z + z - c*c*c + 30j )
        num1 = ( z*z*z*z*z + (+15.5+18j)*z*z*z*z + (3-100j)*z*z*z + (-3.5+4j)*z*z + 69*z - 4.0 + 30j )
        num2 = ( z*z*z*z*z + c*c*c*z*z*z*z + c*z*z*z + z*z + z + c )
        return num1*num2/(den1*den2)
    elif eqn==14:
        num = ( z*z*z + c )
        den = ( z*z*z*z*z + (2.5+1j)*z*z*z*z + (1.5-1j)*z*z*z + (-0.5+4j)*z*z + z - 1.0 + 3j )
        for i in range(5):
            num *= ( z + 5*c )*( z - 3j*c)
        for i in range(15):
            den *= ( z*z*z - 5*c )
        return num/den
    elif eqn==15:
        num = ( z*z*z + c*z )
        den = ( z*z*z*z*z + (2.5+1j)*z*z*z*z + (1.5-1j)*z*z*z + (-0.5+4j)*z*z + z - 1.0 + 3j )
        for i in range(20):
            num *= ( c*z + 5*c -1j)*( z*z - 3j*c)
        for i in range(20):
            den *= ( z*z*z - 5*c ) 
        return num/den
    elif eqn==16:
        num = ( z*z*z + c*z )
        den = ( z*z*z*z*z  - 1.0 * c + 3j )
        for i in range(20):
            num *= ( c*z + 5*c -1j)*( z*z - 3j*c) + c * (z.real % 1)
        for i in range(20):
            den *= ( z*z*z - 5*c ) + (z.imag % 1)
        return num/den
    elif eqn==17:
        num = ( z*z*z + c )
        den = ( z*z*z*z*z  - 1.0 * c + 3j )
        for i in range(20):
            num += ( z*z - 3j*c) * ( (c*z).real % 1)
        for i in range(20):
            den += ( z*z*z + 5*c ) * ( (c*z).imag % 1)
        return num/den
    elif eqn==18:
        num = ( z*z*z + c ) * np.sin(z)
        den = ( z*z*z*z*z  - 1.0 * c + 3j ) * np.cos(z)
        for i in range(20):
            num *= ( z*z - 3j*c) 
        for i in range(20):
            den *= ( z*z*z + np.abs(z)*c )
        return num/den
    elif eqn==19:
        z = ( z*z*z + c*z )     
    elif eqn==20:
        z =  z*z + c*np.exp(1j*2*np.pi*z)        
    elif eqn==21:
        z =  z*z + c*z*np.exp(1j*2*np.pi*np.abs(z)) 
    elif eqn==22:
        znew=0
        for i in range(10):
            znew += z*z + c*z*np.exp(1j*2*np.pi*np.abs(z))    
        z=znew/10
    return z
    

@njit("int32(complex128, complex128, int32, int32, float64)", fastmath=True, cache=True)
def _julia_escape_single(
    z0: np.complex128,
    c: np.complex128,
    eqn: int = 0,
    max_iter: int = 400,
    bailout2: float = 4.0,
) -> np.int32:
    z = z0
    for k in range(max_iter):
        z = julia_equation(z,c,eqn)
        if (z.real*z.real + z.imag*z.imag) > bailout2: return k
    return max_iter

# vectorized, parallel caller
@njit("int32[:](complex128[:], complex128, int32, int32, float64)",
      parallel=True, fastmath=True, cache=True)
def julia_escape_vec(z0, c, eqn, max_iter, bailout2):
    n = z0.size
    out = np.empty(n, np.int32)
    for i in prange(n):
        out[i] = _julia_escape_single(z0[i], c, eqn, max_iter, bailout2)
    return out


@njit("complex128[:](int64, complex128, float64, complex128, int32, int32)", fastmath=True, cache=True)
def julia_sample(N, c, w, center, thresh, eqn):
    z0 = _points(N, w,center)
    iters = julia_escape_vec(z0, c, eqn, thresh+1, 4.0)
    keep = 0
    for i in range(N):
        if iters[i] > thresh:
            keep += 1
    out = np.empty(keep, np.complex128)
    j = 0
    for i in range(N):
        if iters[i] > thresh:
            out[j] = z0[i]
            j += 1
    return out

#=========================================
# C sampler
#=========================================

@njit("complex128[:](int64, float64, float64, float64, int32, int32)", fastmath=True, cache=True)
def c_sampler(N:int, lo:np.int64, hi:np.float64, w:np.float64, eqn:np.int32, max_iter:np.int32 = 400)-> np.ndarray:
    cs = _points(N,w,0)
    zs = _points(1000,w,0)
    pct_max = np.full(cs.size,0,dtype=np.float64)
    for i, c in enumerate(cs):
        escape_iter = julia_escape_vec(zs,c,eqn,max_iter,4.0)
        pct_max[i] = np.sum(escape_iter==max_iter)/zs.size
    passed = (pct_max>lo) & (pct_max<hi)
    return cs[passed]


#=========================================
#
#=========================================
def julia(N: int, c: np.complex128= -0.8 + 0.156j,maxi:int=200,w:float=1.5,eqn:int=1) -> np.ndarray:
    N = int(N)
    out  = np.zeros(N, np.complex128)
    kept = 0
    # 1) probe sample
    boost = 10
    s = julia_sample(
        np.int64(boost*N), 
        np.complex128(c), 
        np.float64(w), 
        np.complex128(0+0j), 
        np.int32(maxi),
        np.int32(eqn)
    )
    take = min(s.size, N)
    if take:
        out[:take] = s[:take]
        kept += take
    else:
        return out
    p = max(0.01, (s.size / (boost*N)))
    draw = 2*int(math.ceil(N / p))
    if draw < 1: draw = 1
    center = np.mean(out[:kept])
    w = 0.5 * max(
        np.ptp(out[:kept].real),
        np.ptp(out[:kept].imag)
    ) * 1.5
    rounds = 0
    while kept < N:
        rounds += 1
        if rounds > 10 : break
        need = (N - kept)
        s = julia_sample(
            np.int64(draw), 
            np.complex128(c), 
            np.float64(w),
            np.complex128(center), 
            np.int32(300),
            np.int32(eqn)
        )
        take = min(s.size, need)
        if take:
            out[kept:kept + take] = s[:take]
            kept += take
        center = np.mean(out[:kept])
        w = 0.5 * max(
            np.ptp(out[:kept].real),
            np.ptp(out[:kept].imag)
        ) * 1.15
    return out[:min(kept,N)]  # exactly N filled unless the defensive break triggered

def dict2julia(d):

    n    = int(d["n"][0].real)     if "n"    in d else 1_000
    c    = d["c"][0]               if "c"    in d else (-0.8 + 0.156j)
    maxi = int(d["maxi"][0].real)  if "maxi" in d else 200
    w    = d["w"][0].real          if "w"    in d else 1.5
    eqn  = int(d["eqn"][0].real)   if "eqn"  in d else 1
    
    z=julia(n,c,maxi,w,eqn)
    return z


def jsample(
        N, 
        max_rounds,
        eqn, 
        bottom: float = 0.05, 
        top: float = 0.05, 
        w:float=1.0, 
        iter: int = 400
    ):
    N = int(complex(N).real)
    eqn = int(complex(eqn).real)
    bottom = complex(bottom).real
    top = complex(top).real
    w = complex(w).real
    iter = int(complex(iter).real)
    batch = 500
    out = []
    rounds = 0
    wasted = 0
    while len(out) < N and rounds < max_rounds:
        print(f"jsample round {rounds} sampling: {batch}: have {len(out)}, width {w} bottom {bottom} top {top}")
        rounds += 1
        s = np.asarray(c_sampler(batch, bottom, top, w, np.int32(eqn), iter))
        if s.size == 0: 
            wasted += 1
            if wasted>0.1*max_rounds:
                w=w*2
                batch=batch*1.25
                wasted=0
                max_rounds+=5
                bottom = 0.0 * 0.25 + bottom * (1-0.25)
                top = top * 0.75 + 1.0 * (1-0.75)
            continue
        need = N - len(out)
        keep = min(need, s.size)
        for z in s[:keep]: out.append(complex(z))
    print(f"jsample round {rounds}: have {len(out)}")
    return [f"{z.real:+.5f}{z.imag:+.5f}j" for z in out]

def julia0(N):
    samples = c_sampler(10000,0.01,0.05,1.0,0,200)
    samples = samples[:min(samples.size,N)]
    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in samples ]
    return waypoints

def julia1(N):
    samples = c_sampler(10000,0.01,0.05,1.0,1,200)
    samples = samples[:min(samples.size,N)]
    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in samples ]
    return waypoints

def julia2(N):
    samples = c_sampler(10000,0.01,0.05,1.0,2,200)
    samples = samples[:min(samples.size,N)]
    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in samples ]
    return waypoints

def julia3(N):
    samples = c_sampler(10000,0.01,0.05,1.0,3,200)
    samples = samples[:min(samples.size,N)]
    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in samples ]
    return waypoints

def julia4(N):
    samples = c_sampler(10000,0.01,0.05,1.0,4,200)
    samples = samples[:min(samples.size,N)]
    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in samples ]
    return waypoints

def julia5(N):
    samples = c_sampler(10000,0.01,0.05,1.0,5,200)
    samples = samples[:min(samples.size,N)]
    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in samples ]
    return waypoints

def julia6(N):
    samples = c_sampler(10000,0.01,0.05,1.0,6,200)
    samples = samples[:min(samples.size,N)]
    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in samples ]
    return waypoints

#=========================================
# "interesting" image classification
#=========================================

if __name__ == "__main__":
    p = argparse.ArgumentParser("galaxy-cli", description="Julia set renderer")
    p.add_argument("--spec", required=True,help="Julia set specification (can include expandspec braces)")
    p.add_argument("--show-specs", action="store_true", help="Show expanded specs")
    p.add_argument("--pix", type=int, default=5000, help="Tile width/height in pixels (default 25000)")
    p.add_argument("--out", type=str, default="julia.png", help="Output PNG path")
    p.add_argument("--cols", type=int, default=None, help="Columns if chain expands to multiple tiles")
    p.add_argument("--rows", type=int, default=None, help="Rows if chain expands to multiple tiles")
    p.add_argument("--invert", action="store_true", help="Invert black/white")
    p.add_argument("--margin", type=float, default=0.0, help="Logical margin fraction around geometry")
    p.add_argument("--thumb",type=int, default=None,  help="Save thumbnail")
    p.add_argument("--clip",action="store_true", help="Clip julia samples")
    p.add_argument("--const", action="append", default=[],help="Add/override NAME=VALUE."
    )
    args = p.parse_args()

    for kv in args.const:
        print(f"const {kv}")
        k, v = specparser._parse_const_kv(kv)
        specparser.set_const(k, v)
        expandspec.set_const(k, v)

    expandspec.FUNCS["jsample"]=jsample
    expandspec.FUNCS["julia0"]=julia0
    expandspec.FUNCS["julia1"]=julia1
    expandspec.FUNCS["julia2"]=julia2
    expandspec.FUNCS["julia3"]=julia3
    expandspec.FUNCS["julia4"]=julia4
    expandspec.FUNCS["julia5"]=julia5
    expandspec.FUNCS["julia6"]=julia6

    specs = expandspec.expand_cartesian_lists(args.spec)
   
    if args.show_specs: 
        for spec in specs:
            print(f"{spec}")

    dicts = []
    for spec in specs:
        names, A = specparser.parse_names_and_args(spec, MAXA=12)
        d = dict(zip(names, A))
        d["spec"] = spec
        dicts.append(d)
    
    
    canvases = []
    titles = []
    for i,d in enumerate(dicts,start=1):
        print(f"{i}/{len(dicts)} Rendering {d['spec']}")
        z = dict2julia(d)
        if args.clip:
            z = z-np.mean(z)
            minx = np.min(z.real)
            maxx = np.max(z.real)
            miny = np.min(z.imag)
            maxy = np.max(z.imag)
            ptpx = maxx - minx
            ptpy = maxy - miny
            z = z[z.real<(maxx-0.1*ptpx)]
            z = z[z.imag<(maxy-0.1*ptpy)]
            z = z[z.real>(minx+0.1*ptpx)]
            z = z[z.imag>(miny+0.1*ptpy)]
            z = z-np.mean(z)
        canvas = galaxy_raster.render_to_canvas(z, args.pix, args.margin)
        canvases.append(canvas)
        titles.append(f"{d['spec']} | eqn:{int(d['eqn'][0].real)} | w:{round(d['w'][0].real,2)} | maxi:{int(d['maxi'][0].real)}")
    

    n = len(canvases)

    if args.cols:
        cols = args.cols
    elif args.rows:
        cols = int(round(n / args.rows))
    else: 
        cols = max(1, int(round(math.sqrt(n))))

    galaxy_raster.save_mosaic_png_bilevel(
        tiles= canvases, 
        titles=titles,
        cols=cols, 
        gap=20,
        out_path=args.out, 
        invert=args.invert,
        thumbnail=args.thumb,
    )

    print(f"saved: {args.out}")


