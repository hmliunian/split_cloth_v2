"""Joint feature extraction and interval alignment from the frozen experiment."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from dtw import dtw

N_PROGRESS = 32

def norm(x, floor):
    low, high = np.quantile(x, [.05,.95], axis=0)
    return (x-low)/np.maximum(high-low, floor)

def dense_features(q, t, representation):
    if representation == 'joint':
        pos = np.unwrap(q[:,[0,1,2,3,4,5,7,8,9,10,11,12]], axis=0)
        blocks = [np.sin(pos), np.cos(pos), norm(pos,.2)]
    else:
        xyz = q[:,[0,1,2,7,8,9]]
        angles = np.unwrap(q[:,[3,4,5,10,11,12]],axis=0)
        pos = np.column_stack([norm(xyz,.03),angles])
        blocks = [norm(xyz,.03),np.sin(angles),np.cos(angles)]
    grip = np.clip(norm(q[:,[6,13]],1e-4),0,1)
    # Fixed smoothing on the original 20 FPS axis, before derivatives.
    velocity = np.gradient(uniform_filter1d(pos,size=5,axis=0,mode='nearest'),t,axis=0)
    blocks.extend([grip,velocity,np.gradient(grip,t,axis=0)])
    for seconds in [.5,1.0]:
        for sign in [-1,1]:
            target = np.clip(t+sign*seconds,t[0],t[-1])
            shifted = np.column_stack([np.interp(target,t,pos[:,k]) for k in range(pos.shape[1])])
            blocks.append(sign*(shifted-pos))
    return np.column_stack(blocks), [b.shape[1] for b in blocks]

class Episode:
    def __init__(self, meta, arrays, representation='joint'):
        self.meta = meta; self.key = meta['key']; self.t = arrays['timestamps']
        self.n = len(self.t); self.end_t = self.t[-1]+np.median(np.diff(self.t))
        self.frames = np.array([c['source_frame_id'] for c in meta['candidates']], dtype=int)
        self.v = np.r_[-1,self.frames,self.n-1]; self.m = len(self.frames)
        assert np.all(np.diff(self.v)>0) and self.v[-2]<self.n-1
        q = arrays[representation]
        self.phi, self.blocks = dense_features(q,self.t,representation)
        self.motion = np.unwrap(arrays['joint'][:,[0,1,2,3,4,5,7,8,9,10,11,12]],axis=0)
        self.events = np.array([[int(a in c['opening_arms']) for a in ['left','right']] for c in meta['candidates']])
        self.timing = np.column_stack([self.t[self.frames]/self.end_t,
                                      np.arange(self.m)/max(1,self.m-1),np.full(self.m,self.end_t),
                                      np.diff(np.r_[0,self.t[self.frames]]),
                                      np.diff(np.r_[self.t[self.frames],self.end_t])])

    def intervals(self, weights, min_duration, min_motion):
        grid = np.arange(self.t[0],self.end_t,.2)
        sampled = np.column_stack([np.interp(grid,self.t,self.phi[:,k]) for k in range(self.phi.shape[1])]) * weights
        endpoints = np.r_[self.t[self.v[:-1]+1],self.end_t]
        bins = np.searchsorted(grid,endpoints,side='left'); bins[-1] = len(grid)
        segments = {}
        for i in range(len(self.v)-1):
            for j in range(i+1,len(self.v)):
                duration = endpoints[j]-endpoints[i]
                count = int(bins[j]-bins[i])
                if duration+1e-9 < min_duration or count < 2:
                    continue
                data = sampled[bins[i]:bins[j]]
                progress = np.linspace(0,len(data)-1,N_PROGRESS)
                seq = np.column_stack([np.interp(progress,np.arange(len(data)),data[:,k]) for k in range(data.shape[1])])
                a,b = int(self.v[i]+1),int(self.v[j]+1)
                motion = float(np.max(np.ptp(self.motion[a:b],axis=0)))
                segments[i,j] = {'seq':seq,'count':count,'duration':duration,
                                  'can_be_flip':motion>=min_motion, 'motion_range':motion}
        return segments


def best_two(items):
    output=[];seen=set()
    for cost,path in sorted(items,key=lambda x:x[0]):
        token=tuple(path)
        if not np.isfinite(cost) or token in seen:continue
        seen.add(token);output.append((float(cost),path))
        if len(output)==2:break
    return output
