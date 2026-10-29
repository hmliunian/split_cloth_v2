"""Full-length review videos; main four-stage and exploratory action timelines."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
from pathlib import Path
import json, os
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_io import read,save,sha

COLORS=['#42b7cf','#9c8cf5','#efb85f','#64c396']
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
FONTS={n:font(n) for n in [14,16,18,20,24,28]}

def render_one(meta, ROOT):
    key=meta['key'];p=read(ROOT/'labels'/key/'result.json')
    axis=np.load(ROOT/meta['dataset_npz'])['timestamps'];n=len(axis)
    output=ROOT/'videos'/f'{key}__reverse_suffix.mp4';temporary=output.with_suffix('.part.mp4')
    if output.exists():
        record=read(output.with_suffix('.json'))
        assert record['output_sha256']==sha(output)
        assert record['result_sha256']==sha(ROOT/'labels'/key/'result.json')
        assert record['source_sha256']==sha(meta['source_video_path'])
        assert record['renderer_sha256']==sha(Path(__file__))
        return record
    assert not temporary.exists(), 'An interrupted partial export exists; use a new output run'
    before=sha(meta['source_video_path'])
    base=Image.new('RGB',(1280,720),'#101a2a');draw=ImageDraw.Draw(base)
    draw.text((24,18),f'Reverse sample {meta["sample_rank"]:02d}/{meta["sample_count"]} | {meta["episode_id"]} | Fold-cloth model',font=FONTS[24],fill='white')
    draw.text((24,51),'Search S5 -> S4 -> S3 -> S2 | Free S1 prefix | Full-length original speed | Unconfirmed',font=FONTS[16],fill='#a8bad4')
    draw.rounded_rectangle((690,82,1255,545),radius=12,fill='#1b2940')
    draw.text((710,140),'Display = (S1 + S2), S3, S4, S5',font=FONTS[18],fill='white')
    mapping=[('b4','S5 starts: move / place / finish'),('b3','S4 starts: final fold'),('b2','S3 starts: second sleeve')]
    for s,(boundary,name) in enumerate(mapping):
        y=180+s*66;f=p['display_frames'][int(boundary[1])-2]
        draw.text((710,y),f'{boundary}  {name}',font=FONTS[18],fill=COLORS[s])
        text=f'Previous stage ends: f{f} / {axis[f]:.2f}s' if f is not None else 'Not determined'
        draw.text((710,y+26),text,font=FONTS[16],fill='#d7e4f3')
    b1=p['action_boundaries']['b1'];raw=p['raw_action_frames'][0]
    draw.text((710,389),f'b1: {"f"+str(b1["frame"]) if b1 else "unknown"}   raw: {"f"+str(raw) if raw is not None else "unknown"}',font=FONTS[18],fill='#f2c178')
    draw.text((710,419),'S1: free preparation; no flip count',font=FONTS[18],fill='#f2c178')
    draw.text((710,451),f'Model status: {p["status"]}',font=FONTS[16],fill='#d7e4f3')
    draw.text((710,479),'Placement terminal status: not evaluated',font=FONTS[16],fill='#a8bad4')
    draw.text((710,508),'For visual inspection; no reference scoring',font=FONTS[16],fill='#a8bad4')
    x0,x1=24,1255;length=x1-x0
    def xboundary(f):return int(round(x0+(f+1)/n*length))
    draw.text((x0,551),'V1-V4 display',font=FONTS[14],fill='#b9c8df')
    bounds=p['display_frames']
    if all(f is not None for f in bounds):
        for k,(a,b) in enumerate(zip([-1,*bounds],[*bounds,n-1])):
            xa,xb=xboundary(a),xboundary(b)
            draw.rectangle((xa,573,xb,594),fill=COLORS[k]);draw.text((xa+4,574),f'V{k+1}',font=FONTS[14],fill='#07121f')
    else:draw.rectangle((x0,573,x1,594),fill='#536076')
    draw.text((x0,601),'Raw action hypothesis (unconfirmed) | S1 preparation is not scored',font=FONTS[14],fill='#b9c8df')
    if not p.get('segments'):
        draw.rectangle((x0,624,x1,644),fill='#536076')
        draw.text((x0+4,625),'No feasible action decomposition',font=FONTS[14],fill='white')
    for segment in p.get('segments',[]):
        stage=segment['stage'];a,b=segment['start_frame']-1,segment['end_frame'];xa,xb=xboundary(a),xboundary(b)
        if stage=='PREP':label='S1 prep';color='#536076'
        else:label='S'+stage[1:];color=COLORS[int(stage[1:])-2]
        draw.rectangle((xa,624,xb,644),fill=color)
        if xb-xa>24:draw.text((xa+3,625),label,font=FONTS[14],fill='white')
        draw.line((xb,624,xb,644),fill='#d3dce7',width=1)
    draw.text((x0,653),'Opening candidates | gold = selected b2-b4 | gray = other candidates',font=FONTS[14],fill='#b9c8df')
    draw.line((x0,683,x1,683),fill='#6f8098',width=1)
    for c in meta['candidates']:
        f=c['source_frame_id'];xx=xboundary(f)
        draw.line((xx,677,xx,690),fill='#ffd066' if f in bounds else '#7786a2',width=2)
    with av.open(meta['source_video_path']) as source, av.open(str(temporary),'w',options={'movflags':'+faststart'}) as target:
        stream=target.add_stream('libx264',rate=Fraction(meta['source_fps']))
        stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,90000)
        stream.codec_context.time_base=Fraction(1,90000);stream.codec_context.thread_count=2
        stream.options={'crf':'20','preset':'veryfast'}
        count=0
        for i,frame in enumerate(source.decode(video=0)):
            img=base.copy();d=ImageDraw.Draw(img)
            original=frame.to_image()
            scale=min(650/original.width,460/original.height)
            original=original.resize((round(original.width*scale),round(original.height*scale)),Image.Resampling.LANCZOS)
            img.paste(original,(24+(650-original.width)//2,83+(460-original.height)//2))
            stage=next((s+1 for s,f in enumerate(bounds) if f is not None and i<=f),4) if all(f is not None for f in bounds) else '?'
            d.text((710,98),f'Now V{stage} | f{i}/{n-1} | {axis[i]:.2f} s',font=FONTS[24],fill='white')
            xx=int(x0+(i+.5)/n*length);d.line((xx,570,xx,700),fill='white',width=2)
            out=av.VideoFrame.from_image(img);out.pts=round(float(axis[i])*90000);out.time_base=Fraction(1,90000)
            for packet in stream.encode(out):target.mux(packet)
            count+=1
        for packet in stream.encode():target.mux(packet)
    assert count==n
    pts=[];snaps={0,n-1}
    snaps.add(bounds[0]+1 if bounds[0] is not None else n//2)
    (ROOT/'plots'/'video_checks').mkdir(exist_ok=True)
    with av.open(str(temporary)) as video:
        stream=video.streams.video[0]
        assert stream.codec_context.name=='h264' and stream.width==1280 and stream.height==720
        assert stream.codec_context.pix_fmt=='yuv420p'
        for i,frame in enumerate(video.decode(video=0)):
            pts.append(float(frame.pts*frame.time_base))
            if i in snaps:frame.to_image().save(ROOT/'plots'/'video_checks'/f'{key}_{i}.jpg')
    error=float(np.max(np.abs(np.asarray(pts)-axis)))
    assert len(pts)==n and error<=1/90000+1e-9
    assert sha(meta['source_video_path'])==before==meta['source_video_sha256']
    temporary.rename(output)
    result={'key':key,'path':str(output.relative_to(ROOT)),'frames':n,'max_pts_error_s':error,
            'codec':'h264','pixel_format':'yuv420p','size':[1280,720],
            'source_sha256':before,'output_sha256':sha(output),
            'result_sha256':sha(ROOT/'labels'/key/'result.json'),
            'renderer_sha256':sha(Path(__file__)),
            'action_track':'raw reverse suffix path with unscored S1 preparation, explicitly marked unconfirmed',
            'reverse_search_order':['S5','S4','S3','S2'],
            'codex_review_performed':False,'reference_scores_displayed':False,
            'complete_video_verified':True}
    save(output.with_suffix('.json'),result)
    return result

def render_run(ROOT, workers=2):
    ROOT=Path(ROOT).resolve()
    entries=read(ROOT/'manifest.json')
    for name,digest in read(ROOT/'prediction_sha256.json').items():
        assert sha(ROOT/name)==digest, name
    with ProcessPoolExecutor(max_workers=workers,max_tasks_per_child=1) as pool:
        results=[]
        futures=[pool.submit(render_one,e,ROOT) for e in entries]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print('video verified',len(results),'of',len(entries),result['key'],result['frames'],flush=True)
    ranks={e['key']:e['sample_rank'] for e in entries};results.sort(key=lambda r:ranks[r['key']])
    save(ROOT/'video_verification.json',{'videos':len(results),'frames':sum(r['frames'] for r in results),
                                       'max_pts_error_s':max(r['max_pts_error_s'] for r in results),'details':results})
    for start in range(0,len(entries),10):
        group=entries[start:start+10];contact=Image.new('RGB',(1280,240*len(group)),'#101a2a')
        for row,e in enumerate(group):
            p=read(ROOT/'labels'/e['key']/'result.json');n=e['frame_count']
            frames=[0,(p['display_frames'][0]+1 if p['display_frames'][0] is not None else n//2),n-1]
            for col,f in enumerate(frames):
                img=Image.open(ROOT/'plots'/'video_checks'/f'{e["key"]}_{f}.jpg');img.thumbnail((426,240))
                contact.paste(img,(col*426,row*240))
        contact.save(ROOT/'plots'/f'video_contact_sheet_{start+1:02d}_{start+len(group):02d}.jpg')

    write_gallery(ROOT,entries,results)

def write_gallery(root, entries, results):
    import html
    head = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S5→S2 逆序叠衣视频</title><style>body{max-width:1200px;margin:24px auto;padding:0 16px;font:16px/1.6 system-ui;background:#101a2a;color:#e4edf8}a{color:#76d8ea}details{background:#1b2940;padding:14px;margin:14px 0;border-radius:8px}summary{cursor:pointer}video{width:100%}</style><h1>S5→S4→S3→S2 逆序叠衣分段</h1><p>S1 是自由整理前缀，不推断翻面次数。全部是未确认的模型预测；未做 Codex 复核或准确率评分。</p>'
    rows=[head]
    for meta, record in zip(entries, results):
        key=meta['key'];result=read(root/'labels'/key/'result.json')
        label=html.escape(f"{meta['sample_rank']:02d} · {meta['episode_id']} · {meta['frame_count']} 帧")
        path=html.escape(record['path'],quote=True)
        rows.append(f'<details><summary>{label}</summary><p>b1/b2/b3/b4: {result["action_frames"]}</p><video controls preload="none" src="{path}"></video><p><a href="{path}">完整 MP4</a></p></details>')
    rows.append('</html>')
    (root/'index.html').write_text('\n'.join(rows))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    render_run(args.run,args.workers)
