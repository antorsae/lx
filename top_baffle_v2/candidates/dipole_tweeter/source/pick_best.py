"""Pick the best seed result per configuration into routes3_<key>_h.json (sealed first, then largest R)."""
import json,glob,os,re
SC=os.environ['SC']
for key in ('pt2522_4_portrait','pt2522_4_landscape','amt17d2_1_portrait','amt17d2_1_landscape'):
    c=[]
    for f in glob.glob(SC+f'routes3_{key}_h*.json'):
        try: r=json.load(open(f))['t']
        except Exception: continue
        c.append((r['bad'],-r['R'],f,r))
    c.sort(key=lambda t:(t[0],t[1]));b=c[0]
    json.dump({'t':b[3]},open(SC+f'routes3_{key}_h.json','w'));print(key,'best',os.path.basename(b[2]),'bad',b[0],'R',round(-b[1],1))
