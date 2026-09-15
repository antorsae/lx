"""Compare archived and current STL renders at the same camera and scale."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from rebuild import sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    history=json.loads((HERE/'superseded.json').read_text())
    old=ROOT/history['previous_rectangular_lower_band']/'candidates/nd25fn4_crescent/views/UM_solid_underside.png'
    new=HERE/'views/UM_solid_underside.png'
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',26)
    canvas=Image.new('RGB',(2040,345),'#f5f7f9')
    draw=ImageDraw.Draw(canvas)
    for index,(path,label) in enumerate([(old,'Previous: rectangular rear band'),
                                          (new,'Updated: continuous curved band')]):
        source=Image.open(path).convert('RGB')
        assert source.size==(1400,1000),('camera crop changed',source.size)
        panel=source.crop((40,310,1360,670)).resize((1000,273),Image.Resampling.LANCZOS)
        x=10+1020*index
        draw.text((x+14,15),label,font=font,fill='#213441')
        canvas.paste(panel,(x,61))
    output=HERE/'views/UM_lower_band_comparison.png'
    canvas.save(output)
    manifest={'source_sha256':sha(__file__),
        'source_build_manifest_sha256':sha(HERE/'build_manifest.json'),
        'inputs':{str(path.relative_to(ROOT)):sha(path) for path in (old,new)},
        'output_sha256':sha(output),
        'note':'Same actual-STL camera and scale. Only crop, layout and labels added.'}
    (HERE/'views/lower_band_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
