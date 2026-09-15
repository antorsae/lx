from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from audit_h2c_print import run_job
from lx521_baffle.h2c.printing import write_json
name=sys.argv[1]
j=next(j for j in json.loads((ROOT/'to_print/h2c/catalog.json').read_text())['jobs'] if j['name']==name)
oldwork=ROOT/j['work'];work=ROOT/'build/h2c/experiments'/(name+'_shadow_trial');work.mkdir(parents=True,exist_ok=True)
source=ROOT/j['project'];project=work/source.name;shutil.copy2(source,project)
shutil.copy2(oldwork/'process.json',work/'process.json')
command=json.loads((oldwork/'dry_run.json').read_text());command[command.index('--outputdir')+1]=str(work)
command[command.index('--load-settings')+1]=command[command.index('--load-settings')+1].replace(str(oldwork/'process.json'),str(work/'process.json'));command[-1]=str(project)
write_json(work/'dry_run.json',command);j.update(work=str(work.relative_to(ROOT)),project=str(project.relative_to(ROOT)))
run_job(j)
