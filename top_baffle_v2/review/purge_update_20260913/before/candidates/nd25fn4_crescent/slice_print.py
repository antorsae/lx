"""Run discovery slicing and magnet checks for previously prepared projects."""
import json
import subprocess
import sys
from prepare_print import ROOT,HERE,WORK,slice_command
from finalize_print import KEYS


def main():
    prep=json.loads((WORK/'preparation.json').read_text())
    for key in sys.argv[1:] or KEYS:
        project=ROOT/prep[key]['path'];directory=project.parent
        command=slice_command(project)
        (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (directory/'slice.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        if key!='accessories':
            with (directory/'magnet_analysis.log').open('w') as f:
                subprocess.run([sys.executable,str(HERE/'print_magnets.py'),key],stdout=f,stderr=subprocess.STDOUT,check=True)
        print(key,'discovery completed',flush=True)


if __name__=='__main__':main()
