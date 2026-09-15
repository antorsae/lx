"""Check requested native P2S changeover volume and flow, not only metadata."""
import re
from pathlib import Path
from lx521_baffle.print_policy import policy


def audit_changeover_text(text, *, purge_mm3=None, pla_flow_mm3_s=None):
    p = policy()['materials']
    purge = p['purge_each_direction_mm3'] if purge_mm3 is None else purge_mm3
    speed = p['interface_flush_volumetric_speed_mm3_s'] if pla_flow_mm3_s is None else pla_flow_mm3_s
    # Native P2S templates use this filament cross-section for L and F.
    filament_area = 2.4053
    starts = list(re.finditer(r'^M620 S([01])A\s*$', text, re.M))
    rows = []
    for i, start in enumerate(starts):
        block = text[start.end():starts[i+1].start() if i+1 < len(starts) else len(text)]
        target = int(start.group(1))
        end = re.search(rf'^M621 S{target}A\s*$', block, re.M)
        assert end, ('Incomplete native material change', target)
        block = block[:end.end()]
        requests = {}
        for line in block.splitlines():
            match = re.match(r'M620\.10 A([01])\s+(.*)', line.strip())
            if not match: continue
            values = {k: float(v) for k, v in re.findall(r'([FLTP])(-?\d+(?:\.\d+)?)', match.group(2))}
            assert {'F', 'L', 'T', 'P'} <= values.keys(), line
            requests[int(match.group(1))] = values
        assert set(requests) == {0, 1}, requests
        for values in requests.values():
            assert abs(values['L'] * filament_area - purge) < .025, ('Stale or wrong flush volume', values, purge)
        # A0 is outgoing; A1 is incoming. Test PLA on both sides of the swap.
        pla = requests[1 if target == 1 else 0]
        actual_speed = pla['F'] * filament_area / 60
        assert abs(actual_speed - speed) < .002, ('Stale or wrong PLA flush flow', actual_speed, speed)
        virtual = re.search(r'; VFLUSH_START\s*(.*?)\s*; VFLUSH_END', block, re.S)
        assert virtual, 'Missing native virtual flush block'
        lengths = [float(v) for v in re.findall(r'^;VG1 E([\d.]+)\s', virtual.group(1), re.M)]
        assert lengths and abs(sum(lengths) * filament_area - purge) < .03, ('Virtual flush volume mismatch', lengths, purge)
        assert re.search(rf'^T{target}\s*$', block, re.M), target
        rows.append(dict(from_tool=1-target, to_tool=target,
                         requested_volume_mm3=requests[1]['L'] * filament_area,
                         pla_flush_mm3_s=actual_speed,
                         outgoing_flush_c=requests[0]['T'], incoming_flush_c=requests[1]['T'],
                         incoming_print_c=requests[1]['P']))
    assert rows and {r['to_tool'] for r in rows} == {0, 1}, 'Both material-change directions must occur'
    return dict(status='pass', change_count=len(rows), purge_mm3_each_direction=purge,
                pla_flush_limit_mm3_s=speed, changes=rows,
                scope='Requested native G-code parameters; actual temperature, extrusion and clog-free operation require a physical test.')


def audit_changeover(gcode):
    return audit_changeover_text(Path(gcode).read_text())
