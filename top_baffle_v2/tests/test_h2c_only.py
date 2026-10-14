"""Deprecated printer paths fail before launching work; H2C stays guarded."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def make(*args):
    return subprocess.run(['make', *args], cwd=ROOT, capture_output=True, text=True)


def main():
    for target in ('help', 'all', 'to_print', 'h2c_prepare'):
        result = make('-n', 'PRINTER=P2S', target)
        assert result.returncode != 0 and 'only supported printer' in result.stderr
    for target in ('delivery_refresh', 'to_print_06hf', 'bambu_slice_release', 'obiwan_petg_gui_projects'):
        result = make('-n', target)
        assert result.returncode != 0 and 'deprecated' in result.stderr
    result = make('-n', 'LX_BAMBU_GRAPH_ONLY=1', 'delivery_refresh')
    assert result.returncode != 0 and 'requires SHELL=' in result.stderr
    for target in ('all', 'h2c_geometry', 'h2c_prepare', 'h2c_validate', 'h2c_docs', 'h2c_review'):
        result = make('-n', target)
        assert result.returncode == 0, result.stderr
        assert 'run_memory_guarded.py --' in result.stdout
        assert 'scripts/h2c_pipeline.py' in result.stdout
    result = make('help')
    assert result.returncode == 0
    assert 'PRINTER=P2S' not in result.stdout and 'Only supported printer: H2C' in result.stdout
    print('H2C-only entry points passed')


if __name__ == '__main__': main()
