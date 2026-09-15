"""Verify current CAD Viewer assets and relative-path browser review links."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urljoin
from urllib.request import urlopen

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
NAMES = ['nd25fn4_color_review.glb', 'nd25fn4_flat_wings.glb',
         'nd25fn4_graded_wings.glb', 'LM_interface_detail.glb']


async def verify(base):
    with urlopen(base + '/__cad/catalog', timeout=30) as response:
        catalog = json.load(response)['entries']
    reports = {}
    async with async_playwright() as runtime:
        browser = await runtime.chromium.launch(
            headless=True, args=['--enable-unsafe-swiftshader'])
        try:
            for name in NAMES:
                path = (HERE / 'views' / name).resolve()
                entry = next(row for row in catalog if row['file'] == str(path))
                with urlopen(urljoin(base, entry['url']), timeout=30) as response:
                    payload = response.read()
                    status = response.status
                digest = hashlib.sha256(payload).hexdigest()
                assert digest == hashlib.sha256(path.read_bytes()).hexdigest(), name
                relative = entry['rootRelativeFile']
                # The UI selects catalog entries by rootRelativeFile. An
                # absolute file query fails even when its asset returns 200.
                assert not relative.startswith('/')
                url = base + '/?file=' + quote(relative, safe='')
                page = await browser.new_page(viewport={'width': 1400, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(url, wait_until='networkidle', timeout=60000)
                await page.wait_for_function(
                    "document.body.innerText.includes('Measure')", timeout=30000)
                text = await page.locator('body').inner_text()
                assert name in text and 'FILE DOES NOT EXIST' not in text, (name, text)
                assert await page.locator('canvas').count() > 0 and not errors, (name, errors)
                screenshot = HERE.parents[1] / 'review/nd25fn4_print' / (path.stem + '_viewer.png')
                await page.screenshot(path=str(screenshot))
                await page.close()
                reports[name] = dict(viewer_url=url, catalog_file=str(path),
                    asset_http_status=status, served_sha256=digest, bytes=len(payload),
                    viewer_file_parameter=relative, browser_verified=True,
                    browser_result='Model loaded; no missing-file banner or JavaScript errors',
                    browser_screenshot=str(screenshot.relative_to(HERE.parents[1])))
                print(name, 'asset and browser passed', flush=True)
        finally:
            await browser.close()
    detail = reports.pop('LM_interface_detail.glb')
    (HERE / 'views/LM_interface_viewer_validation.json').write_text(json.dumps(detail, indent=2) + '\n')
    (HERE / 'views/viewer_link_validation.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='URL of the already started CAD Viewer')
    args = parser.parse_args()
    asyncio.run(verify(args.base_url.rstrip('/')))
