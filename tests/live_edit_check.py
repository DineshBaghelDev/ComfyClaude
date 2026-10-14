import asyncio
import argparse
from datetime import timedelta
import json
import re
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imaging


async def main():
    parser = argparse.ArgumentParser(description='Run real local MCP generation/edit checks (requires ComfyUI and models).')
    parser.add_argument('--source', type=Path, help='Reuse an existing source image instead of generating one.')
    parser.add_argument('--feather', type=int, choices=range(32), default=16,
                        help='Inpaint feather radius to exercise (0-31, default 16).')
    parser.add_argument('--edit-model', default='auto', help='auto, sd15_inpaint, z_image, or configured editing model.')
    options = parser.parse_args()
    async with stdio_client(StdioServerParameters(command=sys.executable, args=[str(ROOT / 'server.py')], cwd=str(ROOT))) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            async def call(name, params):
                result = await session.call_tool(name, params, read_timeout_seconds=timedelta(seconds=240))
                texts = '\n'.join(c.text for c in result.content if c.type == 'text')
                print(name, texts, flush=True)
                assert not result.isError, texts
                deadline = time.monotonic() + 1800
                while 'Images:' not in texts and ('job' in texts.lower() or 'running' in texts.lower()):
                    assert 'failed:' not in texts.lower(), texts
                    assert time.monotonic() < deadline, 'MCP job exceeded 30 minutes'
                    match = re.search(r'[0-9a-f]{8}-[0-9a-f-]{27,}', texts)
                    assert match, texts
                    result = await session.call_tool('job_status', {'prompt_id': match.group(), 'wait_seconds': 50})
                    texts = '\n'.join(c.text for c in result.content if c.type == 'text')
                    print('job_status', texts, flush=True)
                    assert not result.isError, texts
                paths = [Path(line.strip()) for line in texts.splitlines() if line.strip().endswith('.png') and Path(line.strip()).exists()]
                return paths

            await call('comfy_status', {})
            await call('list_capabilities', {})
            source = options.source.resolve() if options.source else (await call('generate_image', {'prompt': 'A realistic photograph of a quiet alpine lake with pine trees on the shore and mountains in the distance, clear blue sky, consistent natural daylight, no people', 'width': 768, 'height': 768, 'seed': 20261014, 'quality': 'balanced'}))[0]
            megapixels = 0.26  # Keep the smoke test within a 6 GB GPU's decode budget.
            padding = 128
            prepared, w, h = imaging.prepare(source, ROOT / 'outputs/_work', megapixels)
            original = Image.open(prepared).convert('RGB')
            inp = (await call('inpaint', {'image_path': str(source), 'prompt': 'blue lake water, small gentle ripples, uniform water surface texture, realistic photograph', 'region': [0.28, 0.60, 0.65, 0.80], 'megapixels': megapixels, 'seed': 12345, 'feather': options.feather, 'grow': 16, 'edit_model': options.edit_model}))[0]
            edited = Image.open(inp).convert('RGB')
            assert edited.size == original.size
            diff = ImageChops.difference(original, edited)
            x0, y0, x1, y1 = diff.getbbox() or (0, 0, 0, 0)
            margin = 16 + options.feather + 2  # grow + feather, plus raster rounding
            assert x0 >= round(w * 0.28) - margin and x1 <= round(w * 0.65) + margin + 1
            assert y0 >= round(h * 0.60) - margin and y1 <= round(h * 0.80) + margin + 1
            assert diff.getbbox(), 'inpaint did not change any pixels'
            out = (await call('outpaint', {'image_path': str(source), 'prompt': 'A realistic panoramic photograph of a quiet alpine lake with pine trees continuing along the shore and mountains in the distance, clear blue sky, consistent natural daylight, no people', 'left': padding, 'right': padding, 'megapixels': megapixels, 'seed': 12345, 'feather': 40, 'edit_model': options.edit_model}))[0]
            expanded = Image.open(out).convert('RGB')
            assert expanded.size == (w + 2 * padding, h)
            assert ImageChops.difference(original.crop((40, 0, w - 40, h)), expanded.crop((padding + 40, 0, padding + w - 40, h))).getbbox() is None, 'outpaint changed protected pixels'
            for name, params in [('inpaint', {'image_path': str(source), 'prompt': 'test', 'feather': 32}), ('outpaint', {'image_path': str(source), 'prompt': 'test', 'right': 32, 'feather': -1})]:
                result = await session.call_tool(name, params)
                assert result.isError, 'invalid feather accepted'
            report = {'source': str(source), 'inpaint': str(inp), 'outpaint': str(out), 'inpaint_feather': options.feather, 'edit_model': options.edit_model,
                      'protected_pixels': 'exact match', 'invalid_parameters': 'rejected'}
            (ROOT / 'outputs/mcp_edit_check.json').write_text(json.dumps(report, indent=2))
            imaging.contact_sheet([source, inp, out], ROOT / 'outputs/mcp_edit_check.png', ['original', 'inpaint', 'outpaint'], thumb=512)
            print('PASS', json.dumps(report), flush=True)

if __name__ == '__main__':
    asyncio.run(main())
