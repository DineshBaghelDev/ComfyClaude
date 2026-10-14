"""Test the actual MCP recolour tool without requiring ComfyUI/model generation."""
import argparse
import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imaging
import recoloring


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--color', default='#3267C8')
    options = parser.parse_args()
    source, mask_path = options.source.resolve(), options.mask.resolve()
    dest = ROOT/'outputs/recolor_checks'
    dest.mkdir(parents=True, exist_ok=True)
    async with stdio_client(StdioServerParameters(command=sys.executable,
            args=[str(ROOT/'server.py')], cwd=str(ROOT))) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            schema = await session.list_tools()
            assert 'recolor_image' in [tool.name for tool in schema.tools]
            saved = dest/'recolored.png'
            result = await session.call_tool('recolor_image', {'image_path': str(source),
                'mask_path': str(mask_path), 'color': options.color, 'save_to': str(saved)},
                read_timeout_seconds=timedelta(seconds=240))
            text = '\n'.join(c.text for c in result.content if c.type == 'text')
            print(text, flush=True)
            assert not result.isError, text
            output = next(Path(line) for line in text.splitlines()
                          if line.endswith('.png') and Path(line).exists())
            metadata = json.loads(output.with_suffix('.json').read_text())
            original = recoloring.load_image(source)
            edited = recoloring.load_image(output)
            assert original.size == edited.size
            assert edited.tobytes() == recoloring.load_image(saved).tobytes()
            applied = Image.open(metadata['applied_mask']).convert('L')
            diff = ImageChops.difference(original.convert('RGB'), edited.convert('RGB'))
            unselected = applied.point(lambda value: 0 if value else 255)
            protected = Image.composite(diff, Image.new('RGB', diff.size), unselected)
            assert protected.getbbox() is None, 'unselected pixels changed'
            assert metadata['changed_pixels'] > 0
            assert metadata['max_lightness_error'] < .003
            if 'A' in original.getbands():
                assert original.getchannel('A').tobytes() == edited.getchannel('A').tobytes()
            for params in ({'color': 'invalid-colour', 'mask_path': str(mask_path)},
                           {'color': 'blue'}, {'color': 'blue', 'mask_path': str(mask_path), 'strength': 2}):
                invalid = await session.call_tool('recolor_image', {'image_path': str(source), **params})
                assert invalid.isError, 'invalid recolour input accepted'
            comparison = dest/'comparison.png'
            imaging.contact_sheet([source, output], comparison, ['original', 'masked recolour'], thumb=768)
            report = {'source': str(source), 'mask': str(mask_path), 'output': str(output),
                      'saved_to': str(saved), 'comparison': str(comparison),
                      'size': original.size, 'protected_pixels': 'exact match',
                      'max_lightness_error': metadata['max_lightness_error'],
                      'changed_pixels': metadata['changed_pixels'], 'invalid_inputs': 'rejected'}
            (dest/'report.json').write_text(json.dumps(report, indent=2))
            print('PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
