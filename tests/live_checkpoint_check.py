"""Generate one real NoobAI comic panel through the plugin's MCP stdio transport."""
import asyncio
import json
import re
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]


async def main():
    async with stdio_client(StdioServerParameters(command=sys.executable,
            args=[str(ROOT / 'server.py')], cwd=str(ROOT))) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert 'generate_with_model' in [tool.name for tool in tools.tools]
            caps = await session.call_tool('list_capabilities', {})
            captext = '\n'.join(c.text for c in caps.content if c.type == 'text')
            print(captext, flush=True)
            invalid = await session.call_tool('generate_with_model', {'prompt': 'panel', 'model': 'missing-profile'})
            assert invalid.isError
            result = await session.call_tool('generate_with_model', {
                'model': 'noobai_vpred', 'prompt': '1boy, adult man, solo, short layered black hair, brown eyes, blue collared shirt, '
                    'calm determined expression, (portrait:1.25), (upper body:1.2), looking at viewer, face focus, '
                    'centered composition, softly blurred sunlit stone town square, detailed eyes, detailed hair, '
                    'clean delicate lineart, anime illustration, soft daylight, warm highlights, cool shadows, rich colors, subtle gradients',
                'negative_prompt': 'nsfw, worst quality, low quality, lowres, bad anatomy, deformed, bad hands, '
                    'extra fingers, extra limbs, disfigured face, asymmetrical eyes, blurry, full body, distant figure, '
                    'small subject, silhouette, dark face, underexposed, text, speech bubble, watermark, logo',
                'width': 832, 'height': 1216, 'seed': 42,
                'save_to': str(ROOT / 'outputs/noobai_checks/panel.png')},
                read_timeout_seconds=timedelta(seconds=240))
            def result_text(value):
                return '\n'.join(c.text for c in value.content if c.type == 'text')
            text = result_text(result)
            assert not result.isError, text
            print(text, flush=True)
            for _ in range(24):
                if 'Images:' in text:
                    break
                pid = re.search(r'Job ([\w-]+)', text)
                assert pid, text
                result = await session.call_tool('job_status', {'prompt_id': pid.group(1), 'wait_seconds': 50},
                    read_timeout_seconds=timedelta(seconds=90))
                text = result_text(result)
                assert not result.isError, text
                print(text, flush=True)
            assert 'Images:' in text, text
            path = next(Path(line) for line in text.split('Images:\n', 1)[1].splitlines() if line.endswith('.png'))
            image = Image.open(path).convert('RGB')
            assert image.size == (832, 1216), image.size
            assert max(ImageStat.Stat(image).stddev) > 10, 'flat/blank output'
            assert ImageStat.Stat(image.convert('L')).mean[0] > 35, 'daylight benchmark is underexposed'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['model'] == 'noobai_vpred'
            assert meta['seed'] == 42
            assert meta['profile']['prediction_type'] == 'v_prediction'
            assert Path(meta['saved_to'][0]).exists()
            report = {'output': str(path), 'metadata': meta, 'size': image.size, 'mcp': 'PASS'}
            (ROOT / 'outputs/noobai_checks/report.json').write_text(json.dumps(report, indent=2))
            print('PASS', str(path), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
