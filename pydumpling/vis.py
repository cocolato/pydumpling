from linecache import getline
from os import system
from uuid import uuid4
from os.path import splitext, abspath, dirname, join as pjoin
from json import dumps

from pydumpling import load_dumpling

__dir__ = dirname(abspath(__file__))

def _wrap_code(raw_code: str) -> str:
    return f'<code>{raw_code}</code>'

def _generate_label(f_code,lineno) -> str:
    return f'<b>line: {lineno}</b> \n{_wrap_code("func: " + f_code["co_name"])} \n{_wrap_code(getline(f_code["co_filename"], lineno).strip())}'


def extract_frame(traceback: dict) -> list[dict]:
    frames = []
    for key, value in traceback.items():
        if key == 'tb_frame':
            f_code = value['f_code']
            frames.append(
                {
                    'id': str(uuid4()),
                    'label': _generate_label(f_code, value["f_lineno"]),
                    'extra': {
                        'co_filename': f_code['co_filename'],
                        'co_type': f_code['co_name'] == '<module>' and 'module' or 'function',
                    },
                }
            )
        if isinstance(value, dict):
            frames.extend(extract_frame(value))

    return frames

def generate_vis(dump_file: str) -> None:
    dumpling_result = load_dumpling(dump_file)
    traceback = dumpling_result['traceback'].serialize()
    raw_frames = extract_frame(traceback)

    res = dumps(raw_frames)
    vis_html_file = splitext(dump_file)[0] + '.html'
    with open(f'{__dir__}/templates/traceback.html', mode='r', encoding='utf-8') as handler:
        template = handler.read()
        traceback_html = template.replace('{{traceback}}', res)
        traceback_html = traceback_html.replace('{{dump_file_name}}', dump_file)
        traceback_html = traceback_html.replace(
            '{{exc_type}}', repr(dumpling_result['exc_extra']['exc_type']).replace('<', '').replace('>', '')
        )
        traceback_html = traceback_html.replace(
            '{{exc_value}}', repr(dumpling_result['exc_extra']['exc_value'])
        )


        with open(vis_html_file, mode="w", encoding="utf-8") as f:
            f.write(traceback_html)

    system(f'start {vis_html_file}')