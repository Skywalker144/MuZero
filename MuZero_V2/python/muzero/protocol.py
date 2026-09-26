import json
from enum import IntEnum
from pathlib import Path

SPEC = json.loads((Path(__file__).resolve().parents[2] / 'protocol.json').read_text())
VERSION = SPEC['version']
MAGIC = SPEC['magic'].encode('ascii')
Plane = IntEnum('Plane', {name: index for index, name in enumerate(SPEC['planes'])})
Rule = IntEnum('Rule', {name.upper(): index for index, name in enumerate(SPEC['rules'])})
INPUT_PLANES = len(Plane)
