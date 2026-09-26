import json
from pathlib import Path
import sys

spec = json.loads(Path(sys.argv[1]).read_text())
lines = ['#pragma once', 'namespace muzero {', f"inline constexpr int PROTOCOL_VERSION = {spec['version']};", f"inline constexpr int INPUT_PLANES = {len(spec['planes'])};", f"inline constexpr char GAME_MAGIC[] = \"{spec['magic']}\";"]
lines.append('enum class Rule { ' + ', '.join(f'{name.upper()} = {i}' for i, name in enumerate(spec['rules'])) + ' };')
lines.append('enum class Plane { ' + ', '.join(f'{name} = {i}' for i, name in enumerate(spec['planes'])) + ' };')
lines.append('inline constexpr const char* RULE_NAMES[] = {' + ', '.join('\"' + name + '\"' for name in spec['rules']) + '};')
lines.append('}')
Path(sys.argv[2]).write_text('\n'.join(lines) + '\n')
