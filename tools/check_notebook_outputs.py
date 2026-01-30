from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

LOGGER = logging.getLogger('nbclean')


@dataclass(frozen=True)
class DirtyCell:
    cell_index: int
    cell_id: Optional[str]
    reasons: List[str]
    source_preview: str


@dataclass(frozen=True)
class NotebookReport:
    path: Path
    dirty_cells: List[DirtyCell]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f'failed to read file: {e}') from e

    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f'failed to parse JSON: {e}') from e

    if not isinstance(data, dict):
        raise RuntimeError('notebook root is not a JSON object')

    return data


def _normalize_source(source: Any) -> str:
    """
    Jupyter cell 'source' can be a list of lines or a string.
    """
    if source is None:
        return ''
    if isinstance(source, list):
        return ''.join(str(x) for x in source)
    return str(source)


def _preview(text: str, max_len: int = 140) -> str:
    text = ' '.join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + '…'


def _cell_is_dirty(cell: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Consider a cell 'dirty' if:
      - outputs is non-empty (typical executed output)
      - execution_count is not None (code cell has been run)
    """
    reasons: list[str] = []

    if cell.get('cell_type') != 'code':
        return False, reasons

    outputs = cell.get('outputs') or []
    execution_count = cell.get('execution_count', None)

    if isinstance(outputs, list) and len(outputs) > 0:
        reasons.append(f'outputs[{len(outputs)}]')

    if execution_count is not None:
        reasons.append(f'execution_count={execution_count!r}')

    return (len(reasons) > 0), reasons


def inspect_notebook(path: Path) -> NotebookReport:
    nb = _read_json(path)
    cells = nb.get('cells', [])

    if not isinstance(cells, list):
        raise RuntimeError("notebook 'cells' is not a list")

    dirty_cells: list[DirtyCell] = []
    for idx, cell_any in enumerate(cells):
        if not isinstance(cell_any, dict):
            LOGGER.debug('Skipping non-object cell at %s[%d]', path, idx)
            continue

        is_dirty, reasons = _cell_is_dirty(cell_any)
        if not is_dirty:
            continue

        cell_id = cell_any.get('id')
        source = _normalize_source(cell_any.get('source'))
        dirty_cells.append(
            DirtyCell(
                cell_index=idx,
                cell_id=str(cell_id) if cell_id is not None else None,
                reasons=reasons,
                source_preview=_preview(source),
            )
        )

    return NotebookReport(path=path, dirty_cells=dirty_cells)


def iter_notebooks(root: Path) -> Iterable[Path]:
    for p in root.rglob('*.ipynb'):
        if '.ipynb_checkpoints' in p.parts:
            continue
        yield p


def _should_color() -> bool:
    return sys.stdout.isatty() and os.environ.get('NO_COLOR') is None


def _fmt_report(report: NotebookReport, color: bool) -> str:
    def c(s: str, code: str) -> str:
        if not color:
            return s
        return f'\033[{code}m{s}\033[0m'

    lines: list[str] = []
    lines.append(f'{c("X", "31")} {report.path.as_posix()}')
    for cell in report.dirty_cells:
        ident = f'cell[{cell.cell_index}]'
        if cell.cell_id:
            ident += f' id={cell.cell_id}'
        reason = ', '.join(cell.reasons)
        preview = cell.source_preview or '<empty source>'
        lines.append(f'  - {ident}: {reason}')
        lines.append(f'    ↳ {preview}')
    return '\n'.join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fail if any Jupyter notebook contains executed outputs or execution counts.'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('.'),
        help='Root directory to scan (default: current directory).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging.',
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Exit on first dirty notebook found.',
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    root: Path = args.root.resolve()
    if not root.exists():
        LOGGER.error('Root path does not exist: %s', root)
        return 2

    LOGGER.debug('Scanning for notebooks under: %s', root)

    reports: list[NotebookReport] = []
    try:
        for nb_path in iter_notebooks(root):
            LOGGER.debug('Inspecting: %s', nb_path)
            report = inspect_notebook(nb_path)
            if report.dirty_cells:
                reports.append(report)
                if args.fail_fast:
                    break
    except RuntimeError as e:
        LOGGER.error('Notebook check failed: %s', e)
        return 2
    except Exception as e:
        LOGGER.exception('Unexpected error: %s', e)
        return 2

    if reports:
        color = _should_color()
        print('Notebooks with outputs/execution counts found:\n')
        for r in reports:
            print(_fmt_report(r, color=color))
            print()
        print('Please clear outputs before committing.')
        print('Tip: JupyterLab → Edit → Clear Outputs of All Cells')
        return 1

    print('All notebooks are clean (no outputs, no execution counts).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
