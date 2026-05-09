#!/usr/bin/env python3
import subprocess
import glob
import re
import os
import sys
import tempfile
from typing import Dict, List, Optional


def run_steiner(input_file: str, modified: bool, out_dir: str) -> Optional[Dict]:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_json = os.path.join(out_dir, f"{base_name}_out.json")

    cmd = [sys.executable, "steiner.py", input_file, "-o", out_json]
    if modified:
        cmd.append("-m")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        if result.returncode != 0:
            print(f"Ошибка запуска {cmd}: {result.stderr}", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"Таймаут для {cmd}", file=sys.stderr)
        return None

    data = {}

    mst_match = re.search(r"Начальный вес MST:\s+(\d+)", output)
    if not mst_match:
        mst_match = re.search(r"Initial MST length:\s+(\d+)", output)
    if mst_match:
        data["initial_mst"] = int(mst_match.group(1))

    length_match = re.search(r"Вес Steiner Tree:\s+(\d+)", output)
    if not length_match:
        length_match = re.search(r"Final Steiner tree length:\s+(\d+)", output)
    if length_match:
        data["final_length"] = int(length_match.group(1))

    time_match = re.search(r"Время вычислений:\s+([\d.]+)\s*s", output)
    if not time_match:
        time_match = re.search(r"Time:\s+([\d.]+)\s*s", output)
    if time_match:
        data["time"] = float(time_match.group(1))

    steiner_match = re.search(r"Вершин:\s+\d+\s*\(.*Точек Штейнера:\s*(\d+)\)", output)
    if not steiner_match:
        steiner_match = re.search(r"Points:\s+\d+\s*\(.*Steiner:\s*(\d+)\)", output)
    if steiner_match:
        data["steiner_count"] = int(steiner_match.group(1))

    return data


def build_markdown_table(results: List[Dict]) -> None:
    """Печатает таблицу результатов в формате Markdown."""
    header = [
        "Файл", "Терм.", "Нач. MST",
        "Длина баз.", "Шт. баз.",
        "Длина мод.", "Шт. мод.",
        "t баз., с", "t мод., с",
        "Ускорение (баз/мод)"
    ]
    separator = ["---"] * len(header)

    rows = [header, separator]

    for r in results:
        bt = r.get("basic_time")
        mt = r.get("modified_time")
        speedup = bt / mt if (bt is not None and mt is not None and mt > 0) else None

        row = [
            r.get("file", ""),
            str(r.get("terminals", "")),
            str(r.get("initial_mst", "")) if r.get("initial_mst") is not None else "—",
            str(r.get("basic_length", "")) if r.get("basic_length") is not None else "—",
            str(r.get("basic_steiner", "")) if r.get("basic_steiner") is not None else "—",
            str(r.get("modified_length", "")) if r.get("modified_length") is not None else "—",
            str(r.get("modified_steiner", "")) if r.get("modified_steiner") is not None else "—",
            f"{bt:.4f}" if bt is not None else "—",
            f"{mt:.4f}" if mt is not None else "—",
            f"{speedup:.2f}x" if speedup is not None else "—"
        ]
        rows.append(row)

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    for row in rows:
        line = "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        print(line)


def main():
    files = sorted(glob.glob("SMT-benchmarks/00*_0000.json"))
    pattern = re.compile(r"SMT-benchmarks/00(\d{2})_0000\.json")
    filtered = [f for f in files if (m := pattern.match(f)) and 5 <= int(m.group(1)) <= 30]

    assert filtered, "Не найдено подходящих файлов в папке SMT-benchmarks/"

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in filtered:
            print(f"Обрабатываю {f}...")
            basic = run_steiner(f, modified=False, out_dir=tmpdir)
            modified = run_steiner(f, modified=True, out_dir=tmpdir)

            if basic is None or modified is None:
                print(f"Пропуск {f} из-за ошибки.")
                continue

            m = pattern.match(f)
            term_count = int(m.group(1)) if m else 0

            initial_mst = basic.get("initial_mst") or modified.get("initial_mst")

            results.append({
                "file": os.path.basename(f),
                "terminals": term_count,
                "initial_mst": initial_mst,
                "basic_length": basic.get("final_length"),
                "basic_steiner": basic.get("steiner_count"),
                "modified_length": modified.get("final_length"),
                "modified_steiner": modified.get("steiner_count"),
                "basic_time": basic.get("time"),
                "modified_time": modified.get("time")
            })

    assert results, "Нет данных для построения таблицы."

    print("\n## Результаты бенчмарка\n")
    build_markdown_table(results)

    md_filename = "benchmark_results.md"
    with open(md_filename, "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print("# Результаты бенчмарка задачи Штейнера\n")
        build_markdown_table(results)
        sys.stdout = old_stdout
    print(f"\nMarkdown-отчёт сохранён в {md_filename}")


if __name__ == "__main__":
    main()