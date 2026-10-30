"""Create and execute two readable companion notebooks from the audited modules.

Called after both complete case modules have finished. Model results are not invented
here: the notebook reruns each case and reads the resulting saved evidence.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import asyncio
import argparse

PROJECT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.cache' / 'matplotlib'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(PROJECT / '.cache' / 'jupyter_runtime'))
os.environ.setdefault('JUPYTER_PATH', str(PROJECT / '.cache' / 'jupyter'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import nbformat
from nbclient import NotebookClient
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def build(case):
    name = 'Bäckerei' if case == 'bakery' else 'Getränkehandel'
    cells = [
        nbformat.v4.new_markdown_cell(f'# {name}: Nachfrage, Entscheidung und Wirtschaftlichkeit\n\n'
            '**Vollständig synthetischer Betrieb.** Die Modelle werden tatsächlich trainiert. '
            'Die wirtschaftlichen Ergebnisse sind Simulationen unter dokumentierten Annahmen; '
            'sie sind keine gemessenen Einsparungen eines realen Unternehmens.'),
        nbformat.v4.new_markdown_cell('## Ergebnis und Lesepfad\n\n'
            'Der zugehörige Bericht stellt die gemessenen Prognosefehler und die simulierte '
            'Wirtschaftlichkeit gegenüber der Vergleichsregel dar. Dieses Notebook führt den '
            'kompletten Fall aus und macht die gespeicherten Nachweise lesbar. '
            'Die tatsächlichen Werte erscheinen in den ausgeführten Ergebniszellen.'),
        nbformat.v4.new_markdown_cell('## Kontext und zentrale Annahmen\n\n'
            '- Die Generierung ist durch feste Seeds reproduzierbar.\n'
            '- Kalendermuster, Wettereinflüsse und Kostengrößen sind bewusst gesetzte Szenarioannahmen.\n'
            '- Verkäufe können durch Ausverkauf kleiner als die Nachfrage sein.\n'
            '- Latente Nachfrage dient allein dem Audit und der simulierten Politikbewertung.\n'
            '- Modellauswahl und Kalibrierung erfolgen zeitlich vor dem abschließenden Test.\n'
            '- Unsicherheitsintervalle beschreiben diesen synthetischen Verlauf und die jeweilige Methode; '
            'sie garantieren keine Übertragbarkeit auf einen echten Betrieb.'),
        nbformat.v4.new_markdown_cell('### 1. Umgebung und Projektpfad\n'
            'Python 3.12 und die Versionen aus `requirements.txt` bzw. `requirements.lock.txt` verwenden. '
            'Der Quellcode liegt in einem wiederverwendbaren Modul, damit Notebook und Kommandozeile '
            'dieselbe Berechnung ausführen.'),
        nbformat.v4.new_code_cell('from pathlib import Path\nimport sys, os, json, importlib\n'
            'import pandas as pd\nfrom IPython.display import display, Markdown, Image\n'
            'root = Path.cwd().resolve()\n'
            'if not (root / "run_demo.py").exists():\n'
            '    root = next(p for p in [root, *root.parents] if (p / "run_demo.py").exists())\n'
            'sys.path.insert(0, str(root)) if str(root) not in sys.path else None\n'
            'os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))\n'
            f'case = {case!r}\ncase_dir = root / "cases" / case\n'
            'print("Python:", sys.version.split()[0])\n'
            'print("Fall:", case)\n'),
        nbformat.v4.new_markdown_cell('### 2. Gesamten Fall ausführen\n\n'
            'Diese Zelle erzeugt die Daten, trainiert Vergleichsmodelle und Kandidaten, '
            'berechnet die Testprognosen und simuliert die Entscheidungen. '
            'Sie aktualisiert ausschließlich die Dateien dieses synthetischen Falls.'),
        nbformat.v4.new_code_cell('module = importlib.import_module(f"src.{case}")\n'
            'from threadpoolctl import threadpool_limits\n'
            'with threadpool_limits(limits=2):\n'
            '    result = module.run(case_dir, fast=False)\n'
            '    if case == "bakery":\n'
            '        replication = importlib.import_module("src.bakery_censored")\n'
            '        replication_result = replication.run(case_dir / "censored_replication", fast=False)\n'
            'print("Vollständiger Modell- und Simulationslauf abgeschlossen.")\n'
            'display(pd.DataFrame({"Ausgabefeld": list(result.keys())}))\n'),
        nbformat.v4.new_markdown_cell('## Daten\n\n'
            'Die öffentliche Beobachtungshistorie enthält die vom Betrieb tatsächlich erfassbaren '
            'synthetischen Verkäufe und Zustände. Separate Auditdateien enthalten die latente '
            'Nachfrage. Sie dürfen nicht zu einem vermeintlichen verfügbaren Trainingsmerkmal werden.'),
        nbformat.v4.new_code_cell('csv_files = sorted(case_dir.rglob("*.csv"))\n'
            'public = [p for p in csv_files if not any(t in str(p.relative_to(case_dir)).lower() '
            'for t in ["audit", "latent", "truth", "private"])]\n'
            'display(pd.DataFrame({"Beobachtungs-/Ergebnisdatei": [str(p.relative_to(case_dir)) for p in public]}))\n'
            'if public:\n'
            '    history = [p for p in public if any(t in p.name.lower() for t in ["observed", "history", "historical", "pos"])]\n'
            '    example = history[0] if history else public[0]\n'
            '    preview = pd.read_csv(example)\n'
            '    display(preview.head(6))\n'
            '    print("Beispieldatei:", str(example.relative_to(case_dir)), "Zeilen:", len(preview))\n'),
        nbformat.v4.new_markdown_cell('## Prognosefehler und wirtschaftliche Ergebnisse\n\n'
            'MAE misst den mittleren absoluten Mengenfehler; RMSE gewichtet große Fehler stärker. '
            'WAPE ist die Summe absoluter Fehler geteilt durch die gesamte Zielmenge. '
            'Bias zeigt systematische Über-/Unterschätzung. Modellgüte und Geldnutzen sind getrennte Ergebnisse. '
            'Die folgenden JSON-Strukturen bewahren die vollständigen Kennzahlen und Annahmen.'),
        nbformat.v4.new_code_cell('for filename in ["metrics.json", "economic_results.json"]:\n'
            '    location = case_dir / filename\n'
            '    if location.exists():\n'
            '        values = json.loads(location.read_text(encoding="utf-8"))\n'
            '        display(Markdown(f"### {filename}"))\n'
            '        print(json.dumps(values, ensure_ascii=False, indent=2)[:14000])\n'),
        *([
        nbformat.v4.new_markdown_cell('### Bäckerei: eigenständige Replikation mit Nachfragezensierung\n\n'
            'Nur beim Bäckereifall gibt es einen zusätzlichen, vorab festgelegten Versuch auf einem '
            'neuen Zufallspfad. Ein zensiertes Verteilungsmodell interpretiert ausverkaufte Mengen '
            'als untere Nachfragegrenzen. Der ursprüngliche negative Vergleich bleibt separat bestehen. '
            'Die Modelle dieser Replikation werden auf derselben frischen Trajektorie miteinander verglichen.'),
        nbformat.v4.new_code_cell('if case == "bakery":\n'
            '    replica_dir = case_dir / "censored_replication"\n'
            '    for filename in ["summary.json", "metrics.json", "economic_results.json"]:\n'
            '        location = replica_dir / filename\n'
            '        if location.exists():\n'
            '            display(Markdown(f"**Replikation: {filename}**"))\n'
            '            print(json.dumps(json.loads(location.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)[:14000])\n'
            'else:\n'
            '    print("Die getrennte Zensierungsreplikation gehört zur Bäckerei.")\n'),
        ] if case == 'bakery' else []),
        nbformat.v4.new_markdown_cell('### Visuelle Nachweise\n\n'
            'Die Abbildungen zeigen Prognosevergleich und Entscheidungsfolgen. '
            'Datumsbereich, Einheit und Vergleichsregel stehen in der jeweiligen Abbildung '
            'und werden im präsentierbaren Bericht erläutert.'),
        nbformat.v4.new_code_cell('charts = sorted(case_dir.rglob("*.png"))\n'
            'for chart in charts[:3]:\n'
            '    display(Markdown(f"**{chart.stem.replace(chr(95), chr(32))}**"))\n'
            '    display(Image(filename=str(chart), width=1000))\n'),
        nbformat.v4.new_markdown_cell('## Spezifikation, Grenzen und Transfer\n\n'
            'Die Modellkarte erläutert Features, Informationszeitpunkte, Datenaufteilung, '
            'Algorithmen und Fallgrenzen. Die allgemeine Methodik und Quellen stehen im Ordner `docs`.'),
        nbformat.v4.new_code_cell('cards = [p for p in case_dir.rglob("*.md") if '
            'any(t in p.name.lower() for t in ["model", "method", "card"])]\n'
            'if cards:\n'
            '    display(Markdown(cards[0].read_text(encoding="utf-8")))\n'
            'else:\n'
            '    print("Ausführliche Methodenbeschreibung: docs/METHODIK_UND_GRENZEN.md")\n'),
        nbformat.v4.new_markdown_cell('## Nächster Schritt in einem echten Betrieb\n\n'
            'Zuerst eine reale Entscheidung, ihren Zeitpunkt und verfügbare Exporte bestätigen. '
            'Dann Kosten, Ausverkauf, vorhandene Software, Liefer-/Produktionsrestriktionen und '
            'Mitwirkung klären. Das Modell auf tatsächlichen Daten neu trainieren und zeitlich '
            'gegen die echte bisherige Planung prüfen. Einen Geldnutzen erst mit umgesetzten '
            'Entscheidungen und geeigneten Vergleichseinheiten untersuchen.'),
    ]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.12.14'}}
    target = PROJECT / 'notebooks' / ('01_Baeckerei.ipynb' if case == 'bakery' else '02_Getraenkehandel.ipynb')
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(nb)
    client = NotebookClient(nb, kernel_name='demo-project', timeout=600,
                            resources={'metadata': {'path': str(PROJECT)}})
    client.execute()
    nbformat.validate(nb)
    nbformat.write(nb, target)
    print(f'Executed notebook: {target.name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'bakery', 'beverage'], default='all')
    args = parser.parse_args()
    kernel = PROJECT / '.cache' / 'jupyter' / 'kernels' / 'demo-project'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Demo project', 'language': 'python'
    }), encoding='utf-8')
    for selected_case in (['bakery', 'beverage'] if args.case == 'all' else [args.case]):
        build(selected_case)
