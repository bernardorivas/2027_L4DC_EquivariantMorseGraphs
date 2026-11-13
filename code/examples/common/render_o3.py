"""Render the O(3) quotient flow, CMGDB cells, and Conley-Morse graphs."""

import sys

sys.dont_write_bytecode = True

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
WORK = tempfile.TemporaryDirectory(prefix='o3-morse-')

os.environ['MPLCONFIGDIR'] = os.path.join(WORK.name, 'mplconfig')

import CMGDB
from CMGDB.morse_graph_parser import MorseGraph
from CMGDB.PlotMorseSets import _merged_outline
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch, Polygon
import numpy as np

sys.path.insert(0, str(HERE))
from . import figure_style as style

FOLDER = 'o3'
LAYOUT = HERE / 'layout.json'
RESULTS = HERE.parent / 'example2/reference/manuscript'
AGGREGATE = RESULTS / 'explicit-aggregate'
from . import o3_geometry as exact

ROLES = ('A-', 'A+', 'C', 'O')
SINKS = ('A-', 'A+')


EDGES = {'O': ['C'], 'C': ['A-', 'A+']}
ORIGINAL_INDEX = {'A-': '(x-1, 0, 0, 0, 0)', 'A+': '(x-1, 0, 0, 0, 0)', 'C': '(0, x-1, 0, x-1, 0)',
                  'O': '(0, 0, 0, 0, x-1)'}
QUOTIENT_INDEX = {'A-': '(x-1, 0, 0)', 'A+': '(x-1, 0, 0)', 'C': '(0, x-1, 0)', 'O': '(0, 0, 0)'}

NAMES = {'A-': '<I>A</I><SUB>&#8722;</SUB>', 'A+': '<I>A</I><SUB>+</SUB>', 'C': '<I>C</I>', 'O': '<I>O</I>'}
canvas = (2.24, 1.35)
axes_box = (.175, .235, .795, .725)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'font.size': 9.2})
plt.rcParams.update(style.mpl_rc())



def phase_axes(fig, ax):
    fig.set_layout_engine(None)
    ax.set_position(axes_box)
    ax.set(xlim=(-1.5, 1.5), ylim=(-.05, 1.5), xlabel='$a$', ylabel='$r$', aspect='auto')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 1])
    ax.xaxis.labelpad = .4
    ax.yaxis.labelpad = .7
    ax.tick_params(length=2, width=.55, pad=1.3, labelsize=9.2)
    for spine in ax.spines.values():
        spine.set_linewidth(.5)


def paint_boxes(ax, boxes, fill, zorder=-2):
    outline = _merged_outline(boxes[:, :2], boxes[:, 2:])
    if outline is None:
        raise ValueError('Expected boxes on a common grid.')
    ax.add_patch(PathPatch(outline, facecolor=fill, edgecolor='none',
                           antialiased=False, zorder=zorder))


def draw_cmgdb_regions(ax, run, basins, ids):
    """Basin cells in the tint of their sink, O-to-C path cells in the tint of C.

    The .npz labels are native node ids of the run; ids maps a role to its id in
    the displayed graph, which sets the color.
    """
    native = {int(node): role for node, role in run['morse_nodes'].items()}
    assert native[run['path_source']] == 'O' and native[run['path_target']] == 'C'
    with np.load(basins) as data:
        boxes, labels, path_cells = (data[key] for key in ('boxes', 'singleton', 'path_cells'))

        included = np.zeros(len(boxes), dtype=bool)
        included[data['image_cells']] = True
    for node, role in native.items():
        if role in SINKS:
            paint_boxes(ax, boxes[included & (labels == node)], style.TINT[ids[role]])
    paint_boxes(ax, boxes[path_cells], style.TINT[ids['C']], zorder=-1)


def draw_trajectory(ax, a, r, position, linewidth):
    points = np.column_stack((a, r))
    display = ax.transData.transform(points)
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(display, axis=0), axis=1))]
    points_to_pixels = ax.figure.dpi / 72
    tip_at = position * length[-1]
    base_at = tip_at - 5.5 * points_to_pixels
    base, tip = np.column_stack([np.interp([base_at, tip_at], length, axis) for axis in display.T])
    tangent = tip - base
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    offset = 1.8 * points_to_pixels * normal
    inverse = ax.transData.inverted()

    before = np.vstack((points[length < base_at], inverse.transform(base)))
    after = np.vstack((inverse.transform(tip), points[length > tip_at]))
    for segment in (before, after):
        ax.plot(*segment.T, color=style.GRAY, linewidth=linewidth, solid_capstyle='butt')
    head = inverse.transform([base + offset, base - offset, tip])
    ax.add_patch(Polygon(head, facecolor=style.GRAY, edgecolor='none', zorder=3))


def plot_flow(path, ids):
    fig, ax = plt.subplots(figsize=canvas)
    phase_axes(fig, ax)
    time = np.linspace(-4, 5, 400)
    for sign in (-1, 1):
        for a, r, position in ((.39, .22, .64), (.34, .59, .56)):
            x, y = exact.flow(sign*a, r, time)
            draw_trajectory(ax, x, y, position, .75)
        angle = np.linspace(np.pi/2, 0 if sign > 0 else np.pi, 180)
        draw_trajectory(ax, np.cos(angle), np.sin(angle), .32, 1.08)
        x = np.linspace(0, sign, 180)
        draw_trajectory(ax, x, np.zeros_like(x), .48, .86)
    draw_trajectory(ax, np.zeros(180), np.linspace(0, 1, 180), .52, 1.08)
    offsets = {'A-': (-8, 4), 'A+': (8, 4), 'C': (0, 5), 'O': (10, 11)}
    for role, (a, r) in exact.ANCHORS.items():
        ax.scatter(a, r, s=10, color=style.MORSE[ids[role]], zorder=6)
        ax.annotate(f'${exact.MATH[role]}$', (a, r), xytext=offsets[role],
                    textcoords='offset points', ha='center', va='center',
                    color=style.INK, fontsize=9.7)
    style.save_panel(fig, path)
    plt.close(fig)


def load_graph(folder, prefix):
    """Parse a saved Morse graph and map each anchor role to its node id."""
    report = json.loads((folder / f'{prefix}-results.json').read_text())
    graph = MorseGraph.from_dot(folder / f'{prefix}-morse-graph.dot')
    by_id = {row['node']: row for row in report['nodes']}
    assert sorted(by_id) == list(graph.nodes) == list(range(len(by_id)))
    assert sorted([a, b] for a, targets in graph.edges.items() for b in targets) == sorted(report['edges'])
    for i in graph.nodes:
        assert graph.labels[i].split(' : ')[1] == '(' + ', '.join(by_id[i]['annotations']) + ')'
        assert len(by_id[i]['anchors']) == 1, (folder, i, by_id[i]['anchors'])
    ids = {by_id[i]['anchors'][0]: i for i in graph.nodes}
    assert sorted(ids) == sorted(ROLES), ids
    return graph, ids


def graph_source(graph, scale, names=None):
    """DOT of a Morse graph with the strokes for scale; names maps node ids to HTML labels of set names."""
    nodes = list(graph.nodes)
    dot = CMGDB.plot_derived_graph(graph, clist=[style.TINT[k] for k in nodes],
                                   margin='0.065, 0.04')
    for k in nodes:
        assert f'fillcolor="{style.TINT[k]}ff"' in dot.source, (k, dot.source)
    edge = style.graphviz_edge(scale)
    header = (f'digraph {{\n  graph [{style.GRAPH_FRAME}, nodesep=.14, ranksep=.27];\n'
              '  node [fontname="Times-Roman", fontsize=15];\n'
              f'  edge [color="{edge["color"]}", penwidth={edge["penwidth"]}, arrowsize=.85];')
    body = dot.source.replace('digraph {', header, 1)

    strokes = ''.join(
        f'{k} [' + ', '.join(f'{key}="{value}"' for key, value in style.graphviz_node(k, scale).items()) + '];\n'
        for k in nodes)
    for k, name in (names or {}).items():
        old = f'{k} [label="{k} : '
        assert body.count(old) == 1, (k, body)
        start = body.index(old) + len(old)
        end = body.index('"', start)
        body = body[:body.index(old)] + f'{k} [label=<{name} : {body[start:end]}>' + body[end + 1:]
    assert body.rstrip().endswith('}')
    return body.rstrip()[:-1] + strokes + '}\n'


def known_graph(ids, index):
    """The Conley-Morse graph with the given indices by role, on the CMGDB ids of the displayed graphs."""
    from CMGDB import DerivedGraph
    role = {i: r for r, i in ids.items()}
    nodes = sorted(role)
    edges = {i: sorted(ids[t] for t in EDGES.get(role[i], [])) for i in nodes}
    return DerivedGraph(nodes=nodes, edges=edges, labels={i: index[role[i]] for i in nodes})


def check_quotient(graph, quotient):
    """A saved CMGDB quotient graph has the known Hasse edges and quotient annotations."""
    assert {i: sorted(graph.edges.get(i, [])) for i in graph.nodes} == quotient.edges, graph.edges
    for i in graph.nodes:
        assert graph.labels[i] == f'{i} : {quotient.labels[i]}', (i, graph.labels[i])


def plot_morse(out, layout, case, folder, prefix, run, basins):
    graph, ids = load_graph(folder, prefix)
    fig, ax = CMGDB.PlotMorseSets(
        str(folder / f'{prefix}-morse-sets.csv'), proj_dims=[0, 1],
        clist=[style.MORSE[k] for k in graph.nodes],
        fig_w=canvas[0], fig_h=canvas[1], xlim=[-1.5, 1.5], ylim=[-.05, 1.5],
        xlabel='$a$', ylabel='$r$', fontsize=9.2, linewidth=.4,
        rasterize=False, show=False,
    )
    phase_axes(fig, ax)
    draw_cmgdb_regions(ax, run, basins, ids)
    style.save_panel(fig, out / f'{case}-cells.pdf')
    plt.close(fig)
    printed = style.printed_width_bp(layout, f'{FOLDER}/{case}-graph.pdf')
    return style.render_graph(lambda scale: graph_source(graph, scale), out / f'{case}-graph.pdf', printed)


def main():
    global RESULTS, AGGREGATE
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True, help=f'figures directory; writes OUT/{FOLDER}')
    parser.add_argument('--layout', type=Path, default=LAYOUT)
    parser.add_argument('--data', type=Path, default=RESULTS)
    args = parser.parse_args()
    out = args.out.resolve() / FOLDER
    layout = json.loads(args.layout.read_text())
    RESULTS = args.data
    AGGREGATE = RESULTS / 'explicit-aggregate'

    summary = json.loads((RESULTS / 'summary.json').read_text())
    runs = summary['runs']
    assert 'matched_grid' in summary
    aggregate = json.loads((AGGREGATE / 'aggregate-results.json').read_text())
    assert aggregate['source_result_sha256'] == runs['explicit']['inputs_sha256']['native-results.json']
    fitted = RESULTS / runs['fitted']['source_run']
    explicit_graph, explicit_ids = load_graph(AGGREGATE, 'aggregate')
    fitted_graph, fitted_ids = load_graph(fitted, 'native')

    assert explicit_ids == fitted_ids == {'A-': 0, 'A+': 1, 'C': 2, 'O': 3}, (explicit_ids, fitted_ids)
    quotient = known_graph(explicit_ids, QUOTIENT_INDEX)
    check_quotient(explicit_graph, quotient)
    check_quotient(fitted_graph, quotient)
    original = known_graph(explicit_ids, ORIGINAL_INDEX)
    names = {i: NAMES[r] for r, i in explicit_ids.items()}

    with WORK:
        out.mkdir(parents=True, exist_ok=True)
        plot_flow(out / 'quotient-flow.pdf', explicit_ids)
        printed = style.printed_width_bp(layout, f'{FOLDER}/original-graph.pdf')
        scale = style.render_graph(lambda s: graph_source(original, s, names), out / 'original-graph.pdf', printed)
        print(f'original-graph.pdf: {style.pdf_size_bp(out / "original-graph.pdf")[0]:.1f} bp wide, scale {scale:.4f}')
        scales = [plot_morse(out, layout, 'explicit', AGGREGATE, 'aggregate', runs['explicit'],
                             RESULTS / 'explicit-basins.npz'),
                  plot_morse(out, layout, 'learned', fitted, 'native', runs['fitted'],
                             RESULTS / 'fitted-basins.npz')]
    print(f'{out} (graph print scale {scales[0]:.4f}, {scales[1]:.4f})')


if __name__ == '__main__':
    main()
