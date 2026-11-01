"""Offline-safe structural and numeric smoke test for ECCV16 tracking IoU fix.
Does not touch the dataset or results. If motmetrics is installed, test its real IoU distance too.
"""
from pathlib import Path
import ast
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = ('eccv16_tracking_eval.py', 'eccv16_tracking_qualitative.py')

def read_helper_and_check_calls(name):
    tree = ast.parse((HERE / name).read_text(encoding='utf-8'), filename=name)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'xyxy_to_xywh']
    assert len(helpers) == 1, f'{name}: conversion helper missing'
    namespace = {'np': np}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), name, 'exec'), namespace)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'iou_matrix':
            if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'distances':
                calls.append(node)
    assert len(calls) == 1, f'{name}: unexpected iou_matrix call count {len(calls)}'
    for arg in calls[0].args[:2]:
        assert isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'xyxy_to_xywh', f'{name}: unconverted boxes in iou_matrix'
    return namespace['xyxy_to_xywh']


def assert_close(actual, expected, message):
    assert np.allclose(actual, expected, atol=1e-12), f'{message}: {actual} != {expected}'


def main():
    converters = [read_helper_and_check_calls(name) for name in SCRIPTS]
    example = np.array([[100., 100., 110., 110.], [105., 105., 115., 115.]])
    for convert in converters:
        original = example.copy()
        xywh = convert(example)
        assert_close(xywh, [[100., 100., 10., 10.], [105., 105., 10., 10.]], 'overlap conversion')
        assert_close(example, original, 'input was mutated')
        assert convert([]).shape == (0, 4), 'empty conversion failed'
    try:
        if not hasattr(np, 'asfarray'):
            np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
        import motmetrics as mm
    except ImportError:
        print('PASS: both scripts convert boxes, geometry and non-mutation verified.')
        print('motmetrics not installed here; run again in the PhysioTrack-v2 conda environment for actual library check.')
        return
    a, b = example[:1], example[1:]
    close_b = np.array([[101., 101., 111., 111.]])
    for convert in converters:
        # A 5x5 overlap between two 10x10 boxes gives IoU=25/175 < 0.50.
        # Use max_iou=1 here to inspect the *distance*, without filtering.
        d = mm.distances.iou_matrix(convert(a), convert(b), max_iou=1.0)
        assert d.shape == (1, 1) and np.isfinite(d[0, 0]), f'unexpected unfiltered IoU distance: {d}'
        assert_close(d, [[1.0 - 25.0 / 175.0]], 'actual motmetrics IoU distance')

        # Under the real benchmark criterion IoU >= 0.50, this pair MUST be rejected.
        d_rejected = mm.distances.iou_matrix(convert(a), convert(b), max_iou=0.5)
        assert d_rejected.shape == (1, 1) and np.isnan(d_rejected[0, 0]), (
            f'low-overlap pair should be rejected: {d_rejected}'
        )

        # A 9x9 overlap gives IoU=81/119 > 0.50 and MUST be accepted.
        d_accepted = mm.distances.iou_matrix(convert(a), convert(close_b), max_iou=0.5)
        assert d_accepted.shape == (1, 1) and np.isfinite(d_accepted[0, 0]), (
            f'high-overlap pair should be accepted: {d_accepted}'
        )
        assert_close(d_accepted, [[1.0 - 81.0 / 119.0]], 'accepted IoU distance')

        d_same = mm.distances.iou_matrix(convert(a), convert(a), max_iou=0.5)
        assert_close(d_same, [[0.]], 'identical box must have zero distance')
        distant = [[300., 300., 310., 310.]]
        d_far = mm.distances.iou_matrix(convert(a), convert(distant), max_iou=0.5)
        assert np.isnan(d_far[0, 0]), 'disjoint box should not be accepted'
    print('PASS: both scripts convert xyxy -> xywh; actual motmetrics IoU tests passed.')

if __name__ == '__main__':
    main()
