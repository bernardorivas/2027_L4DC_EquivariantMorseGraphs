# Reduced attitude control

[Concise presentation](presentation.pdf). The left column uses the original thrust-axis dynamics and the right column the selected learned sphere model. Both use the three-node structural aggregate, including connecting cells and a recomputed Conley index.

| Panel | Original | Learned |
|---|---|---|
| Sphere preimages | [SVG](original-sphere.svg) | [SVG](learned-sphere.svg) |
| Interval cells | [SVG](original-interval.svg) | [SVG](learned-interval.svg) |
| Aggregate quotient graph | [SVG](original-quotient.svg) | [SVG](learned-quotient.svg) |
| Reconstructed sphere indices | [SVG](original-reconstructed.svg) | [SVG](learned-reconstructed.svg) |

The SVGs contain vector geometry and editable text without embedded images. Reproduce the presentation and both-controller PDF/SVG panels from `code/`:

```bash
python examples/example3/run.py periodic presentation --out outputs/example3-presentation
```

The [reproduction instructions](../periodic/README.md) describe the data, models, aggregate construction, and numerical evidence.
