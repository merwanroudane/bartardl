# ARDL-BART lecture

A professional, colour-themed Beamer lecture covering the theory of ARDL-BART,
the simulation and empirical evidence, and a hands-on tutorial with the
`bartardl` package.

- **Slides:** [`ardl_bart_lecture.pdf`](ardl_bart_lecture.pdf) (32 frames, 16:9)
- **Source:** [`ardl_bart_lecture.tex`](ardl_bart_lecture.tex)

## Compile

The deck reads the figures in [`../figures/`](../figures) (regenerate them with
`python docs/make_figures.py` if needed), then:

```bash
cd docs/lecture
pdflatex ardl_bart_lecture.tex
pdflatex ardl_bart_lecture.tex     # second pass for the frame counter
```

Requires a TeX distribution with `beamer`, `tcolorbox`, `listings`, `booktabs`,
`colortbl`, `tikz`, `pgfplots` and `fontawesome5` (all standard; MiKTeX installs
any missing package on the fly).

## Contents

1. Motivation & the ARDL set-up
2. BART theory — sum-of-trees, the three priors, back-fitting MCMC, grow/prune
   acceptance ratio, inclusion proportions
3. The competitors — LASSO, Elastic Net, Bayesian-network (SSVS)
4. Simulation evidence — Friedman DGPs, Tables 1–2, box plots
5. Macroeconomic application — Table 3 data, Table 4 ranking, GDP drivers
6. Applied lab with `bartardl` — install, quick start, simulation, horse-race,
   figures & the Parula palette
7. Guidance & take-aways
