# ED 1-Bio HMC: Stratum Tectonicas Candidate Finder & Analyzer

Tools for finding and analyzing 1-bio High Metal Content (HMC) worlds in
Elite Dangerous that are candidates for **Stratum Tectonicas**.

The project consists of three scripts:

- **stratum-finder** -- scans Elite Dangerous journal files and identifies
  candidate bodies based on known Stratum criteria.
- **stratum-analyzer** -- takes finder output and runs statistical analysis
  plus generates plots to identify indicators that correlate with Stratum.
- **stratum-filter** -- extracts raw journal entries for specific body names,
  useful for deeper investigation of individual candidates.

## Installation

Requires **Python 3.12+**.

```bash
# Create a virtual environment and install
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
```

Dependencies: matplotlib, numpy, pandas, seaborn, icecream.

## Stratum Finder

Searches journal log files for bodies meeting **all five** Stratum Tectonicas
candidate criteria:

1. Exactly 1 biological signal
2. Thin atmosphere
3. High metal content body
4. Surface temperature >= 165 K
5. Parent star type in: TTS, F, M, T, Y, K, D, W, Ae, L, or None (barycenter)

Bodies whose genus turns out to be something other than Stratum or Bacterium
(e.g. Fungoida, or unknown/unscanned) are separated into a `_oddities` file
for later investigation.

### Usage

```bash
# Full scan (default journal directory)
stratum-finder -o finder_output.json

# Scan a specific directory with verbose output
stratum-finder -d /path/to/journals -o candidates.json -v

# Limit to recent journals (faster iteration)
stratum-finder --since 2026-02-01T000000 -o recent.json

# Date range
stratum-finder --since 2026-01-01T000000 --until 2026-01-31T235959 -o jan.json

# Generate a human-readable report alongside JSON
stratum-finder -o candidates.json --report summary.txt
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-d, --directory` | ED journal folder | Directory containing journal files |
| `-f, --file` | | Single log file to process (mutually exclusive with -d) |
| `-o, --output` | *required* | Output JSON file for candidates |
| `--report` | | Generate human-readable summary report |
| `--format` | json | Output format: `json` or `jsonl` |
| `--pattern` | `*.log` | File glob pattern |
| `--since` | | Only process journals with timestamp >= value |
| `--until` | | Only process journals with timestamp <= value |
| `-v, --verbose` | | Verbose output during processing |

### Output

The main output JSON contains one entry per candidate body with fields
including: system name, body name, surface temperature, gravity (in g),
pressure, mass, radius, atmosphere type and composition, materials, system age,
orbital parameters, and whether Stratum was confirmed (`HasStratum`).

## Stratum Analyzer

Reads finder output and performs statistical analysis to identify which
planetary properties correlate with Stratum Tectonicas presence.

### Analyses performed

- **Categorical features**: atmosphere type, volcanism, tidal lock, terraformable state
- **Numerical features**: mass, radius, gravity, temperature, pressure, orbital period
- **Atmosphere composition**: per-component presence rates and mean percentages
- **Body composition**: ice/rock/metal ratios
- **Materials**: surface material percentages

### Plots generated (with `--plots`)

- 2D scatter plots for all feature pair combinations
- Correlation matrices (Stratum vs non-Stratum)
- Distribution comparison histograms with box plots
- System age histogram (Stratum vs Bacterium by age bucket)
- Atmosphere composition indicator chart
- Gravitational binding energy proxy (GM^2/R) vs temperature

### Usage

```bash
# Basic analysis with console report
stratum-analyzer -i finder_output.json

# Save report and generate all plots
stratum-analyzer -i finder_output.json -o report.txt --plots

# Plots in a custom directory
stratum-analyzer -i finder_output.json --plots --plot-dir my_plots

# Filter to cold bodies only
stratum-analyzer -i finder_output.json --plots --max-temp 200 --plot-dir plots_cold

# Skip specific plot types
stratum-analyzer -i finder_output.json --plots --no-correlations --no-distributions
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-i, --input` | *required* | Input JSON from stratum-finder |
| `-o, --output` | | Save report to text file |
| `--json-output` | | Save raw analysis data as JSON |
| `--plots` | | Generate all plot types |
| `--plot-dir` | `stratum_plots` | Output directory for plots |
| `--max-temp` | | Only include bodies <= this temperature (K) |
| `--min-temp` | | Only include bodies >= this temperature (K) |
| `--no-correlations` | | Skip correlation matrices |
| `--no-distributions` | | Skip distribution plots |
| `--no-age-histogram` | | Skip age histogram |

## Stratum Log Filter

Extracts raw journal log entries for specific body names. Useful for pulling
the full journal data of known candidates for detailed investigation.

### Usage

```bash
# Filter logs for bodies listed in a file
stratum-filter -d /path/to/journals -i body_names.txt -o matches.json
```

The input list file uses the format `BodyName;0` or `BodyName;1` (0 = no
Stratum, 1 = has Stratum), one per line.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-d, --directory` | | Directory containing log files |
| `-f, --file` | | Single log file to process |
| `-i, --input-list` | *required* | File with body names and stratum info |
| `-o, --output` | `matches.json` | Output file |
| `--format` | json | Output format: `json` or `jsonl` |
| `--pattern` | `*.log` | File glob pattern |
| `--case-insensitive` | | Case-insensitive body name matching |

## Key Findings

Analysis of ~1200 candidate bodies revealed the following indicators for
Stratum Tectonicas:

| Rank | Indicator | Signal |
|---|---|---|
| 1 | **Atmosphere type** -- CarbonDioxideRich or Oxygen atmosphere | Very strong (100% Stratum rate) |
| 2 | **Oxygen in atmosphere composition** | Exclusive to Stratum |
| 3 | **Surface temperature < 200 K** | Strong (~52% Stratum rate vs 18% overall) |
| 4 | **CarbonDioxide atmosphere** | Moderate (~22% Stratum rate) |
| 5 | Surface gravity < 0.40 g | Weak signal |

Properties with **no significant signal**: pressure, system age, star type,
orbital parameters, body composition, surface materials.

Below 200 K, Stratum vs Bacterium appears essentially random -- temperature is
the primary discriminator, and once controlled for, no other property reliably
predicts genus.
