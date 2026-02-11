#!/usr/bin/env python3
"""
Stratum Analysis Script

This script analyzes Elite Dangerous scan data to identify patterns and indicators
that correlate with bodies having Stratum Tectonicas.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import statistics
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_scan_data(file_path: str) -> List[Dict[str, Any]]:
    """Load scan data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Error: Expected a list of scan entries, got {type(data)}")
            return []

        return data

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        return []
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return []


def analyze_categorical_feature(data: List[Dict[str, Any]], feature_name: str,
                               feature_path: List[str] = None) -> Dict[str, Any]:
    """
    Analyze a categorical feature for correlation with Stratum.

    Args:
        data: List of scan entries
        feature_name: Name of the feature for reporting
        feature_path: Path to the feature in nested dict (e.g., ['AtmosphereType'])
    """
    if feature_path is None:
        feature_path = [feature_name]

    stratum_counts = defaultdict(lambda: {'with_stratum': 0, 'without_stratum': 0, 'total': 0})

    for entry in data:
        has_stratum = entry.get('HasStratum', False)

        # Navigate to the feature value
        value = entry
        try:
            for key in feature_path:
                value = value.get(key, None)
                if value is None:
                    break
        except (AttributeError, TypeError):
            value = None

        if value is not None:
            value_str = str(value)
            stratum_counts[value_str]['total'] += 1
            if has_stratum:
                stratum_counts[value_str]['with_stratum'] += 1
            else:
                stratum_counts[value_str]['without_stratum'] += 1

    # Calculate percentages and sort by stratum likelihood
    results = []
    for value, counts in stratum_counts.items():
        if counts['total'] > 0:
            stratum_percentage = (counts['with_stratum'] / counts['total']) * 100
            results.append({
                'value': value,
                'with_stratum': counts['with_stratum'],
                'without_stratum': counts['without_stratum'],
                'total': counts['total'],
                'stratum_percentage': stratum_percentage
            })

    # Sort by stratum percentage (descending)
    results.sort(key=lambda x: x['stratum_percentage'], reverse=True)

    return {
        'feature_name': feature_name,
        'results': results,
        'total_entries': len(data)
    }


def analyze_numerical_feature(data: List[Dict[str, Any]], feature_name: str,
                            feature_path: List[str] = None) -> Dict[str, Any]:
    """
    Analyze a numerical feature for correlation with Stratum.
    """
    if feature_path is None:
        feature_path = [feature_name]

    with_stratum = []
    without_stratum = []

    for entry in data:
        has_stratum = entry.get('HasStratum', False)

        # Navigate to the feature value
        value = entry
        try:
            for key in feature_path:
                value = value.get(key, None)
                if value is None:
                    break
        except (AttributeError, TypeError):
            value = None

        if value is not None and isinstance(value, (int, float)):
            if has_stratum:
                with_stratum.append(value)
            else:
                without_stratum.append(value)

    result = {
        'feature_name': feature_name,
        'with_stratum_count': len(with_stratum),
        'without_stratum_count': len(without_stratum)
    }

    if with_stratum:
        result['with_stratum_stats'] = {
            'mean': statistics.mean(with_stratum),
            'median': statistics.median(with_stratum),
            'min': min(with_stratum),
            'max': max(with_stratum),
            'std_dev': statistics.stdev(with_stratum) if len(with_stratum) > 1 else 0
        }

    if without_stratum:
        result['without_stratum_stats'] = {
            'mean': statistics.mean(without_stratum),
            'median': statistics.median(without_stratum),
            'min': min(without_stratum),
            'max': max(without_stratum),
            'std_dev': statistics.stdev(without_stratum) if len(without_stratum) > 1 else 0
        }

    return result


def analyze_range_feature(data: List[Dict[str, Any]], feature_name: str,
                        feature_path: List[str] = None, min_max: tuple = None) -> Dict[str, Any]:
    """
    Analyze a range feature for correlation with Stratum.
    """
    if feature_path is None:
        feature_path = [feature_name]

    with_stratum = []
    without_stratum = []

    for entry in data:
        has_stratum = entry.get('HasStratum', False)

        # Navigate to the feature value
        value = entry
        try:
            for key in feature_path:
                value = value.get(key, None)
                if value is None:
                    break
        except (AttributeError, TypeError):
            value = None

        if value is not None and isinstance(value, (int, float)):
            if min_max[0] <= value <= min_max[1]:
                if has_stratum:
                    with_stratum.append(value)
                else:
                    without_stratum.append(value)

    result = {
        'feature_name': feature_name,
        'with_stratum_count': len(with_stratum),
        'without_stratum_count': len(without_stratum)
    }

    if with_stratum:
        result['with_stratum_stats'] = {
            'mean': statistics.mean(with_stratum),
            'median': statistics.median(with_stratum),
            'min': min(with_stratum),
            'max': max(with_stratum),
            'std_dev': statistics.stdev(with_stratum) if len(with_stratum) > 1 else 0
        }

    if without_stratum:
        result['without_stratum_stats'] = {
            'mean': statistics.mean(without_stratum),
            'median': statistics.median(without_stratum),
            'min': min(without_stratum),
            'max': max(without_stratum),
            'std_dev': statistics.stdev(without_stratum) if len(without_stratum) > 1 else 0
        }

    return result


def analyze_materials(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze material composition for correlation with Stratum."""
    material_analysis = defaultdict(lambda: {'with_stratum': [], 'without_stratum': []})

    for entry in data:
        has_stratum = entry.get('HasStratum', False)
        materials = entry.get('Materials', [])

        for material in materials:
            material_name = material.get('Name', '').lower()
            percentage = material.get('Percent', 0)

            if material_name and isinstance(percentage, (int, float)):
                if has_stratum:
                    material_analysis[material_name]['with_stratum'].append(percentage)
                else:
                    material_analysis[material_name]['without_stratum'].append(percentage)

    # Calculate statistics for each material
    results = []
    for material, data_dict in material_analysis.items():
        with_stratum = data_dict['with_stratum']
        without_stratum = data_dict['without_stratum']

        if len(with_stratum) > 0 and len(without_stratum) > 0:
            result = {
                'material': material,
                'with_stratum_count': len(with_stratum),
                'without_stratum_count': len(without_stratum),
                'with_stratum_mean': statistics.mean(with_stratum),
                'without_stratum_mean': statistics.mean(without_stratum),
                'difference': statistics.mean(with_stratum) - statistics.mean(without_stratum)
            }
            results.append(result)

    # Sort by absolute difference (materials with biggest differences)
    results.sort(key=lambda x: abs(x['difference']), reverse=True)

    return {'materials': results}


def analyze_composition(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Ice/Rock/Metal composition for correlation with Stratum."""
    composition_types = ['Ice', 'Rock', 'Metal']
    results = {}

    for comp_type in composition_types:
        results[comp_type.lower()] = analyze_numerical_feature(
            data, f"{comp_type} Composition", ['Composition', comp_type]
        )

    return results


def extract_plot_data(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract numerical data for plotting.

    Returns:
        Dictionary with structure: {
            'with_stratum': {'mass': [...], 'radius': [...], ...},
            'without_stratum': {'mass': [...], 'radius': [...], ...}
        }
    """
    plot_features = {
        'mass': 'MassEM',
        'radius': 'Radius',
        'gravity': 'SurfaceGravity',
        'temperature': 'SurfaceTemperature',
        'pressure': 'SurfacePressure'
    }

    plot_data = {
        'with_stratum': {feature: [] for feature in plot_features.keys()},
        'without_stratum': {feature: [] for feature in plot_features.keys()}
    }

    for entry in data:
        has_stratum = entry.get('HasStratum', False)
        category = 'with_stratum' if has_stratum else 'without_stratum'

        for feature_name, json_key in plot_features.items():
            value = entry.get(json_key)
            if value is not None and isinstance(value, (int, float)):
                plot_data[category][feature_name].append(value)

    return plot_data


def generate_2d_plots(plot_data: Dict[str, Dict[str, List[float]]],
                     output_dir: str = "stratum_plots") -> None:
    """
    Generate 2D scatter plots for all combinations of features.
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    features = list(plot_data['with_stratum'].keys())
    feature_labels = {
        'mass': 'Mass (Earth Masses)',
        'radius': 'Radius (m)',
        'gravity': 'Surface Gravity (m/s²)',
        'temperature': 'Surface Temperature (K)',
        'pressure': 'Surface Pressure (Pa)'
    }

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Generate all combinations of features (excluding self-pairs)
    combinations = list(itertools.combinations(features, 2))

    print(f"Generating {len(combinations)} 2D scatter plots...")

    for i, (feature_x, feature_y) in enumerate(combinations):
        print(f"  Creating plot {i+1}/{len(combinations)}: {feature_x} vs {feature_y}")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract data for both categories
        x_with = plot_data['with_stratum'][feature_x]
        y_with = plot_data['with_stratum'][feature_y]
        x_without = plot_data['without_stratum'][feature_x]
        y_without = plot_data['without_stratum'][feature_y]

        # Create scatter plots
        if x_without and y_without:
            scatter1 = ax.scatter(x_without, y_without,
                                c='lightcoral', alpha=0.6, s=30,
                                label=f'Without Stratum (n={len(x_without)})',
                                edgecolors='darkred', linewidth=0.3)

        if x_with and y_with:
            scatter2 = ax.scatter(x_with, y_with,
                                c='dodgerblue', alpha=0.7, s=40,
                                label=f'With Stratum (n={len(x_with)})',
                                edgecolors='darkblue', linewidth=0.5)

        # Customize the plot
        ax.set_xlabel(feature_labels[feature_x], fontsize=12)
        ax.set_ylabel(feature_labels[feature_y], fontsize=12)
        ax.set_title(f'Stratum Distribution: {feature_labels[feature_x]} vs {feature_labels[feature_y]}',
                    fontsize=14, fontweight='bold')

        # Add legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Use log scale for certain features that might have wide ranges
        if feature_x in ['radius', 'pressure']:
            ax.set_xscale('log')
        if feature_y in ['radius', 'pressure']:
            ax.set_yscale('log')

        # Add statistics text box
        stats_text = []
        if x_with and y_with:
            stats_text.append(f'With Stratum: n={len(x_with)}')
            stats_text.append(f'  {feature_x} mean: {np.mean(x_with):.2e}')
            stats_text.append(f'  {feature_y} mean: {np.mean(y_with):.2e}')

        if x_without and y_without:
            stats_text.append(f'Without Stratum: n={len(x_without)}')
            stats_text.append(f'  {feature_x} mean: {np.mean(x_without):.2e}')
            stats_text.append(f'  {feature_y} mean: {np.mean(y_without):.2e}')

        if stats_text:
            ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8))

        # Tight layout and save
        plt.tight_layout()

        # Create filename
        filename = f"{feature_x}_vs_{feature_y}.png"
        filepath = Path(output_dir) / filename

        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"All plots saved to '{output_dir}/' directory")


def generate_correlation_matrix(plot_data: Dict[str, Dict[str, List[float]]],
                               output_dir: str = "stratum_plots") -> None:
    """
    Generate correlation matrix heatmaps for both Stratum groups.
    """
    features = list(plot_data['with_stratum'].keys())
    feature_labels = {
        'mass': 'Mass',
        'radius': 'Radius',
        'gravity': 'Gravity',
        'temperature': 'Temperature',
        'pressure': 'Pressure'
    }

    Path(output_dir).mkdir(exist_ok=True)

    for category in ['with_stratum', 'without_stratum']:
        # Create DataFrame for correlation analysis
        data_dict = {}
        min_length = float('inf')

        # Find minimum length to ensure all arrays are same size
        for feature in features:
            if plot_data[category][feature]:
                min_length = min(min_length, len(plot_data[category][feature]))

        if min_length == float('inf') or min_length < 2:
            print(f"Skipping correlation matrix for {category} - insufficient data")
            continue

        # Truncate all arrays to minimum length and create DataFrame
        import pandas as pd
        for feature in features:
            if plot_data[category][feature]:
                data_dict[feature_labels[feature]] = plot_data[category][feature][:min_length]

        if len(data_dict) < 2:
            continue

        df = pd.DataFrame(data_dict)
        correlation_matrix = df.corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})

        title = f'Feature Correlation Matrix - {"With" if category == "with_stratum" else "Without"} Stratum'
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        filename = f"correlation_matrix_{category}.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Correlation matrix saved: {filename}")


def generate_distribution_plots(analyses: Dict[str, Any], plot_data: Dict[str, Dict[str, List[float]]],
                               output_dir: str = "stratum_plots", output_file: Optional[str] = None) -> None:
    """
    Generate distribution comparison plots for each feature.
    """
    Path(output_dir).mkdir(exist_ok=True)

    features = list(plot_data['with_stratum'].keys())
    feature_labels = {
        'mass': 'Mass (Earth Masses)',
        'radius': 'Radius (m)',
        'gravity': 'Surface Gravity (m/s²)',
        'temperature': 'Surface Temperature (K)',
        'pressure': 'Surface Pressure (Pa)'
    }

    print("Generating distribution comparison plots...")

    for feature in features:
        data_with = plot_data['with_stratum'][feature]
        data_without = plot_data['without_stratum'][feature]

        if not data_with and not data_without:
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Histogram comparison
        if data_without:
            ax1.hist(data_without, bins=30, alpha=0.7, color='lightcoral',
                    label=f'Without Stratum (n={len(data_without)})', density=True)
        if data_with:
            ax1.hist(data_with, bins=30, alpha=0.7, color='dodgerblue',
                    label=f'With Stratum (n={len(data_with)})', density=True)

        ax1.set_xlabel(feature_labels[feature])
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution Comparison: {feature_labels[feature]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot comparison
        box_data = []
        box_labels = []
        if data_without:
            box_data.append(data_without)
            box_labels.append('Without Stratum')
        if data_with:
            box_data.append(data_with)
            box_labels.append('With Stratum')

        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors = ['lightcoral', 'dodgerblue']
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax2.set_ylabel(feature_labels[feature])
        ax2.set_title('Box Plot Comparison')
        ax2.grid(True, alpha=0.3)

        # Use log scale for certain features
        if feature in ['radius', 'pressure']:
            ax1.set_yscale('log')
            ax2.set_yscale('log')

        plt.tight_layout()

        filename = f"distribution_{feature}.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    print("Distribution plots completed")


def generate_report(analyses: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Generate a comprehensive analysis report."""
    report_lines = []

    def add_line(text="", level=0):
        indent = "  " * level
        report_lines.append(f"{indent}{text}")

    add_line("=" * 80)
    add_line("ELITE DANGEROUS STRATUM TECTONICAS ANALYSIS REPORT")
    add_line("=" * 80)
    add_line()

    # Overview
    total_bodies = analyses['overview']['total_bodies']
    stratum_bodies = analyses['overview']['stratum_bodies']
    non_stratum_bodies = analyses['overview']['non_stratum_bodies']
    stratum_percentage = analyses['overview']['stratum_percentage']

    add_line("OVERVIEW")
    add_line("-" * 40)
    add_line(f"Total bodies analyzed: {total_bodies}")
    add_line(f"Bodies with Stratum: {stratum_bodies} ({stratum_percentage:.1f}%)")
    add_line(f"Bodies without Stratum: {non_stratum_bodies} ({100-stratum_percentage:.1f}%)")
    add_line()

    # Categorical features
    categorical_features = [
        'planet_class', 'atmosphere_type', 'volcanism', 'landable', 'tidal_lock', 'terraformable'
    ]

    for feature in categorical_features:
        if feature in analyses:
            analysis = analyses[feature]
            add_line(f"{analysis['feature_name'].upper()} ANALYSIS")
            add_line("-" * 40)

            for result in analysis['results'][:10]:  # Top 10
                add_line(f"{result['value']:.<30} {result['stratum_percentage']:>6.1f}% "
                        f"({result['with_stratum']}/{result['total']})")
            add_line()

    # Numerical features
    numerical_features = [
        'mass_em', 'radius', 'surface_gravity', 'surface_temperature',
        'surface_pressure', 'orbital_period'
    ]

    for feature in numerical_features:
        if feature in analyses:
            analysis = analyses[feature]
            add_line(f"{analysis['feature_name'].upper()} ANALYSIS")
            add_line("-" * 40)

            if 'with_stratum_stats' in analysis and 'without_stratum_stats' in analysis:
                ws_stats = analysis['with_stratum_stats']
                wos_stats = analysis['without_stratum_stats']

                add_line(f"With Stratum (n={analysis['with_stratum_count']}):")
                add_line(f"  Mean: {ws_stats['mean']:.6f}, Median: {ws_stats['median']:.6f}")
                add_line(f"  Range: {ws_stats['min']:.6f} - {ws_stats['max']:.6f}")
                add_line()
                add_line(f"Without Stratum (n={analysis['without_stratum_count']}):")
                add_line(f"  Mean: {wos_stats['mean']:.6f}, Median: {wos_stats['median']:.6f}")
                add_line(f"  Range: {wos_stats['min']:.6f} - {wos_stats['max']:.6f}")
                add_line()

                mean_diff = ws_stats['mean'] - wos_stats['mean']
                add_line(f"Difference in means: {mean_diff:.6f}")
            add_line()

    # Range features
    range_features = ['temperature_range']

    print(analyses['temperature_range'])

    for feature in range_features:
        if feature in analyses:
            analysis = analyses[feature]
            add_line(f"{analysis['feature_name'].upper()} ANALYSIS")
            add_line("-" * 40)

            if 'with_stratum_stats' in analysis and 'without_stratum_stats' in analysis:
                ws_stats = analysis['with_stratum_stats']
                wos_stats = analysis['without_stratum_stats']

                add_line(f"With Stratum (n={analysis['with_stratum_count']}):")
                add_line(f"  Mean: {ws_stats['mean']:.6f}, Median: {ws_stats['median']:.6f}")
                add_line(f"  Range: {ws_stats['min']:.6f} - {ws_stats['max']:.6f}")
                add_line()
                add_line(f"Without Stratum (n={analysis['without_stratum_count']}):")
                add_line(f"  Mean: {wos_stats['mean']:.6f}, Median: {wos_stats['median']:.6f}")
                add_line(f"  Range: {wos_stats['min']:.6f} - {wos_stats['max']:.6f}")
                add_line()

                mean_diff = ws_stats['mean'] - wos_stats['mean']
                add_line(f"Difference in means: {mean_diff:.6f}")
                percentage = 100.0 * analysis['with_stratum_count'] / (analysis['with_stratum_count'] + analysis['without_stratum_count'])
                add_line(f"Chance of Stratum: {percentage:.1f}%")
            add_line()

    # Materials analysis
    if 'materials' in analyses:
        add_line("MATERIALS ANALYSIS (Top 15 by difference)")
        add_line("-" * 40)
        for material in analyses['materials']['materials'][:15]:
            diff = material['difference']
            add_line(f"{material['material'].capitalize():.<20} {diff:>+7.2f}% "
                    f"({material['with_stratum_mean']:.2f}% vs {material['without_stratum_mean']:.2f}%)")
        add_line()

    # Composition analysis
    if 'composition' in analyses:
        add_line("COMPOSITION ANALYSIS")
        add_line("-" * 40)
        for comp_type in ['ice', 'rock', 'metal']:
            if comp_type in analyses['composition']:
                comp_analysis = analyses['composition'][comp_type]
                if 'with_stratum_stats' in comp_analysis and 'without_stratum_stats' in comp_analysis:
                    ws_mean = comp_analysis['with_stratum_stats']['mean']
                    wos_mean = comp_analysis['without_stratum_stats']['mean']
                    diff = ws_mean - wos_mean
                    add_line(f"{comp_type.capitalize()} composition: {diff:>+7.4f} "
                            f"({ws_mean:.4f} vs {wos_mean:.4f})")
        add_line()

    # Key findings
    add_line("KEY FINDINGS AND RECOMMENDATIONS")
    add_line("-" * 40)

    # Find strongest categorical indicators
    strong_indicators = []
    for feature in categorical_features:
        if feature in analyses:
            analysis = analyses[feature]
            for result in analysis['results']:
                if result['total'] >= 5 and result['stratum_percentage'] > 80:
                    strong_indicators.append(f"• {analysis['feature_name']}: {result['value']} "
                                           f"({result['stratum_percentage']:.1f}% Stratum rate)")

    if strong_indicators:
        add_line("Strong positive indicators (>80% Stratum rate with ≥5 samples):")
        for indicator in strong_indicators[:5]:
            add_line(indicator, 1)
        add_line()

    # Material recommendations
    if 'materials' in analyses:
        top_materials = analyses['materials']['materials'][:3]
        if top_materials:
            add_line("Materials most associated with Stratum:")
            for material in top_materials:
                if material['difference'] > 0:
                    add_line(f"• Higher {material['material']} content "
                           f"(+{material['difference']:.2f}% average)", 1)
            add_line()

    report_text = '\n'.join(report_lines)

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to '{output_file}'")
        except Exception as e:
            print(f"Error saving report to '{output_file}': {e}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Elite Dangerous scan data for Stratum Tectonicas indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i scan_results.json
  %(prog)s -i results.json -o stratum_analysis_report.txt
  %(prog)s -i data.json --json-output analysis.json
  %(prog)s -i results.json --plots --plot-dir my_plots
  %(prog)s -i data.json --plots --no-correlations

Plot Features:
  The --plots option generates 2D scatter plots for all combinations of:
  - Mass (Earth Masses) vs Radius, Gravity, Temperature, Pressure
  - Radius (m) vs Gravity, Temperature, Pressure
  - Gravity (m/s²) vs Temperature, Pressure
  - Temperature (K) vs Pressure
  Plus correlation matrices and distribution comparisons.
        """
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Input JSON file containing scan data')
    parser.add_argument('-o', '--output',
                       help='Output text file for the analysis report')
    parser.add_argument('--json-output',
                       help='Output JSON file for raw analysis data')
    parser.add_argument('--min-samples', type=int, default=3,
                       help='Minimum samples required for categorical analysis (default: 3)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate 2D scatter plots for all feature combinations')
    parser.add_argument('--plot-dir', default='stratum_plots',
                       help='Directory for plot output (default: stratum_plots)')
    parser.add_argument('--no-correlations', action='store_true',
                       help='Skip correlation matrix generation')
    parser.add_argument('--no-distributions', action='store_true',
                       help='Skip distribution comparison plots')

    args = parser.parse_args()

    # Load data
    print(f"Loading scan data from '{args.input}'...")
    scan_data = load_scan_data(args.input)

    if not scan_data:
        print("No scan data loaded. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(scan_data)} scan entries")

    # Basic overview
    stratum_count = sum(1 for entry in scan_data if entry.get('HasStratum', False))
    overview = {
        'total_bodies': len(scan_data),
        'stratum_bodies': stratum_count,
        'non_stratum_bodies': len(scan_data) - stratum_count,
        'stratum_percentage': (stratum_count / len(scan_data)) * 100 if scan_data else 0
    }

    print(f"Bodies with Stratum: {stratum_count}/{len(scan_data)} ({overview['stratum_percentage']:.1f}%)")
    print("\nPerforming analysis...")

    # Run all analyses
    analyses = {'overview': overview}

    # Categorical features
    analyses['planet_class'] = analyze_categorical_feature(scan_data, "Planet Class", ['PlanetClass'])
    analyses['atmosphere_type'] = analyze_categorical_feature(scan_data, "Atmosphere Type", ['AtmosphereType'])
    analyses['volcanism'] = analyze_categorical_feature(scan_data, "Volcanism", ['Volcanism'])
    analyses['landable'] = analyze_categorical_feature(scan_data, "Landable", ['Landable'])
    analyses['tidal_lock'] = analyze_categorical_feature(scan_data, "Tidal Lock", ['TidalLock'])
    analyses['terraformable'] = analyze_categorical_feature(scan_data, "Terraformable State", ['TerraformState'])

    # Numerical features
    analyses['mass_em'] = analyze_numerical_feature(scan_data, "Mass (Earth Masses)", ['MassEM'])
    analyses['radius'] = analyze_numerical_feature(scan_data, "Radius", ['Radius'])
    analyses['surface_gravity'] = analyze_numerical_feature(scan_data, "Surface Gravity", ['SurfaceGravity'])
    analyses['surface_temperature'] = analyze_numerical_feature(scan_data, "Surface Temperature", ['SurfaceTemperature'])
    analyses['surface_pressure'] = analyze_numerical_feature(scan_data, "Surface Pressure", ['SurfacePressure'])
    analyses['orbital_period'] = analyze_numerical_feature(scan_data, "Orbital Period", ['OrbitalPeriod'])

    # Range features
    analyses['temperature_range'] = analyze_range_feature(scan_data, "Temperature Range", ["SurfaceTemperature"], (165.0, 200.0))

    # Special analyses
    analyses['materials'] = analyze_materials(scan_data)
    analyses['composition'] = analyze_composition(scan_data)

    # Generate and display report
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    report = generate_report(analyses, args.output)

    if not args.output:
        print(report)

    # Save JSON output if requested
    if args.json_output:
        try:
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(analyses, f, indent=2, default=str)
            print(f"Raw analysis data saved to '{args.json_output}'")
        except Exception as e:
            print(f"Error saving JSON output to '{args.json_output}': {e}")

    # Generate plots if requested
    if args.plots:
        print("\nExtracting plot data...")
        plot_data = extract_plot_data(scan_data)

        # Check if we have sufficient data
        total_with = sum(len(values) for values in plot_data['with_stratum'].values())
        total_without = sum(len(values) for values in plot_data['without_stratum'].values())

        if total_with == 0 and total_without == 0:
            print("Warning: No numerical data available for plotting")
        else:
            print(f"Plot data extracted - With Stratum: {total_with//5} bodies, Without: {total_without//5} bodies")

            # Generate 2D scatter plots
            generate_2d_plots(plot_data, args.plot_dir)

            # Generate correlation matrices
            if not args.no_correlations:
                try:
                    generate_correlation_matrix(plot_data, args.plot_dir)
                except ImportError:
                    print("Warning: pandas not available, skipping correlation matrices")
                except Exception as e:
                    print(f"Warning: Could not generate correlation matrices: {e}")

            # Generate distribution plots
            if not args.no_distributions:
                generate_distribution_plots(analyses, plot_data, args.plot_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
