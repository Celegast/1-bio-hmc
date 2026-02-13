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


def analyze_atmosphere_composition(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze AtmosphereComposition component percentages for correlation with Stratum.

    For each component (SO2, CO2, Oxygen, etc.) calculates:
    - Mean percentage in Stratum vs non-Stratum bodies (when the component is present)
    - Presence rate (fraction of bodies that have the component at all)
    """
    comp_data: Dict[str, Dict] = defaultdict(
        lambda: {'with_stratum': [], 'without_stratum': [],
                 'presence_stratum': 0, 'presence_other': 0}
    )
    total_stratum = sum(1 for d in data if d.get('HasStratum'))
    total_other   = len(data) - total_stratum

    for entry in data:
        has_stratum = entry.get('HasStratum', False)
        for item in (entry.get('AtmosphereComposition') or []):
            name = item.get('Name') or item.get('name') or item.get('Component', '?')
            pct  = item.get('Percent') or item.get('percent') or 0
            if not isinstance(pct, (int, float)):
                continue
            bucket = comp_data[name]
            if has_stratum:
                bucket['with_stratum'].append(pct)
                bucket['presence_stratum'] += 1
            else:
                bucket['without_stratum'].append(pct)
                bucket['presence_other'] += 1

    results = []
    for comp, d in comp_data.items():
        ws = d['with_stratum']
        wo = d['without_stratum']
        presence_s = d['presence_stratum'] / total_stratum  * 100 if total_stratum else 0
        presence_o = d['presence_other']   / total_other    * 100 if total_other   else 0
        results.append({
            'component':           comp,
            'with_stratum_mean':   statistics.mean(ws) if ws else 0.0,
            'without_stratum_mean':statistics.mean(wo) if wo else 0.0,
            'mean_difference':     (statistics.mean(ws) if ws else 0.0) -
                                   (statistics.mean(wo) if wo else 0.0),
            'with_stratum_count':  len(ws),
            'without_stratum_count': len(wo),
            'presence_stratum_pct':  presence_s,
            'presence_other_pct':    presence_o,
            'presence_difference':   presence_s - presence_o,
        })

    results.sort(key=lambda x: abs(x['presence_difference']), reverse=True)
    return {
        'components': results,
        'total_stratum': total_stratum,
        'total_other':   total_other,
    }


def generate_atmosphere_composition_plot(atm_analysis: Dict[str, Any],
                                         output_dir: str = "stratum_plots") -> None:
    """
    Generate a two-panel bar chart for atmosphere composition indicators:
      Top panel:    presence rate (% of bodies containing each component)
      Bottom panel: mean % when the component is present
    """
    Path(output_dir).mkdir(exist_ok=True)

    components = [r['component'] for r in atm_analysis['components']]
    if not components:
        return

    presence_s = [r['presence_stratum_pct']   for r in atm_analysis['components']]
    presence_o = [r['presence_other_pct']      for r in atm_analysis['components']]
    mean_s     = [r['with_stratum_mean']        for r in atm_analysis['components']]
    mean_o     = [r['without_stratum_mean']     for r in atm_analysis['components']]

    x = np.arange(len(components))
    bar_width = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    # --- Panel 1: presence rate ---
    b1s = ax1.bar(x - bar_width / 2, presence_s, bar_width,
                  color='dodgerblue', alpha=0.85, edgecolor='darkblue', linewidth=0.5,
                  label='With Stratum')
    b1o = ax1.bar(x + bar_width / 2, presence_o, bar_width,
                  color='lightcoral', alpha=0.85, edgecolor='darkred', linewidth=0.5,
                  label='Without Stratum')
    for bar, val in list(zip(b1s, presence_s)) + list(zip(b1o, presence_o)):
        if val:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    ax1.set_ylabel('Bodies with component (%)')
    ax1.set_title('Atmosphere Composition Indicators: Presence Rate', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3)

    # --- Panel 2: mean % when present ---
    b2s = ax2.bar(x - bar_width / 2, mean_s, bar_width,
                  color='dodgerblue', alpha=0.85, edgecolor='darkblue', linewidth=0.5,
                  label='With Stratum')
    b2o = ax2.bar(x + bar_width / 2, mean_o, bar_width,
                  color='lightcoral', alpha=0.85, edgecolor='darkred', linewidth=0.5,
                  label='Without Stratum')
    for bar, val in list(zip(b2s, mean_s)) + list(zip(b2o, mean_o)):
        if val:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    ax2.set_ylabel('Mean component % (when present)')
    ax2.set_title('Atmosphere Composition Indicators: Mean Percentage', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / "atmosphere_composition.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Atmosphere composition plot saved: atmosphere_composition.png")


def generate_binding_energy_plot(data: List[Dict[str, Any]],
                                 output_dir: str = "stratum_plots") -> None:
    """
    Scatter plot of GM^2/R (gravitational binding energy proxy) vs Surface Temperature,
    colour-coded by Stratum / Bacterium.
    """
    GRAV_CONST = 6.67430e-11       # m^3 kg^-1 s^-2
    EARTH_MASS = 5.972168e24       # kg

    Path(output_dir).mkdir(exist_ok=True)

    stratum_x, stratum_y = [], []
    other_x, other_y = [], []

    for entry in data:
        mass_em = entry.get('MassEM')
        radius  = entry.get('Radius')
        temp    = entry.get('SurfaceTemperature')
        if mass_em is None or radius is None or temp is None:
            continue
        if radius == 0:
            continue

        mass_kg = mass_em * EARTH_MASS
        gm2_r = GRAV_CONST * mass_kg ** 2 / radius

        if entry.get('HasStratum'):
            stratum_x.append(gm2_r)
            stratum_y.append(temp)
        else:
            other_x.append(gm2_r)
            other_y.append(temp)

    if not stratum_x and not other_x:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    if other_x:
        ax.scatter(other_x, other_y, alpha=0.5, s=18, color='lightcoral',
                   edgecolors='darkred', linewidths=0.3,
                   label=f'Without Stratum (n={len(other_x)})')
    if stratum_x:
        ax.scatter(stratum_x, stratum_y, alpha=0.7, s=22, color='dodgerblue',
                   edgecolors='darkblue', linewidths=0.3,
                   label=f'With Stratum (n={len(stratum_x)})')

    ax.set_xlabel(r'$G \cdot M^2 \,/\, R$  (J)', fontsize=11)
    ax.set_ylabel('Surface Temperature (K)', fontsize=11)
    ax.set_title(r'Gravitational Binding Energy Proxy ($G M^2 / R$) vs Temperature',
                 fontweight='bold')
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / "binding_energy_vs_temp.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Binding-energy plot saved: binding_energy_vs_temp.png")


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
        'gravity': 'Surface Gravity (g)',
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
        'gravity': 'Surface Gravity (g)',
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

        # Shared bin edges so both series are comparable
        all_vals = list(data_without or []) + list(data_with or [])
        bin_range = (min(all_vals), max(all_vals)) if all_vals else None
        n_bins = 30
        _, bin_edges = np.histogram(all_vals, bins=n_bins, range=bin_range)

        n_out = np.histogram(data_without, bins=bin_edges)[0] if data_without else np.zeros(n_bins)
        n_in  = np.histogram(data_with,    bins=bin_edges)[0] if data_with    else np.zeros(n_bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = (bin_edges[1] - bin_edges[0]) * 0.45

        bars_out = ax1.bar(bin_centers - bar_width / 2, n_out, bar_width,
                           color='lightcoral', alpha=0.85, edgecolor='darkred', linewidth=0.4,
                           label=f'Without Stratum (n={len(data_without or [])})')
        bars_in  = ax1.bar(bin_centers + bar_width / 2, n_in,  bar_width,
                           color='dodgerblue', alpha=0.85, edgecolor='darkblue', linewidth=0.4,
                           label=f'With Stratum (n={len(data_with or [])})')

        for bar, count in zip(bars_out, n_out):
            if count:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         str(int(count)), ha='center', va='bottom', fontsize=5, color='darkred')
        for bar, count in zip(bars_in, n_in):
            if count:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         str(int(count)), ha='center', va='bottom', fontsize=5, color='darkblue')

        ax1.set_xlabel(feature_labels[feature])
        ax1.set_ylabel('Count')
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
            bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            colors = ['lightcoral', 'dodgerblue']
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax2.set_ylabel(feature_labels[feature])
        ax2.set_title('Box Plot Comparison')
        ax2.grid(True, alpha=0.3)

        # Use log scale on box plot y-axis for wide-range features
        if feature in ['radius', 'pressure']:
            ax2.set_yscale('log')

        plt.tight_layout()

        filename = f"distribution_{feature}.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    print("Distribution plots completed")


def generate_age_histogram(data: List[Dict[str, Any]],
                           output_dir: str = "stratum_plots",
                           bin_width_my: int = 500) -> None:
    """
    Generate a grouped bar chart of Stratum vs Bacterium counts by system age.

    Bodies are binned into groups of bin_width_my million years. Each bin
    shows two bars side by side: one for Stratum, one for Bacterium (all
    other confirmed genera are stacked into an 'Other' segment on the
    Bacterium bar so the chart stays readable).
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Collect counts per bin and genus
    stratum_bins: dict = {}
    bacterium_bins: dict = {}
    other_bins: dict = {}

    for entry in data:
        age = entry.get('system_age_my')
        if age is None:
            continue
        bin_label = int(age // bin_width_my) * bin_width_my
        genus = entry.get('genus', 'Unknown')
        has_stratum = entry.get('HasStratum', False)

        if has_stratum:
            stratum_bins[bin_label] = stratum_bins.get(bin_label, 0) + 1
        elif genus.lower() == 'bacterium':
            bacterium_bins[bin_label] = bacterium_bins.get(bin_label, 0) + 1
        else:
            other_bins[bin_label] = other_bins.get(bin_label, 0) + 1

    all_bins = sorted(set(stratum_bins) | set(bacterium_bins) | set(other_bins))
    if not all_bins:
        print("Warning: No age data available for age histogram")
        return

    group_spacing = 0.72
    bar_width = group_spacing * 0.45  # two bars fill 90% of spacing → small inter-group gap
    x = np.arange(len(all_bins)) * group_spacing

    stratum_counts   = [stratum_bins.get(b, 0)  for b in all_bins]
    bacterium_counts = [bacterium_bins.get(b, 0) for b in all_bins]
    other_counts     = [other_bins.get(b, 0)     for b in all_bins]

    fig, ax = plt.subplots(figsize=(14, 5))

    bars_s = ax.bar(x - bar_width / 2, stratum_counts, bar_width,
                    label='Stratum Tectonicas', color='dodgerblue', alpha=0.85,
                    edgecolor='darkblue', linewidth=0.4)
    bars_b = ax.bar(x + bar_width / 2, bacterium_counts, bar_width,
                    label='Bacterium', color='lightcoral', alpha=0.85,
                    edgecolor='darkred', linewidth=0.4)
    # Stack 'Other' on top of Bacterium bars
    if any(other_counts):
        ax.bar(x + bar_width / 2, other_counts, bar_width,
               bottom=bacterium_counts,
               label='Other genus', color='wheat', alpha=0.85,
               edgecolor='goldenrod', linewidth=0.4)

    ax.set_xlabel('System Age (MY)', fontsize=10)
    ax.set_ylabel('Candidates', fontsize=10)
    ax.set_title('Stratum vs Bacterium Candidates by System Age (500 MY bins)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{b/1000:.1f}k' if b >= 1000 else str(b) for b in all_bins],
        rotation=60, ha='right', fontsize=7
    )
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Annotate count above each bar (skip zeros)
    for bar, count in zip(bars_s, stratum_counts):
        if count:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(count), ha='center', va='bottom', fontsize=6, color='darkblue')
    for bar, count, other in zip(bars_b, bacterium_counts, other_counts):
        total = count + other
        if total:
            ax.text(bar.get_x() + bar.get_width() / 2, total + 0.2,
                    str(total), ha='center', va='bottom', fontsize=6, color='darkred')

    plt.tight_layout()
    filepath = Path(output_dir) / "age_histogram.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Age histogram saved: age_histogram.png")


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

    # Atmosphere composition analysis
    if 'atmosphere_composition' in analyses:
        atm = analyses['atmosphere_composition']
        add_line("ATMOSPHERE COMPOSITION ANALYSIS")
        add_line("-" * 40)
        add_line(f"{'Component':<22} {'Str pres%':>10} {'Bac pres%':>10} "
                 f"{'Str mean%':>10} {'Bac mean%':>10} {'Pres diff':>10}")
        for r in atm['components']:
            add_line(f"  {r['component']:<20} "
                     f"{r['presence_stratum_pct']:>9.1f}% "
                     f"{r['presence_other_pct']:>9.1f}% "
                     f"{r['with_stratum_mean']:>9.1f}% "
                     f"{r['without_stratum_mean']:>9.1f}% "
                     f"{r['presence_difference']:>+9.1f}%")
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
                    strong_indicators.append(f"- {analysis['feature_name']}: {result['value']} "
                                           f"({result['stratum_percentage']:.1f}% Stratum rate)")

    if strong_indicators:
        add_line("Strong positive indicators (>80% Stratum rate with >=5 samples):")
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
                    add_line(f"- Higher {material['material']} content "
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
  - Gravity (g) vs Temperature, Pressure
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
    parser.add_argument('--no-age-histogram', action='store_true',
                       help='Skip Stratum/Bacterium by system age bar chart')

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
    analyses['atmosphere_composition'] = analyze_atmosphere_composition(scan_data)

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

        # Age histogram uses raw scan_data directly (not plot_data)
        if not args.no_age_histogram:
            generate_age_histogram(scan_data, args.plot_dir)

        # Atmosphere composition plot
        if 'atmosphere_composition' in analyses:
            generate_atmosphere_composition_plot(analyses['atmosphere_composition'], args.plot_dir)

        # Binding energy proxy vs temperature
        generate_binding_energy_plot(scan_data, args.plot_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
