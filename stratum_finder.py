"""
Stratum Tectonicas Candidate Finder

This script intelligently searches Elite Dangerous log files for bodies that are
likely candidates for Stratum Tectonicas based on biological signals and planetary criteria.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict


# Criteria for Stratum Tectonicas
STRATUM_CRITERIA = {
    'atmosphere': ['thin'],  # atmosphere must contain 'thin'
    'planet_class': ['High metal content body'],
    'min_temperature': 165.0,  # Kelvin
    'parent_star_types': ['TTS', 'F', 'M', 'T', 'Y', 'K', 'D', 'W', 'Ae', 'L', 'None'] # ToDo: 'None' is for barycenters
}

# Criteria for Bacterium (for reference)
BACTERIUM_CRITERIA = {
    'atmosphere': ['thin']
}


class SystemData:
    """Store information about a star system during log processing."""

    def __init__(self):
        self.system_name: Optional[str] = None
        self.system_address: Optional[int] = None
        self.bodies: Dict[int, Dict[str, Any]] = {}  # BodyID -> body data
        self.biological_signals: Dict[int, int] = {}  # BodyID -> signal count
        self.genus_info: Dict[int, str] = {}  # BodyID -> genus name (Stratum, Bacterium, etc.)
        self.stars: Dict[int, Dict[str, Any]] = {}  # BodyID -> star data

    def clear(self):
        """Clear all stored data."""
        self.system_name = None
        self.system_address = None
        self.bodies.clear()
        self.biological_signals.clear()
        self.genus_info.clear()
        self.stars.clear()


def get_parent_star_type(body: Dict[str, Any], system_data: SystemData) -> Optional[str]:
    """
    Extract the parent star type from a body's parent structure.

    Args:
        body: Body scan data
        system_data: System data containing star information

    Returns:
        Star type string or None
    """
    parents = body.get('Parents', [])

    for parent_dict in parents:
        if 'Star' in parent_dict:
            star_id = parent_dict['Star']
            if star_id in system_data.stars:
                star = system_data.stars[star_id]
                star_type = star.get('StarType', '')
                return star_type

    return 'None'


def extract_star_type_prefix(star_type: str) -> str:
    """
    Extract the prefix from a star type (e.g., 'F' from 'F5').

    Args:
        star_type: Full star type string

    Returns:
        Star type prefix
    """
    if not star_type:
        return ''

    # Handle special cases
    if star_type.startswith('TTS'):
        return 'TTS'
    if star_type.startswith('Ae'):
        return 'Ae'
    if star_type.startswith('None'):
        return 'None'

    # Return first character for standard types
    return star_type[0] if star_type else ''


def check_atmosphere_criteria(atmosphere: str, required: List[str]) -> bool:
    """
    Check if atmosphere meets criteria.

    Args:
        atmosphere: Atmosphere description string
        required: List of required atmosphere keywords

    Returns:
        True if any required keyword is in the atmosphere string
    """
    if not atmosphere:
        return False

    atmosphere_lower = atmosphere.lower()
    return any(keyword.lower() in atmosphere_lower for keyword in required)


def evaluate_stratum_candidate(body: Dict[str, Any], system_data: SystemData,
                               bio_count: int) -> Dict[str, Any]:
    """
    Evaluate if a body is a candidate for Stratum Tectonicas.
    All five criteria must be met.

    Returns:
        Dictionary with evaluation results
    """
    result = {
        'is_candidate': False,
        'criteria_met': {},
        'criteria_failed': {},
        'biological_count': bio_count
    }

    # Check biological count (must be exactly 1)
    if bio_count != 1:
        result['criteria_failed']['biological_count'] = f'Expected 1, got {bio_count}'
        return result

    result['criteria_met']['biological_count'] = '1 (exact match)'

    # Check atmosphere
    atmosphere = body.get('Atmosphere', '')
    if check_atmosphere_criteria(atmosphere, STRATUM_CRITERIA['atmosphere']):
        result['criteria_met']['atmosphere'] = f'Thin: {atmosphere}'
    else:
        result['criteria_failed']['atmosphere'] = f'Not thin: {atmosphere}'
        return result

    # Check planet class
    planet_class = body.get('PlanetClass', '')
    if planet_class in STRATUM_CRITERIA['planet_class']:
        result['criteria_met']['planet_class'] = planet_class
    else:
        result['criteria_failed']['planet_class'] = f'{planet_class} (expected High metal content body)'
        return result

    # Check temperature
    temperature = body.get('SurfaceTemperature')
    if temperature is not None:
        if temperature >= STRATUM_CRITERIA['min_temperature']:
            result['criteria_met']['temperature'] = f'{temperature:.1f}K (>= {STRATUM_CRITERIA["min_temperature"]}K)'
        else:
            result['criteria_failed']['temperature'] = f'{temperature:.1f}K (< {STRATUM_CRITERIA["min_temperature"]}K)'
            return result
    else:
        result['criteria_failed']['temperature'] = 'Temperature data not available'
        return result

    # Check parent star type
    parent_star_type = get_parent_star_type(body, system_data)
    #print(f"  check parent star type '{parent_star_type}'")
    if parent_star_type:
        star_prefix = extract_star_type_prefix(parent_star_type)
        #print(f"  check star prefix '{star_prefix}'")
        if star_prefix in STRATUM_CRITERIA['parent_star_types']:
            result['criteria_met']['parent_star'] = f'{parent_star_type} (type {star_prefix})'
        else:
            result['criteria_failed']['parent_star'] = f'{parent_star_type} (type {star_prefix} not in allowed list)'
            return result
    else:
        result['criteria_failed']['parent_star'] = 'Parent star type not available'
        return result

    # All criteria met - this is a valid candidate
    result['is_candidate'] = True

    return result


def extract_timestamp_from_filename(filename: str) -> str:
    """
    Extract timestamp from Elite Dangerous journal filename.
    Format: Journal.2025-10-21T213318.01.log
    Returns sortable timestamp string or filename if pattern doesn't match.
    """
    import re

    # Pattern: Journal.YYYY-MM-DDTHHMMSS.NN.log
    pattern = r'Journal\.(\d{4}-\d{2}-\d{2}T\d{6})\.\d{2}\.log'
    match = re.search(pattern, filename)

    if match:
        return match.group(1)  # Returns: 2025-10-21T213318

    # Fallback to filename for sorting
    return filename


def process_log_file(file_path: str, system_data: SystemData,
                     candidates: List[Dict[str, Any]]) -> int:
    """
    Process a single log file and find Stratum candidates.
    System data is preserved across files.

    Returns:
        Number of new candidates found in this file
    """
    new_candidates_count = 0
    current_system = system_data.system_name

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_number = 0

            for line in f:
                line_number += 1
                line = line.strip()

                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    event = entry.get('event', '')

                    # Track system changes
                    if event in ['FSDJump', 'Location', 'CarrierJump']:
                        new_system = entry.get('StarSystem')
                        if new_system != current_system:
                            system_data.clear()
                            current_system = new_system
                            system_data.system_name = new_system
                            system_data.system_address = entry.get('SystemAddress')

                    # Collect star data
                    elif event == 'Scan':
                        body_id = entry.get('BodyID')
                        body_name = entry.get('BodyName', '')

                        # Check if it's a star
                        if 'StarType' in entry:
                            if body_id is not None:
                                system_data.stars[body_id] = entry

                        # Check if it's a planet
                        elif 'PlanetClass' in entry:
                            if body_id is not None:
                                system_data.bodies[body_id] = entry

                    # Collect biological signal data
                    elif event == 'FSSBodySignals':
                        body_id = entry.get('BodyID')
                        signals = entry.get('Signals', [])

                        bio_count = 0
                        geo_count = 0
                        for signal in signals:
                            signal_type = signal.get('Type', '')
                            if 'Biological' in signal_type:
                                bio_count = signal.get('Count', 0)
                            if 'Geological' in signal_type:
                                geo_count = signal.get('Count', 0)

                        if body_id is not None and bio_count > 0:
                            if geo_count > 0:
                                # Most likely a Horizons bio
                                system_data.biological_signals[body_id] = 0
                            else:
                                system_data.biological_signals[body_id] = bio_count

                    # Collect genus information from SAA signals (comes after SAAScanComplete)
                    elif event == 'SAASignalsFound':
                        body_id = entry.get('BodyID')
                        body_name = entry.get('BodyName')
                        genuses = entry.get('Genuses', [])

                        if body_id is not None and genuses:
                            # Take the first genus (typically there's only one for bio signals)
                            genus_localised = genuses[0].get('Genus_Localised', '')
                            if genus_localised:
                                system_data.genus_info[body_id] = genus_localised

                                # Update existing candidate if already processed
                                for candidate in candidates:
                                    candidate_system = candidate.get('system_address')

                                    if (candidate['body_data'].get('BodyName') == body_name and
                                        candidate_system == system_data.system_address):
                                        candidate['HasStratum'] = genus_localised.lower() == 'stratum'
                                        candidate['genus'] = genus_localised
                                        break

                    # When SAA scan completes, evaluate the body
                    elif event == 'SAAScanComplete':
                        body_id = entry.get('BodyID')

                        # Check if this body has biological signals
                        if body_id in system_data.biological_signals:
                            bio_count = system_data.biological_signals[body_id]

                            if body_id in system_data.bodies:
                                body = system_data.bodies[body_id]
                                # print(f"Evaluating {body.get('BodyName')}")
                                evaluation = evaluate_stratum_candidate(body, system_data, bio_count)

                                if evaluation['is_candidate']:
                                    #print(f"  is candidate")
                                    # Determine HasStratum based on genus (may be updated later by SAASignalsFound)
                                    genus = system_data.genus_info.get(body_id, '')
                                    has_stratum = genus.lower() == 'stratum' if genus else False

                                    candidate_info = {
                                        'body_name': body.get('BodyName'),
                                        'system_name': system_data.system_name,
                                        'system_address': system_data.system_address,
                                        'body_data': body,
                                        'evaluation': evaluation,
                                        'HasStratum': has_stratum,
                                        'genus': genus if genus else 'Unknown',
                                        'source_file': file_path,
                                        'line_number': line_number
                                    }
                                    candidates.append(candidate_info)
                                    new_candidates_count += 1

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Warning: Log file '{file_path}' not found.")
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

    return new_candidates_count


def process_log_directory(directory: str, file_pattern: str = "*.log",
                         verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process all log files in a directory in chronological order.
    Preserves system context across files.

    Returns:
        List of all candidate bodies found
    """
    all_candidates = []
    system_data = SystemData()  # Shared system data across all files
    log_dir = Path(directory)

    if not log_dir.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return all_candidates

    log_files = list(log_dir.glob(file_pattern))

    if not log_files:
        print(f"Warning: No files matching pattern '{file_pattern}' found in '{directory}'.")
        return all_candidates

    # Sort files by timestamp in filename for chronological processing
    log_files.sort(key=lambda f: extract_timestamp_from_filename(f.name))

    if verbose:
        print(f"Processing {len(log_files)} files in chronological order...")

    for log_file in log_files:
        if verbose:
            print(f"Processing: {log_file.name}")

        new_count = process_log_file(str(log_file), system_data, all_candidates)

        if new_count > 0 and verbose:
            print(f"  Found {new_count} new candidate(s)")

    # Check any remaining bodies with biological signals at end of all files
    for body_id, bio_count in system_data.biological_signals.items():
        if body_id in system_data.bodies:
            body = system_data.bodies[body_id]

            # Skip if already processed
            already_exists = False
            for candidate in all_candidates:
                if (candidate['body_data'].get('BodyID') == body_id and
                    candidate.get('system_address') == system_data.system_address):
                    already_exists = True
                    break

            if not already_exists:
                evaluation = evaluate_stratum_candidate(body, system_data, bio_count)

                if evaluation['is_candidate']:
                    # Determine HasStratum based on genus
                    genus = system_data.genus_info.get(body_id, '')
                    has_stratum = genus.lower() == 'stratum' if genus else False

                    candidate_info = {
                        'body_name': body.get('BodyName'),
                        'system_name': system_data.system_name,
                        'system_address': system_data.system_address,
                        'body_data': body,
                        'evaluation': evaluation,
                        'HasStratum': has_stratum,
                        'genus': genus if genus else 'Unknown',
                        'source_file': 'end_of_directory',
                        'line_number': 'end_of_directory'
                    }
                    all_candidates.append(candidate_info)

    return all_candidates


def save_results(candidates: List[Dict[str, Any]], output_file: str,
                format_type: str = "json"):
    """Save candidate results to output file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if format_type == "json":
                json.dump(candidates, f, indent=2, default=str)
            else:  # jsonl
                for candidate in candidates:
                    f.write(json.dumps(candidate, default=str) + '\n')

        print(f"\nResults saved to '{output_file}'")

    except Exception as e:
        print(f"Error saving results to '{output_file}': {e}")


def print_simple_summary(candidates: List[Dict[str, Any]]):
    """Print a simple list of candidate bodies found."""
    if candidates:
        print("\nCandidates found:")
        for candidate in candidates:
            body_name = candidate['body_name']
            system_name = candidate['system_name']
            has_stratum = candidate.get('HasStratum', False)
            genus = candidate.get('genus', 'Unknown')
            #status = '✓ Stratum' if has_stratum else f'✗ {genus}'
            status = '✓' if has_stratum else f'✗ {genus}'
            #print(f"  {body_name} ({system_name}) - {status}")
            print(f"  {body_name} - {status}")

        # Count by genus
        stratum_count = sum(1 for c in candidates if c.get('HasStratum', False))
        other_count = len(candidates) - stratum_count

        print(f"\nTotal: {len(candidates)} candidate(s)")
        if stratum_count > 0:
            print(f"  - {stratum_count} with Stratum Tectonicas")
        if other_count > 0:
            print(f"  - {other_count} with other genus")
    else:
        print("\nNo candidates found that meet all five criteria.")#!/usr/bin/env python3


def generate_summary_report(candidates: List[Dict[str, Any]], output_file: Optional[str] = None):
    """Generate a human-readable summary report."""
    lines = []

    def add_line(text=""):
        lines.append(text)

    add_line("=" * 80)
    add_line("STRATUM TECTONICAS CANDIDATES (ALL CRITERIA MET)")
    add_line("=" * 80)
    add_line()

    add_line(f"Total candidates found: {len(candidates)}")
    add_line()

    if not candidates:
        add_line("No candidates found that meet all five criteria.")
        report_text = '\n'.join(lines)

        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"Summary report saved to '{output_file}'")
            except Exception as e:
                print(f"Error saving report: {e}")
        else:
            print(report_text)

        return

    # List all candidates
    add_line("DETAILED CANDIDATE LIST")
    add_line("=" * 80)

    for i, candidate in enumerate(candidates, 1):
        body_name = candidate['body_name']
        system_name = candidate['system_name']
        evaluation = candidate['evaluation']
        body = candidate['body_data']

        add_line()
        add_line(f"#{i} - {body_name}")
        add_line(f"System: {system_name}")
        add_line()

        # Show genus information
        has_stratum = candidate.get('HasStratum', False)
        genus = candidate.get('genus', 'Unknown')
        add_line(f"Biological Genus: {genus}")
        add_line(f"HasStratum: {has_stratum}")
        add_line()

        add_line("All Criteria Met:")
        for criterion, value in evaluation['criteria_met'].items():
            add_line(f"  ✓ {criterion}: {value}")

        add_line()
        add_line("Body Properties:")
        add_line(f"  Planet Class: {body.get('PlanetClass', 'N/A')}")
        add_line(f"  Atmosphere: {body.get('Atmosphere', 'N/A')}")
        add_line(f"  Temperature: {body.get('SurfaceTemperature', 'N/A')}K")
        add_line(f"  Mass: {body.get('MassEM', 'N/A')} Earth Masses")
        add_line(f"  Gravity: {body.get('SurfaceGravity', 'N/A')} m/s²")
        add_line(f"  Pressure: {body.get('SurfacePressure', 'N/A')} Pa")
        add_line(f"  Landable: {body.get('Landable', 'N/A')}")

        # Show materials if available
        materials = body.get('Materials', [])
        if materials:
            add_line()
            add_line("  Materials (Top 5):")
            for material in materials[:5]:
                mat_name = material.get('Name', 'Unknown')
                mat_percent = material.get('Percent', 0)
                add_line(f"    - {mat_name.capitalize()}: {mat_percent:.2f}%")

        add_line()
        add_line("-" * 80)

    report_text = '\n'.join(lines)

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Summary report saved to '{output_file}'")
        except Exception as e:
            print(f"Error saving report: {e}")
    else:
        print(report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Find Stratum Tectonicas candidates in Elite Dangerous log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Search Criteria:
  - Exactly 1 biological signal
  - Thin atmosphere
  - High metal content body
  - Temperature >= 165K
  - Parent star types: TTS, F, M, T, Y, K, D, W, Ae, L, None (barycenters)

Examples:
  %(prog)s -d /path/to/logs -o candidates.json
  %(prog)s -f Journal.log -o results.json --report summary.txt
  %(prog)s -d logs/ --pattern "Journal*.log" -o stratum_candidates.json
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', help='Single log file to process')
    input_group.add_argument('-d', '--directory', help='Directory containing log files')

    # Output options
    parser.add_argument('-o', '--output', required=True,
                       help='Output JSON file for candidate data')
    parser.add_argument('--report',
                       help='Generate human-readable summary report')
    parser.add_argument('--format', choices=['json', 'jsonl'], default='json',
                       help='Output format (default: json)')

    # Processing options
    parser.add_argument('--pattern', default='*.log',
                       help='File pattern for directory processing (default: *.log)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output during processing')

    args = parser.parse_args()

    # Process files
    if args.verbose:
        print("Searching for Stratum Tectonicas candidates...")
        print("Only bodies meeting ALL five criteria will be reported.")
        print()

    if args.file:
        system_data = SystemData()
        all_candidates = []
        process_log_file(args.file, system_data, all_candidates)
        candidates = all_candidates
    else:
        candidates = process_log_directory(args.directory, args.pattern, args.verbose)

    # Save results
    if candidates:
        # Move body_data one level up to match the format the analyzer expects
        cleaned_candidates = []
        for c in candidates:
            new_candidate = c['body_data'].copy()
            new_candidate.update(c)
            new_candidate['body_data'] = []
            cleaned_candidates.append(new_candidate)

        save_results(cleaned_candidates, args.output, args.format)

        # Generate detailed report if requested
        if args.report:
            generate_summary_report(candidates, args.report)
        else:
            # Print simple summary to console
            print_simple_summary(candidates)
    else:
        print("\nNo candidates found that meet all five criteria.")

    if args.verbose:
        print(f"\nSearch complete! Found {len(candidates)} candidate(s).")


if __name__ == "__main__":
    main()
