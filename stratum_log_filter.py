#!/usr/bin/env python3
"""
Log File Body Name Filter

This script processes log files containing JSON entries and extracts
entries that match specified body names from an input list.
"""

import json
import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple


def load_body_names(input_file: str) -> Dict[str, bool]:
    """
    Load body names and stratum information from input file.

    Args:
        input_file: Path to file containing body names and stratum info (format: "BodyName;0" or "BodyName;1")

    Returns:
        Dictionary mapping body names to HasStratum boolean values
    """
    body_data = {}

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue

                # Parse the line format: "BodyName;0" or "BodyName;1"
                if ';' not in line:
                    print(f"Warning: Invalid format on line {line_number}: '{line}' (expected format: 'BodyName;0' or 'BodyName;1')")
                    continue

                parts = line.split(';', 1)  # Split only on first semicolon
                if len(parts) != 2:
                    print(f"Warning: Invalid format on line {line_number}: '{line}'")
                    continue

                body_name = parts[0].strip()
                stratum_value = parts[1].strip()

                if stratum_value not in ['0', '1']:
                    print(f"Warning: Invalid stratum value on line {line_number}: '{stratum_value}' (expected '0' or '1')")
                    continue

                has_stratum = stratum_value == '1'
                body_data[body_name] = has_stratum

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file '{input_file}': {e}")
        sys.exit(1)

    return body_data


def process_log_file(file_path: str, target_bodies: Dict[str, bool], case_sensitive: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Process a single log file and extract matching Scan entries.
    Returns only the latest scan for each body.

    Args:
        file_path: Path to the log file
        target_bodies: Dictionary mapping body names to HasStratum boolean values
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Dictionary mapping body names to their latest scan entries
    """
    body_scans = {}  # Dictionary to store latest scan for each body

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_number = 0

            for line in f:
                line_number += 1
                line = line.strip()

                if not line:
                    continue

                try:
                    # Parse JSON entry
                    entry = json.loads(line)

                    # Only process Scan events
                    if entry.get('event') != 'Scan':
                        continue

                    # Check if entry has BodyName field
                    if 'BodyName' in entry:
                        body_name = entry['BodyName']
                        matched_body = None

                        # Perform matching based on case sensitivity
                        if case_sensitive:
                            if body_name in target_bodies:
                                matched_body = body_name
                        else:
                            # Case-insensitive matching
                            for target_name in target_bodies:
                                if body_name.lower() == target_name.lower():
                                    matched_body = target_name
                                    break

                        if matched_body:
                            # Get timestamp for comparison
                            timestamp = entry.get('timestamp', '')

                            # Check if we already have a scan for this body
                            if body_name in body_scans:
                                existing_timestamp = body_scans[body_name].get('timestamp', '')
                                # Keep the later timestamp (lexicographic comparison works for ISO format)
                                if timestamp <= existing_timestamp:
                                    continue  # Skip this entry, we have a later one

                            # Add metadata about the match and stratum info
                            entry['_match_info'] = {
                                'source_file': file_path,
                                'line_number': line_number
                            }
                            entry['HasStratum'] = target_bodies[matched_body]
                            body_scans[body_name] = entry

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_number} in {file_path}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Warning: Log file '{file_path}' not found.")
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

    return body_scans


def process_log_directory(directory: str, target_bodies: Dict[str, bool], file_pattern: str = "*.log",
                         case_sensitive: bool = True) -> List[Dict[str, Any]]:
    """
    Process all log files in a directory for Scan events.
    Returns only the latest scan for each body across all files.

    Args:
        directory: Directory containing log files
        target_bodies: Dictionary mapping body names to HasStratum boolean values
        file_pattern: File pattern to match (e.g., "*.log", "*.json")
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        List of latest scan entries for each matched body
    """
    all_body_scans = {}  # Dictionary to store latest scan for each body across all files
    log_dir = Path(directory)

    if not log_dir.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return []

    log_files = list(log_dir.glob(file_pattern))

    if not log_files:
        print(f"Warning: No files matching pattern '{file_pattern}' found in '{directory}'.")
        return []

    print(f"Processing {len(log_files)} files...")

    for log_file in log_files:
        print(f"Processing: {log_file}")
        file_body_scans = process_log_file(str(log_file), target_bodies, case_sensitive)

        # Merge with existing scans, keeping the latest timestamp for each body
        for body_name, scan_entry in file_body_scans.items():
            timestamp = scan_entry.get('timestamp', '')

            if body_name in all_body_scans:
                existing_timestamp = all_body_scans[body_name].get('timestamp', '')
                # Keep the later timestamp (lexicographic comparison works for ISO format)
                if timestamp > existing_timestamp:
                    all_body_scans[body_name] = scan_entry
            else:
                all_body_scans[body_name] = scan_entry

        print(f"  Found {len(file_body_scans)} unique body scans in this file")

    print(f"Total unique bodies found across all files: {len(all_body_scans)}")
    return list(all_body_scans.values())


def save_results(matches: List[Dict[str, Any]], output_file: str, format_type: str = "json"):
    """
    Save matching results to output file.

    Args:
        matches: List of matching entries
        output_file: Output file path
        format_type: Output format ("json" or "jsonl")
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if format_type == "json":
                json.dump(matches, f, indent=2, default=str)
            else:  # jsonl format
                for match in matches:
                    f.write(json.dumps(match, default=str) + '\n')

        print(f"Results saved to '{output_file}'")

    except Exception as e:
        print(f"Error saving results to '{output_file}': {e}")


def print_summary(matches: List[Dict[str, Any]], target_bodies: Dict[str, bool]):
    """Print a summary of the results."""
    print(f"\n=== SUMMARY ===")
    print(f"Target bodies searched: {len(target_bodies)}")
    print(f"Total matches found: {len(matches)}")

    if matches:
        # Group by body name
        body_counts = {}
        stratum_counts = {'with_stratum': 0, 'without_stratum': 0}

        for match in matches:
            body_name = match.get('BodyName', 'Unknown')
            body_counts[body_name] = body_counts.get(body_name, 0) + 1

            # Count stratum
            if match.get('HasStratum', False):
                stratum_counts['with_stratum'] += 1
            else:
                stratum_counts['without_stratum'] += 1

        print(f"\nMatches by body name:")
        for body_name, count in sorted(body_counts.items()):
            has_stratum = "Yes" if any(m.get('BodyName') == body_name and m.get('HasStratum') for m in matches) else "No"
            print(f"  {body_name}: {count} (Stratum: {has_stratum})")

        print(f"\nStratum distribution:")
        print(f"  Bodies with Stratum: {stratum_counts['with_stratum']}")
        print(f"  Bodies without Stratum: {stratum_counts['without_stratum']}")

        # Group by scan type
        scan_type_counts = {}
        for match in matches:
            scan_type = match.get('ScanType', 'Unknown')
            scan_type_counts[scan_type] = scan_type_counts.get(scan_type, 0) + 1

        print(f"\nMatches by scan type:")
        for scan_type, count in sorted(scan_type_counts.items()):
            print(f"  {scan_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Scan event log entries matching specified body names with Stratum information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d /path/to/logs -i body_names.txt -o results.json
  %(prog)s -f single_log.json -i targets.txt -o matches.jsonl --format jsonl
  %(prog)s -d logs/ -i bodies.txt --case-insensitive --pattern "*.json"

Input file format:
  Each line should contain: BodyName;0 or BodyName;1
  Where 0 = no Stratum, 1 = has Stratum
  Example:
    Chua Eop QF-L d9-3 CD 2;0
    Chua Eop QF-L d9-3 CD 4;1
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', help='Single log file to process')
    input_group.add_argument('-d', '--directory', help='Directory containing log files')

    parser.add_argument('-i', '--input-list', required=True,
                       help='File containing body names and stratum info (format: "BodyName;0" or "BodyName;1")')

    # Output options
    parser.add_argument('-o', '--output', default='matches.json',
                       help='Output file for results (default: matches.json)')
    parser.add_argument('--format', choices=['json', 'jsonl'], default='json',
                       help='Output format (default: json)')

    # Processing options
    parser.add_argument('--pattern', default='*.log',
                       help='File pattern for directory processing (default: *.log)')
    parser.add_argument('--case-insensitive', action='store_true',
                       help='Perform case-insensitive matching')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip printing summary')

    args = parser.parse_args()

    # Load target body names
    print(f"Loading body names and stratum info from '{args.input_list}'...")
    target_bodies = load_body_names(args.input_list)
    print(f"Loaded {len(target_bodies)} body names to search for")

    if not target_bodies:
        print("Error: No body names found in input file.")
        sys.exit(1)

    # Show stratum distribution in input
    stratum_count = sum(1 for has_stratum in target_bodies.values() if has_stratum)
    print(f"  Bodies with Stratum: {stratum_count}")
    print(f"  Bodies without Stratum: {len(target_bodies) - stratum_count}")

    # Process files
    case_sensitive = not args.case_insensitive

    if args.file:
        file_body_scans = process_log_file(args.file, target_bodies, case_sensitive)
        matches = list(file_body_scans.values())
    else:
        matches = process_log_directory(args.directory, target_bodies,
                                      args.pattern, case_sensitive)

    # Save results
    if matches:
        save_results(matches, args.output, args.format)
    else:
        print("No matches found.")

    # Print summary
    if not args.no_summary:
        print_summary(matches, target_bodies)


if __name__ == "__main__":
    main()
