#!/usr/bin/env python3
"""
Calculate compression ratios for the codebase visualization PNG
"""

import os
import json
from pathlib import Path

def calculate_codebase_compression():
    """Calculate total size of all files and compression ratios"""
    total_size_bytes = 0
    file_count = 0
    file_details = []

    # Calculate size of all code files in the directory
    code_extensions = {'.py', '.js', '.cpp', '.c', '.h', '.java', '.cs', '.php', '.rb', '.go', '.rs', '.yaml', '.yml', '.json', '.md', '.txt'}

    for file_path in Path('.').rglob('*'):
        if file_path.suffix.lower() in code_extensions and file_path.is_file():
            try:
                size = file_path.stat().st_size
                total_size_bytes += size
                file_count += 1
                file_details.append({
                    'file': str(file_path),
                    'size_bytes': size,
                    'size_kb': size / 1024
                })
            except Exception as e:
                print(f'Error with {file_path}: {e}')

    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_kb / 1024

    print(f'Total files processed: {file_count}')
    print(f'Total size: {total_size_bytes:,} bytes')
    print(f'Total size: {total_size_kb:.2f} KB')
    print(f'Total size: {total_size_mb:.2f} MB')

    # PNG size
    png_size_kb = 23117  # Given in the prompt
    compression_ratio = png_size_kb / total_size_kb if total_size_kb > 0 else 0

    print(f'PNG visualization size: {png_size_kb} KB')
    print(f'Compression ratio: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)')
    print(f'Space savings: {(1-compression_ratio)*100:.2f}%')

    # Save detailed analysis
    analysis = {
        'total_files': file_count,
        'total_size_bytes': total_size_bytes,
        'total_size_kb': total_size_kb,
        'total_size_mb': total_size_mb,
        'png_size_kb': png_size_kb,
        'compression_ratio': compression_ratio,
        'space_savings_percent': (1-compression_ratio)*100,
        'largest_files': sorted(file_details, key=lambda x: x['size_kb'], reverse=True)[:20]
    }

    with open('compression_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print('Detailed analysis saved to compression_analysis.json')
    return analysis

if __name__ == "__main__":
    calculate_codebase_compression()
