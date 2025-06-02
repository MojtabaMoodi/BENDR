#!/usr/bin/env python3
"""
Script to check where gender information is stored in Sleep-EDF dataset
"""

import pyedflib
from pathlib import Path

def check_file_for_gender(file_path):
    """Check a single EDF file for gender information"""
    try:
        f = pyedflib.EdfReader(str(file_path))
        header = f.getHeader()
        f._close()
        
        print(f'\n{file_path.name}:')
        
        # Look for any key that might contain gender info
        gender_found = False
        for key, value in header.items():
            if any(keyword in key.lower() for keyword in ['gender', 'sex', 'patient', 'subject']):
                print(f'  {key}: {value}')
                gender_found = True
        
        # Also check common EDF header fields
        for field in ['patientname', 'patientcode', 'admincode']:
            if field in header:
                print(f'  {field}: {header[field]}')
                gender_found = True
                
        if not gender_found:
            print('  No obvious gender-related fields found')
            
        return gender_found
        
    except Exception as e:
        print(f'Error reading {file_path.name}: {e}')
        return False

def main():
    data_dir = Path('/Volumes/Data/SSC/sleep-cassette/')
    
    print("=== Checking PSG Files for Gender Info ===")
    psg_files = list(data_dir.glob('*PSG.edf'))[:5]
    psg_has_gender = any(check_file_for_gender(f) for f in psg_files)
    
    print("\n=== Checking Hypnogram Files for Gender Info ===")
    hypnogram_files = list(data_dir.glob('*Hypnogram.edf'))[:5]
    hypno_has_gender = any(check_file_for_gender(f) for f in hypnogram_files)
    
    print(f"\n=== Summary ===")
    print(f"PSG files contain gender info: {psg_has_gender}")
    print(f"Hypnogram files contain gender info: {hypno_has_gender}")
    
    if not psg_has_gender and not hypno_has_gender:
        print("\nGender info might be:")
        print("1. Encoded in the filename/subject ID")
        print("2. In a separate metadata file")
        print("3. Not available in this dataset")
        
        # Let's check if there's any pattern in the filenames
        print("\n=== Checking filename patterns ===")
        for f in psg_files[:3]:
            print(f"PSG: {f.name}")
        for f in hypnogram_files[:3]:
            print(f"Hypnogram: {f.name}")

if __name__ == "__main__":
    main() 