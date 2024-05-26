"""
OSHA Data Download Module

This module handles downloading and initial processing of OSHA inspection data
from various sources including the DOL Enforcement Data Portal.
"""

import os
import requests
import pandas as pd
import zipfile
from pathlib import Path
from typing import Optional, Dict, List
import logging
from tqdm import tqdm
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSHADataDownloader:
    """Downloads OSHA inspection and violation data from official sources."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the downloader with configuration."""
        self.config = self._load_config(config_path)
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'data_sources': {
                'osha': {
                    'inspection_data_path': 'data/raw/osha_inspections.csv',
                    'violation_data_path': 'data/raw/osha_violations.csv'
                }
            }
        }
    
    def download_enforcement_data(self, year: int = 2023) -> bool:
        """
        Download OSHA enforcement data for a specific year.
        
        Args:
            year: Year for which to download data
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Note: In a real implementation, you would use the actual OSHA API endpoints
            # For this example, we'll simulate downloading data
            logger.info(f"Downloading OSHA enforcement data for {year}...")
            
            # Simulate download - in practice, this would call the actual API
            sample_data = self._generate_sample_data()
            
            # Save the data
            inspection_file = self.data_dir / f"osha_inspections_{year}.csv"
            violation_file = self.data_dir / f"osha_violations_{year}.csv"
            
            sample_data['inspections'].to_csv(inspection_file, index=False)
            sample_data['violations'].to_csv(violation_file, index=False)
            
            logger.info(f"Data saved to {inspection_file} and {violation_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return False
    
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate sample OSHA data for demonstration.
        In a real implementation, this would be replaced with actual API calls.
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample inspection data
        n_inspections = 50000  # Target: 50K+ records
        
        # Sample establishments
        establishments = []
        for i in range(1000):
            establishments.append({
                'establishment_id': f"EST_{i:06d}",
                'establishment_name': f"Company {i}",
                'city': np.random.choice(['Houston', 'Chicago', 'Los Angeles', 'New York', 'Atlanta']),
                'state': np.random.choice(['TX', 'IL', 'CA', 'NY', 'GA']),
                'naics_code': np.random.choice(['236220', '238210', '311612', '484121', '561720'])
            })
        
        # Generate inspections
        inspections = []
        violations = []
        
        for i in range(n_inspections):
            establishment = np.random.choice(establishments)
            
            # Random date in the last 3 years
            start_date = datetime.now() - timedelta(days=3*365)
            random_days = np.random.randint(0, 3*365)
            inspection_date = start_date + timedelta(days=random_days)
            
            # Inspection details
            inspection = {
                'inspection_id': i + 1,
                'establishment_id': establishment['establishment_id'],
                'establishment_name': establishment['establishment_name'],
                'city': establishment['city'],
                'state': establishment['state'],
                'naics_code': establishment['naics_code'],
                'open_date': inspection_date.strftime('%Y-%m-%d'),
                'inspection_type': np.random.choice(['Planned', 'Complaint', 'Accident', 'Referral']),
                'has_injury': np.random.choice([True, False], p=[0.15, 0.85]),  # 15% have injuries
                'scope': np.random.choice(['Complete', 'Partial', 'Records'])
            }
            
            inspections.append(inspection)
            
            # Generate violations for this inspection
            n_violations = np.random.poisson(2)  # Average 2 violations per inspection
            
            for v in range(n_violations):
                violation = {
                    'violation_id': len(violations) + 1,
                    'inspection_id': inspection['inspection_id'],
                    'standard_violated': np.random.choice([
                        '1926.501(b)(1)', '1926.95(a)', '1910.147(c)(4)', 
                        '1926.451(b)(1)', '1910.212(a)(1)', '1926.1053(a)(1)'
                    ]),
                    'violation_type': np.random.choice(['Serious', 'Other', 'Willful', 'Repeat'], 
                                                     p=[0.6, 0.25, 0.1, 0.05]),
                    'penalty_amount': np.random.lognormal(8, 1),  # Log-normal distribution for penalties
                    'is_serious': np.random.choice([True, False], p=[0.6, 0.4]),
                    'is_repeat': np.random.choice([True, False], p=[0.05, 0.95]),
                    'is_willful': np.random.choice([True, False], p=[0.1, 0.9])
                }
                violations.append(violation)
        
        return {
            'inspections': pd.DataFrame(inspections),
            'violations': pd.DataFrame(violations)
        }
    
    def download_historical_data(self, years: List[int]) -> bool:
        """
        Download historical OSHA data for multiple years.
        
        Args:
            years: List of years to download
            
        Returns:
            bool: True if all downloads successful
        """
        success = True
        for year in years:
            if not self.download_enforcement_data(year):
                success = False
        return success
    
    def validate_data(self, file_path: str) -> bool:
        """
        Validate downloaded data for completeness and quality.
        
        Args:
            file_path: Path to the data file to validate
            
        Returns:
            bool: True if data is valid
        """
        try:
            df = pd.read_csv(file_path)
            
            # Basic validation checks
            if df.empty:
                logger.error(f"Data file {file_path} is empty")
                return False
            
            # Check for required columns (will depend on actual OSHA data structure)
            required_columns = ['inspection_id', 'establishment_name', 'open_date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            logger.info(f"Data validation successful for {file_path}")
            logger.info(f"Records: {len(df)}, Columns: {len(df.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False

def main():
    """Main function to download OSHA data."""
    downloader = OSHADataDownloader()
    
    # Download data for recent years
    years = [2021, 2022, 2023]
    
    logger.info("Starting OSHA data download...")
    
    if downloader.download_historical_data(years):
        logger.info("Data download completed successfully")
        
        # Validate the downloaded data
        for year in years:
            inspection_file = f"data/raw/osha_inspections_{year}.csv"
            violation_file = f"data/raw/osha_violations_{year}.csv"
            
            downloader.validate_data(inspection_file)
            downloader.validate_data(violation_file)
    else:
        logger.error("Data download failed")

if __name__ == "__main__":
    main() 