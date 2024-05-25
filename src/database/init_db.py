"""
Database Initialization Script

This script initializes the PostgreSQL database for the workplace safety analytics platform.
"""

import psycopg2
import psycopg2.extras
from pathlib import Path
import yaml
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Handles database initialization and schema creation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.db_config = self.config.get('database', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default database configuration."""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'workplace_safety',
                'username': 'postgres',
                'password': 'password',
                'schema': 'public'
            }
        }
    
    def create_database(self) -> bool:
        """Create the database if it doesn't exist."""
        try:
            # Connect to PostgreSQL server (default postgres database)
            conn_params = self.db_config.copy()
            db_name = conn_params.pop('database')
            conn_params['database'] = 'postgres'  # Connect to default database first
            
            with psycopg2.connect(**conn_params) as conn:
                conn.autocommit = True
                cursor = conn.cursor()
                
                # Check if database exists
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                if cursor.fetchone():
                    logger.info(f"Database '{db_name}' already exists")
                    return True
                
                # Create database
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Database '{db_name}' created successfully")
                
                return True
                
        except Exception as e:
            logger.error(f"Error creating database: {str(e)}")
            return False
    
    def initialize_schema(self) -> bool:
        """Initialize the database schema."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                cursor = conn.cursor()
                
                # Read and execute schema SQL
                schema_file = Path("sql/create_schema.sql")
                if not schema_file.exists():
                    logger.error(f"Schema file not found: {schema_file}")
                    return False
                
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                
                # Execute schema creation
                cursor.execute(schema_sql)
                conn.commit()
                
                logger.info("Database schema initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing schema: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                logger.info(f"Connected to PostgreSQL: {version}")
                return True
                
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def setup_database(self) -> bool:
        """Complete database setup process."""
        logger.info("Starting database setup...")
        
        # Create database
        if not self.create_database():
            return False
        
        # Test connection
        if not self.test_connection():
            return False
        
        # Initialize schema
        if not self.initialize_schema():
            return False
        
        logger.info("Database setup completed successfully")
        return True

def main():
    """Main function to initialize the database."""
    logger.info("=== Database Initialization ===")
    
    initializer = DatabaseInitializer()
    
    if initializer.setup_database():
        logger.info("Database is ready for use!")
    else:
        logger.error("Database setup failed!")
        exit(1)

if __name__ == "__main__":
    main() 