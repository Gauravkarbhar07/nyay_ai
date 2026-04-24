"""
Google Sheets Sync Utility for BNS-Constitution Mapping
========================================================

This utility syncs the BNS-Constitution mapping from Google Sheets to JSON format.
This allows non-technical users to edit the data easily via Google Sheets.

Installation:
    pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client

Setup Steps:
    1. Create a Google Sheet with the data
    2. Share the sheet and get the Sheet ID
    3. Set up Google Cloud credentials
    4. Run this utility to sync
"""

import json
import os
from pathlib import Path

try:
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("⚠️  Google Sheets integration not available. Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")


class BNSGoogleSheetsSyncer:
    """Sync BNS-Constitution mapping from Google Sheets"""
    
    def __init__(self, sheet_id, credentials_json_path=None):
        """
        Initialize syncer
        
        Args:
            sheet_id: Google Sheet ID (from URL)
            credentials_json_path: Path to Google Cloud service account JSON
        """
        self.sheet_id = sheet_id
        self.credentials_json_path = credentials_json_path
        self.service = None
        
        if GOOGLE_SHEETS_AVAILABLE and credentials_json_path:
            self._setup_service()
    
    def _setup_service(self):
        """Set up Google Sheets API service"""
        if not os.path.exists(self.credentials_json_path):
            print(f"❌ Credentials file not found: {self.credentials_json_path}")
            return
        
        try:
            credentials = Credentials.from_service_account_file(
                self.credentials_json_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.service = build('sheets', 'v4', credentials=credentials)
            print("✅ Google Sheets API connected successfully")
        except Exception as e:
            print(f"❌ Failed to setup Google Sheets service: {e}")
    
    def fetch_sheet_data(self, sheet_range="'BNS Sections'!A1:Z1000"):
        """
        Fetch data from Google Sheet
        
        Args:
            sheet_range: Sheet range in format "'Sheet Name'!A1:Z1000"
        
        Returns:
            dict: Data from sheet
        """
        if not self.service:
            print("❌ Google Sheets service not initialized")
            return None
        
        try:
            request = self.service.spreadsheets().values().get(
                spreadsheetId=self.sheet_id,
                range=sheet_range
            )
            result = request.execute()
            return result.get('values', [])
        except Exception as e:
            print(f"❌ Failed to fetch sheet data: {e}")
            return None
    
    def sync_to_json(self, output_path=None):
        """
        Sync Google Sheet data to JSON file
        
        Args:
            output_path: Path where to save JSON (default: bns_constitution_mapping.json)
        """
        if output_path is None:
            output_path = os.path.join(
                Path(__file__).parent,
                'data',
                'bns_constitution_mapping.json'
            )
        
        print(f"📊 Starting sync from Google Sheets to {output_path}")
        
        # Fetch data from sheets
        sheet_data = self.fetch_sheet_data()
        if not sheet_data:
            print("❌ No data received from sheet")
            return False
        
        # Convert sheet data to JSON (implementation depends on your sheet structure)
        # This is a template - customize based on your actual sheet structure
        json_data = self._convert_sheet_to_json(sheet_data)
        
        # Save to file
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Successfully synced to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save JSON: {e}")
            return False
    
    def _convert_sheet_to_json(self, sheet_data):
        """Convert sheet data to JSON format"""
        # This needs to be customized based on your sheet structure
        # For now, returning a template
        
        return {
            "metadata": {
                "version": "1.0",
                "source": "Google Sheets Sync",
                "note": "Customize _convert_sheet_to_json based on your sheet structure"
            },
            "bns_sections": []
        }


def manual_google_sheets_setup():
    """
    Manual setup guide for Google Sheets integration
    """
    guide = """
    ========================================
    GOOGLE SHEETS SETUP GUIDE
    ========================================
    
    OPTION 1: Using Service Account (Recommended)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    1. Go to Google Cloud Console: https://console.cloud.google.com/
    2. Create a new project
    3. Enable Google Sheets API
    4. Create Service Account:
       - Click "Create Credentials" → Service Account
       - Fill in details
       - Create JSON key
       - Save the JSON file
    5. Share your Google Sheet with the service account email
    6. Update the syncer with sheet ID and credentials path
    
    OPTION 2: Using OAuth (For individual users)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    1. Create OAuth 2.0 credentials in Google Cloud Console
    2. Use gspread library for simpler OAuth setup
    3. Run: pip install gspread oauth2client
    
    OPTION 3: Manual CSV Export (Simplest)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    1. Edit data in Google Sheets
    2. Export as CSV
    3. Use CSV to JSON converter (provided in utils.py)
    4. Upload JSON to your app
    
    ========================================
    EXAMPLE GOOGLE SHEET STRUCTURE
    ========================================
    
    | BNS ID | Title | Description | Article | Victim Right | Remedy | Contact |
    |--------|-------|-------------|---------|--------------|--------|---------|
    | 101    | Murder| ...         | 21      | Right to...  | Path 1 | 100     |
    
    Then use _convert_sheet_to_json() to parse this format
    
    ========================================
    RECOMMENDED: Use CSV + Python Script
    ========================================
    
    Instead of complex Google Sheets API integration:
    1. Edit data in Google Sheets
    2. Download as CSV
    3. Run csv_to_json() utility below
    """
    
    print(guide)


def csv_to_json_simple(csv_path, output_path=None):
    """
    Simple CSV to JSON converter for BNS-Constitution mapping
    
    Args:
        csv_path: Path to CSV file
        output_path: Where to save JSON
    
    Usage:
        Download Google Sheet as CSV, then:
        csv_to_json_simple('bns_mapping.csv', 'bns_constitution_mapping.json')
    """
    import csv
    
    try:
        # Read CSV
        sections = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sections.append(row)
        
        json_data = {
            "metadata": {
                "version": "1.0",
                "source": "CSV Import",
                "total_sections": len(sections)
            },
            "bns_sections": sections
        }
        
        # Save JSON
        if output_path is None:
            output_path = 'bns_constitution_mapping.json'
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Converted {csv_path} → {output_path}")
        return True
    except Exception as e:
        print(f"❌ CSV conversion failed: {e}")
        return False


if __name__ == '__main__':
    # Display setup guide
    manual_google_sheets_setup()
    
    print("\n" + "="*50)
    print("QUICK START")
    print("="*50)
    
    print("\n1️⃣  For CSV import (easiest):")
    print("   Download Google Sheet as CSV")
    print("   csv_to_json_simple('your_file.csv', 'output.json')")
    
    print("\n2️⃣  For Google Sheets API:")
    print("   syncer = BNSGoogleSheetsSyncer('YOUR_SHEET_ID', 'credentials.json')")
    print("   syncer.sync_to_json()")
    
    print("\n3️⃣  Test CSV conversion:")
    print("   python google_sheets_sync.py test")
