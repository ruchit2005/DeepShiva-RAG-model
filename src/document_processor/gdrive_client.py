import os
import io
from pathlib import Path
from typing import List, Dict, Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle
from tqdm import tqdm
from config.settings import settings

class GoogleDriveClient:
    """Handles Google Drive authentication and file operations."""
    
    SUPPORTED_MIME_TYPES = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'text/plain': '.txt',
        'text/markdown': '.md',
        'application/vnd.google-apps.document': '.docx',  # Google Docs
        'application/vnd.google-apps.presentation': '.pptx',  # Google Slides
        'application/vnd.google-apps.spreadsheet': '.xlsx',  # Google Sheets
    }
    
    EXPORT_MIME_TYPES = {
        'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    }
    
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API."""
        if settings.GDRIVE_TOKEN_PATH.exists():
            with open(settings.GDRIVE_TOKEN_PATH, 'rb') as token:
                self.creds = pickle.load(token)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(settings.GDRIVE_CREDENTIALS_PATH), 
                    settings.GDRIVE_SCOPES
                )
                self.creds = flow.run_local_server(port=0)
            
            with open(settings.GDRIVE_TOKEN_PATH, 'wb') as token:
                pickle.dump(self.creds, token)
        
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def list_files(self, folder_id: Optional[str] = None, 
                   recursive: bool = True) -> List[Dict]:
        """
        List all supported files in Google Drive.
        
        Args:
            folder_id: Specific folder ID to search in (None for root)
            recursive: Search in subfolders
        
        Returns:
            List of file metadata dictionaries
        """
        files = []
        page_token = None
        
        query_parts = []
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        query_parts.append("trashed = false")
        
        mime_types_query = " or ".join([
            f"mimeType = '{mime}'" 
            for mime in self.SUPPORTED_MIME_TYPES.keys()
        ])
        query_parts.append(f"({mime_types_query})")
        
        query = " and ".join(query_parts)
        
        while True:
            results = self.service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)",
                pageToken=page_token
            ).execute()
            
            files.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            
            if not page_token:
                break
        
        if recursive and folder_id:
            # Get subfolders
            folder_query = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            folders = self.service.files().list(
                q=folder_query,
                fields="files(id)"
            ).execute().get('files', [])
            
            for folder in folders:
                files.extend(self.list_files(folder['id'], recursive=True))
        
        return files
    
    def download_file(self, file_id: str, file_name: str, 
                      mime_type: str, destination: Path) -> Path:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            file_name: Original file name
            mime_type: MIME type of the file
            destination: Directory to save the file
        
        Returns:
            Path to downloaded file
        """
        destination.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        if mime_type in self.EXPORT_MIME_TYPES:
            # Google Workspace file - needs export
            export_mime = self.EXPORT_MIME_TYPES[mime_type]
            extension = self.SUPPORTED_MIME_TYPES[mime_type]
            
            # Remove original extension if exists
            base_name = Path(file_name).stem
            file_path = destination / f"{base_name}{extension}"
            
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType=export_mime
            )
        else:
            # Regular file - direct download
            extension = self.SUPPORTED_MIME_TYPES.get(mime_type, '')
            if not file_name.endswith(extension):
                file_path = destination / f"{file_name}{extension}"
            else:
                file_path = destination / file_name
            
            request = self.service.files().get_media(fileId=file_id)
        
        # Download file
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.close()
        return file_path
    
    def download_all_files(self, folder_id: Optional[str] = None) -> List[Path]:
        """
        Download all supported files from Google Drive.
        
        Args:
            folder_id: Specific folder ID (None for all accessible files)
        
        Returns:
            List of paths to downloaded files
        """
        files = self.list_files(folder_id, recursive=True)
        downloaded_files = []
        
        print(f"Found {len(files)} files to download")
        
        for file in tqdm(files, desc="Downloading files"):
            try:
                file_path = self.download_file(
                    file['id'],
                    file['name'],
                    file['mimeType'],
                    settings.RAW_DATA_DIR
                )
                downloaded_files.append(file_path)
            except Exception as e:
                print(f"Error downloading {file['name']}: {e}")
        
        return downloaded_files
    
    def get_folder_id_by_name(self, folder_name: str) -> Optional[str]:
        """Find folder ID by name."""
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = self.service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()
        
        files = results.get('files', [])
        return files[0]['id'] if files else None