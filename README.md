# LEGO Sorter

A Django web application for scanning and sorting LEGO pieces from PDF catalogs. This application uses computer vision and OCR to automatically detect, identify, and validate LEGO pieces against a ground truth dataset. It helps in digitizing LEGO catalogs and maintaining an accurate inventory of pieces.

## Features

- PDF catalog scanning with automatic piece detection
- OCR-based piece ID and count extraction
- Color detection using computer vision
- Validation against ground truth data
- Export of detected pieces as transparent PNG files
- Web interface for managing piece inventory
- Packet and distribution tracking

## System Dependencies

Before installing Python dependencies, ensure you have these system packages:

- Tesseract OCR for text recognition
- Poppler for PDF processing
- OpenCV system dependencies

### macOS Installation:
```bash
brew install tesseract
brew install poppler
```

### Ubuntu/Debian Installation:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
sudo apt-get install python3-opencv
```

## Python Package Requirements

The following Python packages are required and will be installed via Pipenv:

- pillow - Image processing
- opencv-python - Computer vision operations
- pytesseract - OCR text extraction
- pdf2image - PDF to image conversion
- numpy - Numerical operations
- pandas - Data manipulation
- scikit-learn - Machine learning utilities
- tqdm - Progress bars

## Requirements

- Python 3.12+
- Django 5.2+
- Pipenv (for dependency management)

## Quick Setup

1. Clone the repository and navigate to the project directory:
```bash
cd lego
```

2. Install dependencies using Pipenv:
```bash
pipenv install
```

3. Activate the virtual environment:
```bash
pipenv shell
```

4. Run database migrations:
```bash
python manage.py migrate
```

5. Create a superuser (for admin access):
```bash
python manage.py createsuperuser
```

6. Collect static files:
```bash
python manage.py update_static
```

## Running the Application

1. Start the development server:
```bash
python manage.py runserver
```

2. Access the application:
- Main application: http://localhost:8000
- Admin interface: http://localhost:8000/admin

## Management Commands

The project includes several custom management commands to help with data management:

### scan_pieces
Scans PDF pages containing LEGO pieces and validates them against ground truth data.
```bash
python manage.py scan_pieces --pdf "page_number.pdf" [--training]
```
This command will:
- Convert PDF pages to images
- Detect individual LEGO pieces using computer vision
- Extract piece IDs and counts using OCR
- Detect piece colors
- Validate against ground truth data (in training mode)
- Save detected pieces as transparent PNG files

Options:
- `--pdf`: Path to the PDF file to scan
- `--training`: Enable training mode to validate against ground truth data

### import_csv
Imports LEGO piece data from the CSV file.
```bash
python manage.py import_csv
```
This command will:
- Clear existing piece data
- Import new data from "LEGO Sorting Sheet - Sheet2.csv"
- Set up packet counts and piece information

### update_static
Updates static files and checks for missing piece images.
```bash
python manage.py update_static
```
This command will:
- Collect all static files
- Report the number of piece images found
- Warn about any missing piece images

## Project Structure

- `pieces/` - Main application directory containing models, views, and templates
- `static/pieces/` - Static files including piece images
- `media/` - User-uploaded media files
- `templates/pieces/` - HTML templates
- `lego_sorter/` - Project settings and configuration

## Data Management

The application uses SQLite as its database and stores:
- Piece information (color, shape, page number)
- Total and remaining counts
- Packet distribution information
- Image references

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Testing

Run the test suite:
```bash
python manage.py test
```

To test piece detection on a specific page:
```bash
python manage.py scan_pieces --pdf "page_number.pdf" --training
```
The training mode will validate detections against the ground truth data in the CSV file.

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.