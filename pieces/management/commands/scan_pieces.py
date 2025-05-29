import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from django.core.management.base import BaseCommand
from django.core.management.base import CommandError
from pieces.models import LegoPiece
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm
import traceback
from datetime import datetime

class FileLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        
    def write(self, message, style=None):
        """Write message to log file with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(self.log_file, 'a') as f:
            # Strip any ANSI color codes from message
            clean_message = message.replace('\033[0m', '').replace('\033[91m', '').replace('\033[92m', '')
            f.write(f"[{timestamp}] {clean_message}\n")

class LegoScanner:
    def __init__(self, stdout, training_mode=False, log_file='scanner_output.log'):
        self.stdout = stdout
        self.training_mode = training_mode
        self.logger = FileLogger(log_file)
        self.known_pieces = {}
        self._log("Initializing scanner...")
        self.load_training_data()
    
    def _log(self, message, style=None):
        """Log message to both stdout and file"""
        if hasattr(self.stdout, 'style') and style:
            styled_message = getattr(self.stdout.style, style.lower())(message)
            self.stdout.write(styled_message)
        else:
            self.stdout.write(message)
        self.logger.write(message, style)

    def load_training_data(self):
        """Load ground truth data from CSV and piece images with text below"""
        try:
            # Load CSV data
            csv_path = 'LEGO Sorting Sheet - Sheet2.csv'
            self._log("Reading CSV data...")
            self.ground_truth = pd.read_csv(csv_path)
            self._log(f"Loaded {len(self.ground_truth)} ground truth entries from CSV")
            
            # Load piece template images and their associated text
            pieces_dir = Path('static/pieces')
            self._log("Loading piece templates...")
            template_count = 0
            
            for img_path in pieces_dir.glob('*.png'):
                try:
                    # Read the full image
                    full_img = cv2.imread(str(img_path))
                    if full_img is None:
                        self._log(f"Could not load template image: {img_path}", style='WARNING')
                        continue
                        
                    # Split image into top (piece) and bottom (text) parts
                    height = full_img.shape[0]
                    split_point = int(height * 0.8)  # Adjusted - piece is roughly 80% of image
                    
                    piece_img = full_img[:split_point, :]
                    text_img = full_img[split_point:, :]
                    
                    # Extract text from bottom part using improved preprocessing
                    gray_text = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
                    # Apply adaptive thresholding for better text extraction
                    thresh = cv2.adaptiveThreshold(gray_text, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
                    text = pytesseract.image_to_string(thresh, config='--psm 6')
                    
                    # Parse piece ID from filename or text
                    piece_id = img_path.stem.split('_')[0]  # Remove any _color_count suffix
                    
                    # Store piece image and extracted info
                    self.known_pieces[piece_id] = {
                        'image': piece_img,
                        'text': text.strip(),
                        'color': None,  # Will be set from ground truth
                        'path': str(img_path)
                    }
                    
                    template_count += 1
                    if template_count % 50 == 0:
                        self._log(f"Loaded {template_count} templates...")
                        
                except Exception as e:
                    self._log(f"Error processing template {img_path}: {e}", style='WARNING')
                    continue
            
            # Update piece colors from ground truth
            for _, row in self.ground_truth.iterrows():
                piece_id = str(row['Shape'])
                if piece_id in self.known_pieces:
                    self.known_pieces[piece_id]['color'] = row['Color'].lower()
            
            self._log(f"Successfully loaded {template_count} piece templates")
            
        except Exception as e:
            self._log(f"Error loading training data: {str(e)}", style='ERROR')
            self._log(traceback.format_exc(), style='ERROR')
            raise CommandError(f"Error loading training data: {e}")

    def detect_color(self, image):
        """Enhanced color detection with improved tan/white differentiation"""
        try:
            # Convert to both HSV and LAB color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Create mask to exclude background
            lower_bg = np.array([0, 0, 200])
            upper_bg = np.array([180, 30, 255])
            mask = cv2.bitwise_not(cv2.inRange(hsv, lower_bg, upper_bg))
            
            # Get non-background pixels
            hsv_pixels = hsv[mask > 0].reshape(-1, 3)
            lab_pixels = lab[mask > 0].reshape(-1, 3)
            
            if len(hsv_pixels) == 0:
                return 'unknown'
            
            # Cluster in both color spaces
            kmeans_hsv = KMeans(n_clusters=3, n_init=10)
            kmeans_lab = KMeans(n_clusters=3, n_init=10)
            
            kmeans_hsv.fit(hsv_pixels)
            kmeans_lab.fit(lab_pixels)
            
            # Get dominant colors
            hsv_color = kmeans_hsv.cluster_centers_[0]
            lab_color = kmeans_lab.cluster_centers_[0]
            
            h, s, v = hsv_color
            l, a, b = lab_color
            
            # Enhanced color definitions
            COLOR_DEFS = {
                'white': {
                    'hsv': {'s_max': 30, 'v_min': 140},
                    'lab': {'l_min': 200, 'a_range': 15, 'b_range': 15}
                },
                'black': {
                    'hsv': {'s_max': 30, 'v_max': 50},
                    'lab': {'l_max': 50}
                },
                'red': [
                    {'h_min': 0, 'h_max': 10, 's_min': 100, 'v_min': 50},
                    {'h_min': 170, 'h_max': 180, 's_min': 100, 'v_min': 50}
                ],
                'tan': {
                    'hsv': {'h_min': 15, 'h_max': 35, 's_min': 30, 's_max': 150},
                    'lab': {'l_range': [130, 200], 'a_range': [0, 15], 'b_range': [10, 30]}
                }
            }
            
            # Improved tan/white differentiation
            if (COLOR_DEFS['tan']['hsv']['h_min'] <= h <= COLOR_DEFS['tan']['hsv']['h_max'] and
                COLOR_DEFS['tan']['hsv']['s_min'] <= s <= COLOR_DEFS['tan']['hsv']['s_max'] and
                COLOR_DEFS['tan']['lab']['l_range'][0] <= l <= COLOR_DEFS['tan']['lab']['l_range'][1] and
                COLOR_DEFS['tan']['lab']['a_range'][0] <= a <= COLOR_DEFS['tan']['lab']['a_range'][1] and
                COLOR_DEFS['tan']['lab']['b_range'][0] <= b <= COLOR_DEFS['tan']['lab']['b_range'][1]):
                self._log(f"Detected TAN: h={h:.0f}, s={s:.0f}, l={l:.0f}, a={a:.0f}, b={b:.0f}")
                return 'tan'
            
            # Check white/black
            if s < COLOR_DEFS['white']['hsv']['s_max'] and v >= COLOR_DEFS['white']['hsv']['v_min']:
                if (l >= COLOR_DEFS['white']['lab']['l_min'] and
                    abs(a) <= COLOR_DEFS['white']['lab']['a_range'] and
                    abs(b) <= COLOR_DEFS['white']['lab']['b_range']):
                    self._log(f"Detected WHITE: s={s:.0f}, v={v:.0f}, l={l:.0f}")
                    return 'white'
            
            if s < COLOR_DEFS['black']['hsv']['s_max'] and v <= COLOR_DEFS['black']['hsv']['v_max']:
                if l <= COLOR_DEFS['black']['lab']['l_max']:
                    self._log(f"Detected BLACK: s={s:.0f}, v={v:.0f}, l={l:.0f}")
                    return 'black'
            
            # Check red (special case)
            for red_range in COLOR_DEFS['red']:
                if (red_range['h_min'] <= h <= red_range['h_max'] and
                    s >= red_range['s_min'] and v >= red_range['v_min']):
                    self._log(f"Detected RED: h={h:.0f}, s={s:.0f}, v={v:.0f}")
                    return 'red'
            
            self._log(f"UNKNOWN color: h={h:.0f}, s={s:.0f}, v={v:.0f}, l={l:.0f}, a={a:.0f}, b={b:.0f}")
            return 'unknown'
            
        except Exception as e:
            self._log(f"Error in color detection: {e}", style='ERROR')
            return 'unknown'

    def extract_text(self, image):
        """Enhanced text extraction using multiple preprocessing methods"""
        results = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try different preprocessing methods
            methods = [
                ("Basic threshold", lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ("Adaptive threshold", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                ("Enhanced contrast", lambda img: cv2.threshold(cv2.convertScaleAbs(img, alpha=1.5, beta=0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
            ]
            
            configs = [
                '--psm 6 --oem 3',
                '--psm 7 --oem 3',
                '--psm 8 --oem 3'
            ]
            
            for method_name, process in methods:
                thresh = process(gray)
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(thresh, config=config)
                        result = self._parse_text(text)
                        if result[0]:
                            self._log(f"Found piece ID {result[0]} with count {result[1]} using {method_name}, config={config}")
                        results.append(result)
                    except Exception as e:
                        self._log(f"OCR failed with {method_name}, config={config}: {e}", style='WARNING')
            
            best_result = self._select_best_result(results)
            if best_result[0]:
                self._log(f"Selected best result: ID={best_result[0]}, count={best_result[1]}")
            return best_result
            
        except Exception as e:
            self._log(f"Error in text extraction: {e}", style='ERROR')
            self._log(traceback.format_exc(), style='ERROR')
            return None, None

    def _parse_text(self, text):
        """Parse text to extract piece ID and count with validation"""
        text = ''.join(text.split())
        
        # Look for piece ID (6 digits) and count pattern (Nx where N is number)
        import re
        piece_match = re.search(r'(\d{6})', text)
        count_match = re.search(r'(\d+)x', text)
        
        piece_id = piece_match.group(1) if piece_match else None
        count = count_match.group(1) if count_match else "1"
        
        # Validate piece ID format
        if piece_id:
            # Must be 6 digits and in our known pieces
            if piece_id in self.known_pieces:
                return piece_id, count
            # Check if it might be malformed but fixable
            elif len(piece_id) == 6 and piece_id.isdigit():
                # Look for similar known piece IDs
                similar_ids = [pid for pid in self.known_pieces.keys() 
                             if sum(a == b for a, b in zip(pid, piece_id)) >= 4]
                if similar_ids:
                    # Use the most similar ID
                    piece_id = similar_ids[0]
                    self._log(f"Corrected malformed piece ID to {piece_id}")
                    return piece_id, count
        
        return None, None
        
    def _select_best_result(self, results):
        """Select most likely correct result based on training data"""
        for piece_id, count in results:
            if piece_id and piece_id in self.known_pieces:
                return piece_id, count
        return None, None

    def validate_detection(self, piece_id, color, count, page):
        """Validate detection against ground truth data with detailed output"""
        if not self.training_mode:
            return True
            
        try:
            truth = self.ground_truth[
                (self.ground_truth['Page'] == page) & 
                (self.ground_truth['Shape'].astype(str) == str(piece_id))
            ]
            
            validation_result = {
                'piece_id': piece_id,
                'page': page,
                'detected': {'color': color, 'count': count},
                'ground_truth': None,
                'matches': {'color': False, 'count': False}
            }
            
            if len(truth) == 0:
                self._log("\n╔════ False Positive ════╗")
                self._log(f"║ Page: {page}")
                self._log(f"║ Piece ID: {piece_id}")
                self._log(f"║ Detected Color: {color}")
                self._log(f"║ Detected Count: {count}")
                self._log("║ Status: Not in ground truth")
                self._log("╚════════════════════════╝")
                return False
                
            truth_row = truth.iloc[0]
            validation_result['ground_truth'] = {
                'color': truth_row['Color'],
                'count': truth_row['Count']
            }
            
            # Validate color and count
            validation_result['matches']['color'] = truth_row['Color'].lower() == color.lower()
            validation_result['matches']['count'] = int(truth_row['Count']) == int(count)
            
            # Print validation results
            if validation_result['matches']['color'] and validation_result['matches']['count']:
                self._log("\n╔════ Perfect Match ════╗", style='SUCCESS')
            else:
                self._log("\n╔════ Partial Match ════╗", style='WARNING')
                
            self._log(f"║ Page: {page}")
            self._log(f"║ Piece ID: {piece_id}")
            self._log(f"║ Color: {color} {'✓' if validation_result['matches']['color'] else '✗'}")
            if not validation_result['matches']['color']:
                self._log(f"║ Expected Color: {truth_row['Color']}")
            self._log(f"║ Count: {count} {'✓' if validation_result['matches']['count'] else '✗'}")
            if not validation_result['matches']['count']:
                self._log(f"║ Expected Count: {truth_row['Count']}")
            self._log("╚════════════════════════╝")
            
            return all(validation_result['matches'].values())
            
        except Exception as e:
            self._log(f"Error validating detection: {e}", style='ERROR')
            self._log(traceback.format_exc(), style='ERROR')
            return False

    def detect_pieces(self, image, page_num):
        """Detect individual LEGO pieces in the image with improved contour filtering"""
        pieces = []
        
        try:
            # Convert to grayscale and LAB color space
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply bilateral filter to reduce noise while preserving edges
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Try multiple threshold methods
            methods = [
                ("Adaptive Gaussian", lambda img: cv2.adaptiveThreshold(
                    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)),
                ("Adaptive Mean", lambda img: cv2.adaptiveThreshold(
                    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)),
                ("Otsu", lambda img: cv2.threshold(
                    img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
            ]
            
            all_contours = []
            for method_name, threshold_func in methods:
                binary = threshold_func(gray)
                
                # Apply morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Find contours with hierarchy
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                
                self._log(f"Found {len(contours)} potential pieces with {method_name}")
                
                # Filter contours using hierarchy information
                if hierarchy is not None:
                    hierarchy = hierarchy[0]
                    for i, (contour, h) in enumerate(zip(contours, hierarchy)):
                        # Only process parent contours
                        if h[3] == -1:  # No parent
                            area = cv2.contourArea(contour)
                            if area > 1000:  # Min area threshold
                                all_contours.append(contour)
            
            # Remove duplicate contours with improved algorithm
            unique_contours = []
            used_regions = []
            
            for contour in sorted(all_contours, key=cv2.contourArea, reverse=True):
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w/2, y + h/2)
                area = w * h
                
                is_duplicate = False
                for region in used_regions:
                    rx, ry, rw, rh = region
                    r_center = (rx + rw/2, ry + rh/2)
                    
                    # Check center distance relative to piece size
                    max_dist = min(max(w, h), max(rw, rh)) * 0.5
                    if (abs(center[0] - r_center[0]) < max_dist and 
                        abs(center[1] - r_center[1]) < max_dist):
                        # Check area ratio to handle nested pieces
                        area_ratio = area / (rw * rh)
                        if 0.7 < area_ratio < 1.3:  # Allow 30% variation
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_contours.append(contour)
                    used_regions.append((x, y, w, h))
            
            self._log(f"\nProcessing {len(unique_contours)} unique pieces...")
            
            for i, contour in enumerate(tqdm(unique_contours)):
                try:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Add padding
                    padding = int(min(w, h) * 0.2)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    # Extract piece image
                    piece_img = image[y1:y2, x1:x2]
                    if piece_img.size == 0:
                        continue
                        
                    # Skip if piece is too small
                    if piece_img.shape[0] < 30 or piece_img.shape[1] < 30:
                        continue
                    
                    # Verify using LAB color space
                    piece_lab = cv2.cvtColor(piece_img, cv2.COLOR_BGR2LAB)
                    l_channel = piece_lab[:,:,0]
                    if np.mean(l_channel) > 240:  # Too bright, likely background
                        continue
                    
                    # Detect color
                    color = self.detect_color(piece_img)
                    if color == 'unknown':
                        continue
                    
                    # Extract text
                    piece_id, count = self.extract_text(piece_img)
                    
                    if piece_id:
                        pieces.append((piece_img, piece_id, color, count))
                
                except Exception as e:
                    self._log(f"Error processing contour {i}: {e}", style='WARNING')
                    continue
            
            return pieces
            
        except Exception as e:
            self._log(f"Error detecting pieces: {e}", style='ERROR')
            self._log(traceback.format_exc(), style='ERROR')
            return []

    def scan_pdf(self, pdf_path, output_dir):
        """Scan PDF and detect LEGO pieces with enhanced validation"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            self._log(f"Converting PDF to images: {pdf_path}")
            pages = convert_from_path(pdf_path, dpi=300)
            results = []
            
            # Extract page number from filename if possible
            import re
            page_num_match = re.search(r'page\s*(\d+)', pdf_path)
            page_num = int(page_num_match.group(1)) if page_num_match else 1
            
            stats = {
                'total_detected': 0,
                'perfect_matches': 0,
                'color_matches': 0,
                'count_matches': 0,
                'false_positives': 0,
                'detected_by_page': {},
                'total_ground_truth': 0,
                'missing_pieces': []
            }
            
            # Convert PIL image to OpenCV format
            page_cv = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            
            # Initialize stats for this page
            stats['detected_by_page'][page_num] = []
            
            # Load ground truth for validation
            gt_pieces = self.ground_truth[self.ground_truth['Page'] == page_num]
            valid_piece_ids = set(gt_pieces['Shape'].astype(str))
            
            self._log(f"\nProcessing page {page_num}")
            self._log(f"Found {len(valid_piece_ids)} valid pieces in ground truth")
            
            # Track detected pieces for duplicate prevention
            detected_pieces = set()
            
            # Detect pieces
            pieces = self.detect_pieces(page_cv, page_num)
            
            for piece_img, piece_id, color, count in pieces:
                if piece_id and piece_id not in detected_pieces:
                    piece_info = {
                        'page': page_num,
                        'shape_id': piece_id,
                        'color': color,
                        'count': count
                    }
                    
                    if self.validate_detection(piece_id, color, count, page_num):
                        # Create transparent background
                        gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
                        _, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                        b, g, r = cv2.split(piece_img)
                        rgba = [b, g, r, alpha]
                        dst = cv2.merge(rgba, 4)
                        
                        # Save with transparent background
                        output_path = os.path.join(output_dir, f"{piece_id}_{color}_{count}.png")
                        cv2.imwrite(output_path, dst)
                        results.append(piece_info)
                        stats['detected_by_page'][page_num].append(piece_info)
                        stats['perfect_matches'] += 1
                        detected_pieces.add(piece_id)
                    else:
                        stats['false_positives'] += 1
            
            # Find missing pieces
            for _, gt_row in gt_pieces.iterrows():
                piece_id = str(gt_row['Shape'])
                if piece_id not in detected_pieces:
                    stats['missing_pieces'].append({
                        'id': piece_id,
                        'color': gt_row['Color'],
                        'count': gt_row['Count']
                    })
            
            # Summarize page statistics
            self._summarize_page_stats(page_num, stats, gt_pieces)
            
            # Print overall summary
            total_gt = len(gt_pieces)
            stats['total_ground_truth'] = total_gt
            
            self._log("\nOverall Detection Summary:", style='SUCCESS')
            self._log(f"Total Ground Truth Pieces: {total_gt}")
            self._log(f"Total Detected: {len(results)}")
            self._log(f"Perfect Matches: {stats['perfect_matches']}")
            self._log(f"False Positives: {stats['false_positives']}")
            
            if total_gt > 0:
                accuracy = stats['perfect_matches'] / total_gt * 100
                self._log(f"Overall Accuracy: {accuracy:.1f}%")
            
            return results
            
        except Exception as e:
            self._log(f"Error scanning PDF: {e}", style='ERROR')
            self._log(traceback.format_exc(), style='ERROR')
            return []

    def _summarize_page_stats(self, page_num, stats, gt_pieces):
        """Print detailed page statistics with color breakdown"""
        total_detected = len(stats['detected_by_page'][page_num])
        total_gt = len(gt_pieces)
        
        self._log("\n╔═══════════════════════════════════════╗")
        self._log("║           Page Statistics              ║")
        self._log("╠═══════════════════════════════════════╣")
        self._log(f"║ Page Number: {page_num:<24} ║")
        self._log(f"║ Ground Truth Pieces: {total_gt:<16} ║")
        self._log(f"║ Detected Pieces: {total_detected:<18} ║")
        
        if total_gt > 0:
            recall = (total_detected / total_gt) * 100
            self._log(f"║ Recall: {recall:.1f}%{' ' * (24 - len(f'{recall:.1f}'))}║")
        
        # Color breakdown
        self._log("╠═══════════════════════════════════════╣")
        self._log("║           Color Breakdown             ║")
        
        color_stats = gt_pieces.groupby('Color')['Count'].sum()
        detected_colors = {}
        for piece in stats['detected_by_page'][page_num]:
            color = piece['color']
            detected_colors[color] = detected_colors.get(color, 0) + 1
        
        for color, count in color_stats.items():
            detected = detected_colors.get(color.lower(), 0)
            self._log(f"║ {color.lower():<8} : {detected}/{count:<16} ║")
        
        # Missing pieces
        self._log("╠═══════════════════════════════════════╣")
        self._log("║           Missing Pieces              ║")
        for piece in stats['missing_pieces']:
            self._log(f"║ • {piece['id']} ({piece['color'].lower()}, x{piece['count']:<2}){' ' * 15}║")
            
        self._log("╚═══════════════════════════════════════╝")