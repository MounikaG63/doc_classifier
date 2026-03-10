"""
Document Side Detection using OpenCV (Image-Based)
Detects front/back side of Aadhar, PAN, Voter ID, and Driving License
Based on image features: layout, colors, patterns, text regions
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class DocumentSideDetectorCV:
    """Detect document side using image analysis"""
    
    def __init__(self):
        pass
    
    def detect_photo_region(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image has a photo (front side indicator)
        Photos typically have skin tones and face-like features
        
        Returns: (has_photo, confidence)
        """
        try:
            # Convert to HSV for skin tone detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Broader skin tone range in HSV (more inclusive)
            # Hue: 0-20 (red/orange tones) + 160-180 (red wrap-around)
            lower_skin1 = np.array([0, 10, 60], dtype=np.uint8)
            upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
            
            lower_skin2 = np.array([160, 10, 60], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate percentage of skin-like pixels
            skin_pixels = cv2.countNonZero(mask)
            total_pixels = image.shape[0] * image.shape[1]
            skin_percentage = (skin_pixels / total_pixels) * 100
            
            # Front side typically has 2-40% skin tone (face area)
            # Lowered threshold to catch more photos
            has_photo = skin_percentage >= 2
            confidence = min(skin_percentage, 100)
            
            return has_photo, confidence
        except Exception as e:
            print(f"Error in photo detection: {e}")
            return False, 0
    
    def detect_text_regions(self, image: np.ndarray) -> Dict:
        """
        Detect text regions and their distribution
        Front: text scattered (name, DOB, etc.)
        Back: text concentrated in address block
        
        Returns: {
            'text_density': percentage of image with text,
            'text_concentration': how concentrated text is (0-1),
            'text_regions_count': number of text regions
        }
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get text regions
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Invert to get text as white
            binary = cv2.bitwise_not(binary)
            
            # Find contours (text regions)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours (noise)
            text_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            # Calculate text density
            text_pixels = cv2.countNonZero(binary)
            total_pixels = image.shape[0] * image.shape[1]
            text_density = (text_pixels / total_pixels) * 100
            
            # Calculate text concentration
            if len(text_contours) > 0:
                # Get bounding boxes
                boxes = [cv2.boundingRect(c) for c in text_contours]
                
                # Calculate center of mass for text regions
                centers = [(x + w//2, y + h//2) for x, y, w, h in boxes]
                
                # Calculate variance of centers (concentration)
                if len(centers) > 1:
                    centers_array = np.array(centers)
                    variance = np.var(centers_array)
                    # Normalize variance (lower variance = more concentrated)
                    concentration = 1 / (1 + variance / 10000)
                else:
                    concentration = 1.0
            else:
                concentration = 0
            
            return {
                'text_density': text_density,
                'text_concentration': concentration,
                'text_regions_count': len(text_contours)
            }
        except Exception as e:
            print(f"Error in text region detection: {e}")
            return {
                'text_density': 0,
                'text_concentration': 0,
                'text_regions_count': 0
            }
    
    def detect_color_distribution(self, image: np.ndarray) -> Dict:
        """
        Analyze color distribution
        Front: more varied colors (photo + text)
        Back: more uniform colors (mostly text on white)
        
        Returns: {
            'color_variance': how varied colors are,
            'dominant_color': most common color,
            'is_mostly_white': True if mostly white/light
        }
        """
        try:
            # Reshape image to list of pixels
            pixels = image.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # K-means clustering to find dominant colors
            k = 3
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            
            # Calculate color variance
            color_variance = np.var(centers)
            
            # Check if mostly white (back side indicator)
            white_threshold = 200
            white_pixels = np.sum((image[:,:,0] > white_threshold) & 
                                 (image[:,:,1] > white_threshold) & 
                                 (image[:,:,2] > white_threshold))
            total_pixels = image.shape[0] * image.shape[1]
            white_percentage = (white_pixels / total_pixels) * 100
            is_mostly_white = white_percentage > 60
            
            return {
                'color_variance': float(color_variance),
                'dominant_color': centers[0].tolist(),
                'is_mostly_white': is_mostly_white,
                'white_percentage': white_percentage
            }
        except Exception as e:
            print(f"Error in color distribution: {e}")
            return {
                'color_variance': 0,
                'dominant_color': [0, 0, 0],
                'is_mostly_white': False,
                'white_percentage': 0
            }
    
    def detect_edge_distribution(self, image: np.ndarray) -> Dict:
        """
        Detect edges and their distribution
        Front: edges scattered (photo + text)
        Back: edges concentrated in text areas
        
        Returns: {
            'edge_density': percentage of edges,
            'edge_concentration': how concentrated edges are
        }
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Calculate edge density
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = image.shape[0] * image.shape[1]
            edge_density = (edge_pixels / total_pixels) * 100
            
            # Calculate edge concentration
            # Divide image into quadrants and check edge distribution
            h, w = edges.shape
            quadrants = [
                edges[0:h//2, 0:w//2],
                edges[0:h//2, w//2:w],
                edges[h//2:h, 0:w//2],
                edges[h//2:h, w//2:w]
            ]
            
            quadrant_edges = [cv2.countNonZero(q) for q in quadrants]
            
            # Concentration: how uneven the distribution is
            if sum(quadrant_edges) > 0:
                concentration = max(quadrant_edges) / sum(quadrant_edges)
            else:
                concentration = 0
            
            return {
                'edge_density': edge_density,
                'edge_concentration': concentration
            }
        except Exception as e:
            print(f"Error in edge detection: {e}")
            return {
                'edge_density': 0,
                'edge_concentration': 0
            }
    
    def detect_side(self, image_path: str, doc_type: str) -> Dict:
        """
        Detect document side using image analysis
        Also detects if image has combined view (front + back)
        
        Args:
            image_path: Path to document image
            doc_type: 'aadhar', 'pan', 'voter_id', or 'driving_license'
        
        Returns:
            {
                'side': 'front' or 'back' or 'combined' or 'unknown',
                'confidence': 0-100,
                'details': {...}
            }
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'side': 'unknown',
                    'confidence': 0,
                    'details': {'error': 'Could not read image'}
                }
            
            # Check if image is combined view (front + back)
            is_combined = self.detect_combined_view(image)
            
            if is_combined:
                return {
                    'side': 'combined',
                    'confidence': 95,
                    'details': {
                        'is_combined': True,
                        'message': 'Image contains both front and back sides'
                    }
                }
            
            # Resize for faster processing
            height, width = image.shape[:2]
            if width > 1000 or height > 1000:
                scale = min(1000/width, 1000/height)
                image = cv2.resize(image, (int(width*scale), int(height*scale)))
            
            # Extract features
            has_photo, photo_conf = self.detect_photo_region(image)
            text_info = self.detect_text_regions(image)
            color_info = self.detect_color_distribution(image)
            edge_info = self.detect_edge_distribution(image)
            
            # Scoring logic
            front_score = 0
            back_score = 0
            
            # Photo is strong front indicator
            if has_photo:
                front_score += 40
            else:
                back_score += 20
            
            # Text concentration: front has scattered text, back has concentrated
            if text_info['text_concentration'] < 0.3:
                front_score += 30  # Scattered text = front
            elif text_info['text_concentration'] > 0.6:
                back_score += 30  # Concentrated text = back
            
            # Color variance: front has more varied colors
            if color_info['color_variance'] > 1000:
                front_score += 20
            else:
                back_score += 20
            
            # Mostly white: back side indicator
            if color_info['is_mostly_white']:
                back_score += 20
            else:
                front_score += 10
            
            # Edge concentration: front has scattered edges
            if edge_info['edge_concentration'] < 0.4:
                front_score += 10
            elif edge_info['edge_concentration'] > 0.6:
                back_score += 10
            
            # Determine side
            if front_score > back_score:
                side = 'front'
                confidence = min((front_score / (front_score + back_score)) * 100, 100)
            elif back_score > front_score:
                side = 'back'
                confidence = min((back_score / (front_score + back_score)) * 100, 100)
            else:
                side = 'unknown'
                confidence = 50
            
            # Threshold: if confidence too low, mark as unknown
            if confidence < 55:
                side = 'unknown'
            
            return {
                'side': side,
                'confidence': round(confidence, 2),
                'details': {
                    'has_photo': has_photo,
                    'text_density': round(text_info['text_density'], 2),
                    'text_concentration': round(text_info['text_concentration'], 2),
                    'color_variance': round(color_info['color_variance'], 2),
                    'is_mostly_white': color_info['is_mostly_white'],
                    'edge_density': round(edge_info['edge_density'], 2),
                    'edge_concentration': round(edge_info['edge_concentration'], 2),
                    'front_score': front_score,
                    'back_score': back_score
                }
            }
        
        except Exception as e:
            return {
                'side': 'unknown',
                'confidence': 0,
                'details': {'error': str(e)}
            }
    
    def detect_combined_view(self, image: np.ndarray) -> bool:
        """
        Detect if image has combined view (front + back side by side or stacked)
        Looks for two distinct document sections
        
        Returns: True if combined view detected
        """
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Check for horizontal dividing line (stacked vertically)
            # Divide image into top and bottom halves
            mid_y = height // 2
            top_half = gray[:mid_y, :]
            bottom_half = gray[mid_y:, :]
            
            # Calculate average brightness of each half
            top_brightness = np.mean(top_half)
            bottom_brightness = np.mean(bottom_half)
            
            # If both halves have similar brightness and content, likely combined
            brightness_diff = abs(top_brightness - bottom_brightness)
            
            # Check for horizontal line/separator
            edges = cv2.Canny(gray, 50, 150)
            mid_section = edges[max(0, mid_y-20):min(height, mid_y+20), :]
            horizontal_lines = np.sum(mid_section, axis=1)
            
            # Strong horizontal line indicates separator
            has_strong_horizontal = np.max(horizontal_lines) > width * 0.6
            
            # Method 2: Check for vertical dividing line (side by side)
            mid_x = width // 2
            left_half = gray[:, :mid_x]
            right_half = gray[:, mid_x:]
            
            left_brightness = np.mean(left_half)
            right_brightness = np.mean(right_half)
            
            # Check for vertical line/separator
            mid_section = edges[:, max(0, mid_x-20):min(width, mid_x+20)]
            vertical_lines = np.sum(mid_section, axis=0)
            
            # Strong vertical line indicates separator
            has_strong_vertical = np.max(vertical_lines) > height * 0.6
            
            # Detect combined if:
            # 1. Strong horizontal line (stacked) OR
            # 2. Strong vertical line (side-by-side) OR
            # 3. Extreme aspect ratio (width >> height or height >> width)
            
            aspect_ratio = width / height
            
            if has_strong_horizontal:
                return True
            
            if has_strong_vertical:
                return True
            
            # Extreme aspect ratios
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                return True
            
            # Check if image has two distinct document regions
            # by looking for two separate text blocks
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 10:  # Multiple text regions suggest combined view
                # Check if regions are in two distinct areas
                y_positions = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if h > 20:  # Significant height
                        y_positions.append(y)
                
                if len(y_positions) > 5:
                    y_positions.sort()
                    # Check if there's a gap in the middle (separator)
                    mid_point = height // 2
                    above_mid = sum(1 for y in y_positions if y < mid_point - 50)
                    below_mid = sum(1 for y in y_positions if y > mid_point + 50)
                    
                    if above_mid > 2 and below_mid > 2:
                        return True
            
            return False
        except Exception as e:
            print(f"Error in combined view detection: {e}")
            return False
