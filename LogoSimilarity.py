import pandas as pd
import asyncio
import aiohttp
import aiofiles
import os
import time
import subprocess
import shutil
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
from pathlib import Path
import hashlib
import multiprocessing
from tqdm import tqdm
import xml.etree.ElementTree as ET
import argparse
import re
import math
from collections import Counter, defaultdict
import cv2
from io import BytesIO
import pickle
import functools

# Cache folder for storing intermediate results
CACHE_DIR = '.svg_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# == PART 1: LOGO DOWNLOADING ==

async def scrape_logo(session, domain):
    try:
        url = f"https://{domain}"
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
            soup = BeautifulSoup(html, 'html.parser')

            # Common logo selectors
            selectors = [
                ('link[rel="icon"]', 'href'),
                ('link[rel="shortcut icon"]', 'href'),
                ('link[rel="apple-touch-icon"]', 'href'),
                ('img[src*="logo"]', 'src'),
                ('img[class*="logo"]', 'src'),
                ('img[id*="logo"]', 'src'),
            ]

            for selector, attr in selectors:
                tag = soup.select_one(selector)
                if tag and tag.get(attr):
                    logo_url = urljoin(url, tag[attr])
                    return logo_url
    except Exception:
        return None
    return None

def get_logo_url(domain, fallback=False):
    return f"https://www.google.com/s2/favicons?domain={domain}&sz=256"

async def download_logo(session, domain, output_dir):
    filename = os.path.join(output_dir, f"{domain.replace('/', '_')}.png")
    if os.path.exists(filename):
        return

    # First try Google favicon
    for fallback in [False, True]:
        url = get_logo_url(domain, fallback)
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    if len(content) > 100:
                        async with aiofiles.open(filename, 'wb') as f:
                            await f.write(content)
                        return
        except Exception:
            continue

    # Try scraping the site
    logo_url = await scrape_logo(session, domain)
    if logo_url:
        try:
            async with session.get(logo_url, timeout=10) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    if len(content) > 100:
                        async with aiofiles.open(filename, 'wb') as f:
                            await f.write(content)
                        return
        except Exception:
            pass

    print(f"Failed: {domain}")
    return None

async def download_all(domains, output_dir, concurrency=50):
    failed_domains = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for domain in domains:
            tasks.append(download_logo(session, domain, output_dir))
        
        results = await asyncio.gather(*tasks)
        
        # Count failures
        for i, result in enumerate(results):
            if result is None:
                failed_domains.append(domains[i])
    
    return failed_domains

def download_logos_from_parquet(parquet_file, output_dir="logos", concurrency=50):
    """Download logos from domains in a parquet file"""
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and clean domains
    df = pd.read_parquet(parquet_file)
    
    if 'domain' not in df.columns:
        raise ValueError(f"Expected 'domain' column, but got: {df.columns.tolist()}")
    
    domains = (
        df['domain']
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .drop_duplicates()
    )
    
    domains = [d for d in domains if '.' in d and not d.startswith('localhost')]
    
    print(f"Total domains to process: {len(domains)}")
    
    # Run the downloads
    start = time.time()
    failed_domains = asyncio.run(download_all(domains, output_dir, concurrency))
    duration = time.time() - start
    
    print(f"\n✅ Download complete in {duration:.2f} seconds.")
    print(f"⚠️ Failed downloads: {len(failed_domains)}")
    
    # Save failed domains
    if failed_domains:
        with open("failed_domains.txt", "w") as f:
            f.write("\n".join(failed_domains))
        print("📁 Saved failed domains to failed_domains.txt")

    return output_dir

# == PART 2: SVG CONVERSION AND COMPARISON ==

@functools.lru_cache(maxsize=1024)
def extract_svg_features_cached(svg_path):
    """Cached version of extract_svg_features"""
    cache_file = os.path.join(CACHE_DIR, f"{hashlib.md5(svg_path.encode()).hexdigest()}_features.pkl")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass  # If loading fails, compute features
    
    # Compute features
    features = extract_svg_features(svg_path)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
    except:
        pass
        
    return features

def convert_to_bitmap(image_path, output_path, threshold=128):
    """Convert an image to a black and white bitmap suitable for potrace."""
    try:
        # Open the image first
        img = Image.open(image_path)
        
        # Get original resolution for grouping
        original_size = img.size
        
        # Convert palette images with transparency to RGBA first
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
            
        # Then convert to grayscale
        img = img.convert('L')
        
        # Upscale small images to preserve detail during thresholding
        if img.size[0] < 50 or img.size[1] < 50:
            img = img.resize((100, 100), Image.NEAREST)

        # Try adaptive thresholding using OpenCV
        try:
            import cv2
            img_cv = np.array(img)
            img_cv = cv2.adaptiveThreshold(
                img_cv, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=5,
                C=2
            )
            img = Image.fromarray(img_cv)
        except ImportError:
            # Fallback to fixed threshold if OpenCV isn't available
            img = img.point(lambda p: 255 if p > threshold else 0)

        img.save(output_path)
        return output_path, original_size
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def convert_to_svg(bitmap_path, svg_path):
    """Convert a bitmap to SVG using potrace."""
    try:
        subprocess.run(['potrace', bitmap_path, '-s', '-o', svg_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {bitmap_path} to SVG: {e}")
        return False
    except FileNotFoundError:
        print("Error: potrace command not found. Please install potrace.")
        return False

def extract_svg_features(svg_path):
    """
    Extract features from an SVG file for comparison purposes.
    Returns a dictionary of features that can be compared.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Get SVG namespace
        ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
        
        # OPTIMIZED: Extract only the most discriminative features
        features = {
            'path_count': 0,
            'color_count': 0,
            'colors': set(),
            'shape_distribution': Counter(),
            'path_data_hashes': [],  # Store hashes instead of full path data
            'bounding_box': (0, 0, 0, 0),
            'aspect_ratio': 1.0,
        }
        
        # Count different elements (simplified)
        all_elements = list(root.iter())
        element_tags = [elem.tag.replace(ns, '') for elem in all_elements]
        features['element_counts'] = Counter(element_tags)
        
        # Process paths more efficiently
        for path in root.findall('.//' + ns + 'path'):
            features['path_count'] += 1
            
            if 'd' in path.attrib:
                # Use hash of path data instead of storing the whole path
                path_data = path.attrib['d']
                path_hash = hashlib.md5(path_data.encode()).hexdigest()
                features['path_data_hashes'].append(path_hash)
                
                # Extract colors (simplified)
                if 'style' in path.attrib:
                    style = path.attrib['style']
                    color_match = re.search(r'(?:fill|stroke):(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', style)
                    if color_match:
                        features['colors'].add(color_match.group(1))
        
        features['color_count'] = len(features['colors'])
        
        # Get viewBox or width/height (unchanged)
        if 'viewBox' in root.attrib:
            viewbox = root.attrib['viewBox'].split()
            if len(viewbox) == 4:
                x, y, width, height = map(float, viewbox)
                features['bounding_box'] = (x, y, width, height)
                if height > 0:
                    features['aspect_ratio'] = width / height
        elif 'width' in root.attrib and 'height' in root.attrib:
            try:
                width = float(root.attrib['width'].replace('px', ''))
                height = float(root.attrib['height'].replace('px', ''))
                features['bounding_box'] = (0, 0, width, height)
                if height > 0:
                    features['aspect_ratio'] = width / height
            except ValueError:
                pass
        
        # Count common shapes
        for shape in ['rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon']:
            shape_elements = root.findall('.//' + ns + shape)
            if shape_elements:
                features['shape_distribution'][shape] = len(shape_elements)
        
        # Simplified logo detection heuristic
        is_logo = False
        if features['path_count'] > 0 and features['path_count'] < 15:
            if 'circle' in features['shape_distribution'] or 'ellipse' in features['shape_distribution']:
                is_logo = True

        features['is_logo'] = is_logo
        
        # Simplified image feature calculation
        return features
    
    except Exception as e:
        print(f"Error extracting features from {svg_path}: {e}")
        return None

def generate_svg_hash(svg_path):
    """Generate a hash of the SVG content, ignoring certain attributes."""
    try:
        # Check for cached hash
        cache_file = os.path.join(CACHE_DIR, f"{os.path.basename(svg_path)}.hash")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return f.read().strip()
    
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Normalize SVG for comparison 
        for elem in root.iter():
            # Remove attributes that might not be relevant for structural comparison
            for attr in ['id', 'class', 'style']:
                if attr in elem.attrib:
                    del elem.attrib[attr]
        
        # Generate hash from the normalized XML
        xml_str = ET.tostring(root, encoding='utf-8')
        hash_value = hashlib.sha256(xml_str).hexdigest()
        
        # Cache the hash
        with open(cache_file, 'w') as f:
            f.write(hash_value)
            
        return hash_value
    except Exception as e:
        print(f"Error generating hash for {svg_path}: {e}")
        return None

def compare_svgs(svg_path1, svg_path2, resolution=None):
    """
    Compare two SVGs and return a similarity score between 0 and 1.
    """
    # First try hash-based comparison for exact matches
    hash1 = generate_svg_hash(svg_path1)
    hash2 = generate_svg_hash(svg_path2)
    
    if hash1 is not None and hash2 is not None and hash1 == hash2:
        return 1.0
    
    # If hashes are different but similar, we can compute a hash similarity
    if hash1 and hash2:
        hash_similarity = sum(a == b for a, b in zip(hash1, hash2)) / len(hash1)
        # Early exit if hashes are very dissimilar (optimization)
        if hash_similarity < 0.5:  # Adjust threshold as needed
            return 0.2 * hash_similarity  # Return a low similarity score
    
    # If not a quick match, do feature-based comparison
    features1 = extract_svg_features_cached(svg_path1)
    features2 = extract_svg_features_cached(svg_path2)
    
    similarity = compare_svg_features(features1, features2, resolution)
    return similarity

def compare_path_hashes(hashes1, hashes2):
    """Compare path hashes with an efficient set-based approach."""
    if not hashes1 or not hashes2:
        return 0.0
    
    # Convert to sets for faster intersection/union operations
    set1 = set(hashes1)
    set2 = set(hashes2)
    
    # Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
        
    return intersection / union

def compare_svg_features(features1, features2, resolution=None):
    """
    Compare two SVG feature sets with simplified calculation.
    """
    if features1 is None or features2 is None:
        return 0.0
    
    # Quick pre-check: if path counts differ significantly, items are likely different
    path_count1 = features1.get('path_count', 0)
    path_count2 = features2.get('path_count', 0)
    path_ratio = min(path_count1, path_count2) / max(path_count1, path_count2) if max(path_count1, path_count2) > 0 else 0
    
    # Early exit if path counts are very different
    if path_ratio < 0.5 and abs(path_count1 - path_count2) > 3:
        return 0.2 * path_ratio  # Return a low similarity score
    
    # Compare aspect ratios
    ar1 = features1.get('aspect_ratio', 1.0)
    ar2 = features2.get('aspect_ratio', 1.0)
    ar_sim = 1.0 - min(1.0, abs(ar1 - ar2) / max(ar1, ar2)) if max(ar1, ar2) > 0 else 1.0
    
    # Early exit if aspect ratios are very different
    if ar_sim < 0.7:
        return 0.3 * ar_sim  # Return a low similarity score
    
    # Calculate individual feature similarities (only key features)
    similarities = {
        'path_count': path_ratio,
        'aspect_ratio': ar_sim,
    }
    
    # Compare path data hashes
    similarities['path_data'] = compare_path_hashes(
        features1.get('path_data_hashes', []), 
        features2.get('path_data_hashes', [])
    )
    
    # Compare shape distribution
    shape_sim = compare_counters(
        features1.get('shape_distribution', Counter()),
        features2.get('shape_distribution', Counter())
    )
    similarities['shape_distribution'] = shape_sim
    
    # Compare colors
    colors1 = features1.get('colors', set())
    colors2 = features2.get('colors', set())
    all_colors = colors1 | colors2
    if not all_colors:
        similarities['colors'] = 1.0
    else:
        similarities['colors'] = len(colors1 & colors2) / len(all_colors)
    
    # Simplified weighting system
    weights = {
        'path_count': 0.15,
        'aspect_ratio': 0.10,
        'path_data': 0.45,
        'shape_distribution': 0.20,
        'colors': 0.10
    }
    
    weighted_similarity = sum(
        similarities.get(feature, 0.0) * weight 
        for feature, weight in weights.items()
    )
    
    return weighted_similarity

def compare_counters(counter1, counter2):
    """Simplified counter comparison."""
    if not counter1 or not counter2:
        return 0.0
    
    # Get all keys
    all_keys = set(counter1.keys()) | set(counter2.keys())
    if not all_keys:
        return 1.0  # Both empty counters are considered identical
    
    # Calculate cosine similarity
    dot_product = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in all_keys)
    magnitude1 = math.sqrt(sum(v**2 for v in counter1.values()))
    magnitude2 = math.sqrt(sum(v**2 for v in counter2.values()))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def process_image(args):
    """Process a single image, converting it to SVG."""
    image_path, output_dir, temp_dir, threshold = args
    
    try:
        # Create unique filenames based on the original image name
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        bitmap_path = os.path.join(temp_dir, f"{base_name}.bmp")
        svg_path = os.path.join(output_dir, f"{base_name}.svg")
        
        # Skip if SVG already exists
        if os.path.exists(svg_path):
            # Get original resolution for existing SVG
            try:
                img = Image.open(image_path)
                original_size = img.size
                return image_path, svg_path, True, original_size
            except:
                return image_path, svg_path, True, None
        
        # Convert to bitmap
        result = convert_to_bitmap(image_path, bitmap_path, threshold)
        if result is None:
            return image_path, None, False, None
            
        bitmap_path, original_size = result
        
        # Convert to SVG
        success = convert_to_svg(bitmap_path, svg_path)
        
        # Remove temporary bitmap file
        if os.path.exists(bitmap_path):
            os.remove(bitmap_path)
            
        return image_path, svg_path, success, original_size
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, None, False, None

def create_resolution_buckets(image_metadata, strict_resolution=True):
    """
    Group SVGs into resolution buckets for comparison.
    OPTIMIZED: Uses histogram binning to create broader resolution groups
    """
    buckets = defaultdict(list)
    
    if strict_resolution:
        # Group by exact resolution
        for svg_path, size in image_metadata.items():
            if size:
                bucket_name = f"{size[0]}x{size[1]}"
                buckets[bucket_name].append(svg_path)
    else:
        # Create binned resolution buckets to reduce the number of buckets
        resolution_bins = [(0, 32), (33, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (1025, float('inf'))]
        
        for svg_path, size in image_metadata.items():
            if not size:
                continue
                
            width, height = size
            
            # Find appropriate width and height bins
            width_bin = next((f"{bin_start}-{bin_end}" for bin_start, bin_end in resolution_bins if bin_start <= width <= bin_end), "other")
            height_bin = next((f"{bin_start}-{bin_end}" for bin_start, bin_end in resolution_bins if bin_start <= height <= bin_end), "other")
            
            # Create a bucket based on binned dimensions
            bucket_name = f"{width_bin}x{height_bin}"
            buckets[bucket_name].append(svg_path)
            
            # Also add to aspect ratio buckets
            if width > 0 and height > 0:
                aspect = width / height
                if aspect < 0.8:
                    buckets["portrait"].append(svg_path)
                elif aspect > 1.2:
                    buckets["landscape"].append(svg_path)
                else:
                    buckets["square"].append(svg_path)
    
    # Create a "logo" bucket with potential logos
    logo_bucket = []
    for svg_path in image_metadata.keys():
        try:
            features = extract_svg_features_cached(svg_path)
            if features and features.get('is_logo', False):
                logo_bucket.append(svg_path)
        except:
            pass
    
    # Only create a logo bucket if significant
    if len(logo_bucket) >= 3 and not strict_resolution:
        buckets["logos"] = logo_bucket
    
    return buckets

def get_resolution_specific_threshold(resolution):
    """
    Return a customized similarity threshold based on image resolution.
    Small images need higher thresholds to avoid false positives.
    """
    if not resolution:
        return 0.65  # Default
        
    width, height = resolution
    
    # For tiny images (16x16, 32x32), use a very high threshold
    if width <= 32 and height <= 32:
        return 0.85
        
    # For small images (48x48, 64x64), use a high threshold
    if width <= 64 and height <= 64:
        return 0.75
        
    # For medium images, use a moderate threshold
    if width <= 128 and height <= 128:
        return 0.70
        
    # For large images, use the standard threshold
    return 0.65

def process_bucket(args):
    """Process a single bucket to find similar SVGs (for parallel execution)"""
    bucket_name, svg_files, resolution, similarity_threshold, max_group_size = args
    
    print(f"\nProcessing bucket {bucket_name} with {len(svg_files)} images")
    print(f"Using similarity threshold: {similarity_threshold:.2f} for resolution {resolution}")
    
    # Optimization: Hash-based pre-filtering
    # Group identical files by hash first to reduce comparisons
    hash_groups = defaultdict(list)
    for svg in svg_files:
        svg_hash = generate_svg_hash(svg)
        if svg_hash:
            hash_groups[svg_hash].append(svg)
    
    # Create initial groups from identical hashes
    initial_groups = [files for hash_val, files in hash_groups.items() if len(files) > 1]
    
    # Get unique representatives for non-identical SVGs
    processed_svgs = set()
    for group in initial_groups:
        processed_svgs.update(group)
    
    unique_svgs = [svg for svg in svg_files if svg not in processed_svgs]
    
    # Calculate similarities only between unique SVGs
    similarities = {}
    for i, svg1 in enumerate(unique_svgs):
        for j, svg2 in enumerate(unique_svgs):
            if i >= j:  # Only calculate upper triangle
                continue
            
            pair_key = (svg1, svg2)
            similarity = compare_svgs(svg1, svg2, resolution)
            similarities[pair_key] = similarity
    
    # Group similar images using connected components
    undirected_graph = defaultdict(set)
    
    for (svg1, svg2), similarity in similarities.items():
        if similarity >= similarity_threshold:
            undirected_graph[svg1].add(svg2)
            undirected_graph[svg2].add(svg1)
    
    # Find connected components (groups)
    visited = set()
    bucket_groups = []
    
    # First, add the hash-based identical groups
    bucket_groups.extend(initial_groups)
    
    # Add "visited" for all SVGs in initial groups
    for group in initial_groups:
        visited.update(group)
    
    # Find groups among remaining SVGs
    for svg in unique_svgs:
        if svg in visited:
            continue
            
        # BFS to find all connected nodes
        current_group = []
        queue = [svg]
        visited.add(svg)
        
        while queue:
            current = queue.pop(0)
            current_group.append(current)
            
            for neighbor in undirected_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Only add groups with at least 2 images
        if len(current_group) > 1:
            if max_group_size and len(current_group) > max_group_size:
                # Split overly large groups
                for i in range(0, len(current_group), max_group_size):
                    subgroup = current_group[i:i+max_group_size]
                    if len(subgroup) > 1:
                        bucket_groups.append(subgroup)
            else:
                bucket_groups.append(current_group)
    
    print(f"Found {len(bucket_groups)} groups in bucket {bucket_name}")
    return bucket_groups

def find_similar_images(image_metadata, base_similarity_threshold=0.65, max_group_size=None, strict_resolution=True):
    """
    Find similar SVGs using a multi-stage approach for better performance.
    """
    if not image_metadata:
        return []
    
    # Create buckets for comparison
    buckets = create_resolution_buckets(image_metadata, strict_resolution)
    
    print(f"Created {len(buckets)} buckets for comparison")
    for bucket_name, paths in buckets.items():
        print(f"  - {bucket_name}: {len(paths)} images")
    
    # Process each bucket with better parallelization
    all_groups = []
    
    # Process buckets in parallel
    bucket_results = []
    bucket_arguments = []
    
    for bucket_name, svg_files in sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True):
        if len(svg_files) <= 1:
            continue  # Skip buckets with only one image
        
        # Get resolution from bucket name if possible
        resolution = None
        if 'x' in bucket_name and not bucket_name.startswith(("logo", "portrait", "landscape", "square")):
            try:
                # Handle binned resolution format "start-end x start-end"
                width_part, height_part = bucket_name.split('x')
                if '-' in width_part:
                    # Take middle of bin range as representative value
                    width_start, width_end = map(int, width_part.split('-'))
                    width = (width_start + width_end) // 2
                else:
                    width = int(width_part)
                    
                if '-' in height_part:
                    height_start, height_end = map(int, height_part.split('-'))
                    height = (height_start + height_end) // 2
                else:
                    height = int(height_part)
                    
                resolution = (width, height)
            except:
                pass
        
        # Adjust threshold based on resolution
        similarity_threshold = get_resolution_specific_threshold(resolution)
        similarity_threshold = max(similarity_threshold, base_similarity_threshold)
        
        # Add to processing queue
        bucket_arguments.append((bucket_name, svg_files, resolution, similarity_threshold, max_group_size))
    
    # Process buckets in parallel (better work distribution)
    with multiprocessing.Pool(min(os.cpu_count(), len(bucket_arguments))) as pool:
        bucket_results = list(tqdm(
            pool.imap(process_bucket, bucket_arguments),
            total=len(bucket_arguments),
            desc="Processing buckets"
        ))
    
    # Collect all groups
    for groups in bucket_results:
        all_groups.extend(groups)
    
    # If using strict resolution, skip merging groups
    if strict_resolution:
        return all_groups
    
    # Otherwise merge overlapping groups
    merged_groups = merge_overlapping_groups(all_groups)
    return merged_groups

def merge_overlapping_groups(groups):
    """
    Merge groups that have overlapping elements using an efficient algorithm.
    """
    if not groups:
        return []
        
    print(f"Merging {len(groups)} initial groups...")
    
    # Use a Union-Find data structure for faster merging
    # Map each SVG to its group ID
    svg_to_group = {}
    
    # Assign initial group IDs
    for i, group in enumerate(groups):
        for svg in group:
            if svg in svg_to_group:
                # Found an overlap, merge groups
                old_group = svg_to_group[svg]
                
                # Update all SVGs in old group to new group
                for old_svg, g_id in list(svg_to_group.items()):
                    if g_id == old_group:
                        svg_to_group[old_svg] = i
            else:
                svg_to_group[svg] = i
    
    # Build final groups based on merged group IDs
    final_groups_map = defaultdict(list)
    for svg, group_id in svg_to_group.items():
        final_groups_map[group_id].append(svg)
    
    # Extract the groups
    final_groups = list(final_groups_map.values())
    
    print(f"Merged into {len(final_groups)} final groups")
    return final_groups

def create_similarity_report(groups, groups_dir, image_metadata, svgs_dir):
    """
    Create an HTML report with one index file and one file per group.
    HTML files are saved in groups_dir, but they reference SVG files in svgs_dir.
    Groups are sorted by size (largest first) in the index.
    
    Args:
        groups: List of groups, where each group is a list of SVG paths
        groups_dir: Directory where HTML reports will be saved
        image_metadata: Dictionary mapping SVG paths to their metadata
        svgs_dir: Directory where SVG files are located
    """
    try:
        # Sort groups by size (descending order)
        sorted_groups = sorted(groups, key=len, reverse=True)
        
        # Create index.html with sorted groups
        index_path = os.path.join(groups_dir, 'index.html')
        with open(index_path, 'w') as index_file:
            index_file.write('<!DOCTYPE html>\n<html><head><meta charset="UTF-8">\n')
            index_file.write('<title>Logo Similarity Report</title>\n')
            index_file.write('<style>body { font-family: sans-serif; } ul { list-style: none; } li { margin: 10px 0; }</style>\n')
            index_file.write('</head><body>\n')
            index_file.write(f'<h1>Logo Similarity Report</h1>\n')
            index_file.write(f'<p>{len(sorted_groups)} groups of similar logos (sorted by size)</p>\n')
            index_file.write('<ul>\n')

            for i, group in enumerate(sorted_groups, 1):
                group_file = f'group_{i}.html'
                index_file.write(f'<li><a href="{group_file}" target="_blank">Group {i} ({len(group)} logos)</a></li>\n')

            index_file.write('</ul>\n')
            index_file.write('</body></html>')

        print(f"Created index: {index_path}")

        # Create individual group HTML files
        for i, group in enumerate(sorted_groups, 1):
            group_path = os.path.join(groups_dir, f'group_{i}.html')
            with open(group_path, 'w') as f:
                f.write('<!DOCTYPE html><html><head><meta charset="UTF-8">\n')
                f.write(f'<title>Group {i}</title>\n')
                f.write('<style>body { font-family: sans-serif; } .svg { margin: 10px; display: inline-block; text-align: center; } img { max-width: 128px; max-height: 128px; }</style>\n')
                f.write('</head><body>\n')
                f.write(f'<h2>Group {i}</h2><p>{len(group)} logos</p><p><a href="index.html">Back to index</a></p>\n')

                for svg in group:
                    # Get just the filename from the full SVG path
                    svg_name = os.path.basename(svg)
                    
                    # Calculate the relative path from groups_dir to the SVG file
                    # For the simple structure we're using, we can go up one level and into svgs/
                    svg_relative_path = f"../svgs/{svg_name}"
                    
                    # For domains, extract the domain name from the filename
                    domain_name = svg_name.replace('_', '/').replace('.svg', '')
                    
                    f.write(f'<div class="svg"><img src="{svg_relative_path}" alt="{domain_name}"><br>{domain_name}</div>\n')

                f.write('</body></html>')

            print(f"Created group: {group_path}")

    except Exception as e:
        print(f"Error generating HTML report: {e}")
# == PART 3: MAIN FUNCTIONALITY ==

def convert_logos_to_svg(input_dir, output_dir="output_logos", temp_dir="temp", workers=None, threshold=128):
    """Convert downloaded logo images to SVG format"""
    
    # Create output structure
    output_dir = os.path.abspath(output_dir)
    svgs_dir = os.path.join(output_dir, 'svgs')
    temp_dir = os.path.join(output_dir, temp_dir)
    
    # Create directories
    os.makedirs(svgs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Find all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    image_files = [
        str(f) for f in Path(input_dir).glob('**/*') 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return None
    
    print(f"Found {len(image_files)} images to process")
    
    # Set number of workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    # OPTIMIZATION: Pre-check for existing SVGs to avoid redundant conversions
    existing_count = 0
    to_process = []
    
    for img in image_files:
        base_name = os.path.splitext(os.path.basename(img))[0]
        svg_path = os.path.join(svgs_dir, f"{base_name}.svg")
        
        if os.path.exists(svg_path):
            existing_count += 1
        else:
            to_process.append(img)
    
    print(f"Found {existing_count} already converted SVGs, {len(to_process)} need processing")
    
    # Process only the images that need conversion
    if to_process:
        # Process images in parallel
        process_args = [
            (img, svgs_dir, temp_dir, threshold) 
            for img in to_process
        ]
        
        # Use a smaller chunk size for better load balancing
        chunk_size = max(1, len(process_args) // (workers * 4))
        
        results = []
        with multiprocessing.Pool(workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_image, process_args, chunksize=chunk_size),
                total=len(process_args),
                desc="Converting images to SVG"
            ):
                results.append(result)
        
        # Count conversions
        new_successes = sum(1 for _, _, success, _ in results if success)
        new_failures = len(results) - new_successes
        
        print(f"New conversions: {new_successes} successful, {new_failures} failed")
    
    # Create metadata for ALL SVGs (both pre-existing and newly converted)
    # Get all SVG files in the output directory
    all_svg_files = list(Path(svgs_dir).glob('*.svg'))
    
    # Create metadata dictionary with resolution info
    image_metadata = {}
    
    # First add results from this run
    if to_process:
        for _, svg_path, success, size in results:
            if success and svg_path and size:
                image_metadata[svg_path] = size
    
    # Then add pre-existing SVGs by looking up their original images
    for svg_file in all_svg_files:
        svg_path = str(svg_file)
        if svg_path not in image_metadata:
            base_name = os.path.splitext(os.path.basename(svg_path))[0]
            # Try to find the original image
            for ext in image_extensions:
                img_path = os.path.join(input_dir, f"{base_name}{ext}")
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        image_metadata[svg_path] = img.size
                        break
                    except:
                        # If we can't open the image, still include the SVG with None size
                        image_metadata[svg_path] = None
    
    # Clean up temp directory if it's empty
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    return image_metadata, svgs_dir, output_dir

def compare_logo_svgs(image_metadata, output_dir, similarity_threshold=0.65, strict_resolution=True):
    """
    Compare SVG logos and find similar ones
    """
    if not image_metadata or len(image_metadata) < 2:
        print("Not enough SVGs to compare")
        return None
    
    # Create groups directory
    groups_dir = os.path.join(output_dir, 'groups')
    os.makedirs(groups_dir, exist_ok=True)
    
    # Get svgs_dir from the first key in image_metadata
    svgs_dir = os.path.dirname(next(iter(image_metadata.keys())))
    
    print("Comparing SVGs...")
    
    # Apply a more lenient threshold for logos
    base_threshold = similarity_threshold
    
    # Find similar groups with optimized algorithm
    start_time = time.time()
    similar_groups = find_similar_images(
        image_metadata,
        base_similarity_threshold=base_threshold,
        strict_resolution=strict_resolution
    )
    end_time = time.time()
    
    print(f"Comparison completed in {end_time - start_time:.2f} seconds")
    
    if similar_groups:
        print(f"Found {len(similar_groups)} groups of similar logos:")
        for i, group in enumerate(similar_groups, 1):
            print(f"Group {i}: {len(group)} logos")
                
        # Create a simple HTML report
        create_similarity_report(similar_groups, groups_dir, image_metadata, svgs_dir)
        return similar_groups
    else:
        print("No similar logos found.")
        return None

def process_logos(parquet_file=None, logo_dir=None, output_dir="output_logos", 
                similarity_threshold=0.65, cross_resolution=True, 
                clear_cache=False, workers=None):
    """
    Complete function to download, convert, and compare logos
    
    Args:
        parquet_file: Path to parquet file with domains (optional)
        logo_dir: Directory with existing logo files (optional)
        output_dir: Directory for output files
        similarity_threshold: Threshold for similarity comparison
        cross_resolution: Whether to compare logos across different resolutions
        clear_cache: Whether to clear the cache before running
        workers: Number of worker processes
        
    Returns:
        List of groups of similar logos
    """
    # Clear cache if requested
    if clear_cache and os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("Cache cleared.")
    
    # Step 1: Download logos if parquet file is provided
    if parquet_file:
        print("\n=== DOWNLOADING LOGOS ===")
        logo_dir = download_logos_from_parquet(parquet_file, 
                                              output_dir=os.path.join(output_dir, "logos"))
    
    # Check if we have a logo directory
    if not logo_dir:
        raise ValueError("Either parquet_file or logo_dir must be provided")
    
    # Step 2: Convert logos to SVG
    print("\n=== CONVERTING LOGOS TO SVG ===")
    image_metadata, svgs_dir, output_dir = convert_logos_to_svg(
        logo_dir, 
        output_dir=output_dir,
        workers=workers
    )
    
    # Step 3: Compare logos and find similar ones
    print("\n=== COMPARING LOGOS ===")
    similar_groups = compare_logo_svgs(
        image_metadata, 
        output_dir, 
        similarity_threshold=similarity_threshold,
        strict_resolution=not cross_resolution
    )
    
    # Print final statistics
    print(f"\nFinal Statistics:")
    print(f" - Total logos processed: {len(image_metadata)}")
    print(f" - Similar logo groups: {len(similar_groups) if similar_groups else 0}")
    
    # Calculate cache statistics
    if os.path.exists(CACHE_DIR):
        cache_files = list(Path(CACHE_DIR).glob('*'))
        cache_size_mb = sum(os.path.getsize(f) for f in cache_files) / (1024 * 1024)
        print(f" - Cache entries: {len(cache_files)}")
        print(f" - Cache size: {cache_size_mb:.2f} MB")
    
    return similar_groups

def main():
    """Command line interface for logo processing"""
    parser = argparse.ArgumentParser(description='Download, convert, and compare logos')
    
    # Source arguments (must provide one)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--parquet', help='Path to parquet file with domains')
    source_group.add_argument('--logo_dir', help='Directory with existing logo files')
    
    # Output arguments
    parser.add_argument('--output_dir', help='Directory for output files', default='output_logos')
    
    # Processing options
    parser.add_argument('--similarity', type=float, help='Similarity threshold (0-1)', default=0.8518)
    parser.add_argument('--cross_resolution', action='store_true', help='Compare logos across different resolutions')
    parser.add_argument('--clear_cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--workers', type=int, help='Number of worker processes', default=None)
    
    args = parser.parse_args()
    
    # Process logos
    process_logos(
        parquet_file=args.parquet,
        logo_dir=args.logo_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity,
        cross_resolution=args.cross_resolution,
        clear_cache=args.clear_cache,
        workers=args.workers
    )

if __name__ == "__main__":
    main()