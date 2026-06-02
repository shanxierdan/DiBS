#!/usr/bin/env python3
"""
Sudoku Puzzles Scraper
Scrapes 4x4, 16x16, and 25x25 sudoku puzzles from online sources
"""

import requests
import re
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional
from bs4 import BeautifulSoup
import json

BASE_URL = "https://www.sudoku-puzzles.net"

SIZES = {
    "4x4": {"path": "sudoku-4x4", "difficulties": ["easy", "medium", "hard"], "grid_size": 4},
    "16x16": {"path": "sudoku-16x16", "difficulties": ["easy", "medium", "hard"], "grid_size": 16},
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def parse_sudoku_from_html(html: str, grid_size: int) -> Optional[Tuple[str, str]]:
    soup = BeautifulSoup(html, 'html.parser')

    puzzle_values = []
    solution_values = []

    inputs = soup.find_all('input', {'type': 'text'})

    if not inputs:
        tables = soup.find_all('table')
        for table in tables:
            cells = table.find_all('td')
            for cell in cells:
                text = cell.get_text(strip=True)
                if text and text.isdigit():
                    puzzle_values.append(text)

    if len(inputs) >= grid_size * grid_size:
        for inp in inputs[:grid_size * grid_size]:
            val = inp.get('value', '')
            puzzle_values.append(val if val else '0')

    if len(puzzle_values) == grid_size * grid_size:
        puzzle_str = ''.join(puzzle_values)
        return puzzle_str, None

    return None


def parse_sudoku_from_page(html: str, grid_size: int) -> Optional[str]:
    soup = BeautifulSoup(html, 'html.parser')

    puzzle_values = []

    inputs = soup.find_all('input')
    for inp in inputs:
        if inp.get('type') in ['text', 'hidden'] or inp.get('name', '').startswith('cell'):
            val = inp.get('value', '')
            if val:
                puzzle_values.append(val)
            else:
                puzzle_values.append('0')

    if len(puzzle_values) >= grid_size * grid_size:
        return ''.join(puzzle_values[:grid_size * grid_size])

    script_tags = soup.find_all('script')
    for script in script_tags:
        content = script.string
        if content and 'puzzle' in content.lower():
            matches = re.findall(r'[\d\.0]+', content)
            for match in matches:
                if len(match) >= grid_size * grid_size:
                    return match[:grid_size * grid_size]

    return None


def scrape_puzzle_page(session: requests.Session, url: str, grid_size: int, retry: int = 3) -> Optional[str]:
    for attempt in range(retry):
        try:
            response = session.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                puzzle = parse_sudoku_from_page(response.text, grid_size)
                if puzzle and len(puzzle) == grid_size * grid_size:
                    return puzzle
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return None


def scrape_size(session: requests.Session, size_name: str, config: dict, num_puzzles: int) -> List[str]:
    puzzles = []
    grid_size = config['grid_size']

    print(f"\nScraping {size_name} puzzles...")

    for difficulty in config['difficulties']:
        print(f"  Difficulty: {difficulty}")

        base_url = f"{BASE_URL}/{config['path']}/{difficulty}.html"

        for i in range(num_puzzles // len(config['difficulties'])):
            url = f"{BASE_URL}/{config['path']}/{difficulty}-{i+1}.html" if i > 0 else base_url

            puzzle = scrape_puzzle_page(session, url, grid_size)
            if puzzle:
                puzzles.append(puzzle)
                if len(puzzles) % 10 == 0:
                    print(f"    Collected: {len(puzzles)} puzzles")
            else:
                time.sleep(0.5)
                puzzle = scrape_puzzle_page(session, base_url, grid_size)
                if puzzle:
                    puzzles.append(puzzle)

            time.sleep(random.uniform(0.3, 0.8))

            if len(puzzles) >= num_puzzles:
                break

        if len(puzzles) >= num_puzzles:
            break

    return puzzles[:num_puzzles]


def generate_puzzles_fallback(size: int, num_puzzles: int) -> List[str]:
    import random
    import copy

    print(f"\nGenerating {num_puzzles} {size}x{size} puzzles (fallback)...")

    def is_valid(grid, row, col, num, box_size):
        for i in range(size):
            if grid[row][i] == num or grid[i][col] == num:
                return False
        br, bc = (row // box_size) * box_size, (col // box_size) * box_size
        for i in range(br, br + box_size):
            for j in range(bc, bc + box_size):
                if grid[i][j] == num:
                    return False
        return True

    def fill(grid, box_size, pos=0):
        if pos == size * size:
            return True
        row, col = pos // size, pos % size
        nums = list(range(1, size + 1))
        random.shuffle(nums)
        for num in nums:
            if is_valid(grid, row, col, num, box_size):
                grid[row][col] = num
                if fill(grid, box_size, pos + 1):
                    return True
                grid[row][col] = 0
        return False

    def create_puzzle(grid, target_givens):
        puzzle = copy.deepcopy(grid)
        positions = [(i, j) for i in range(size) for j in range(size)]
        random.shuffle(positions)
        removed = 0
        target = size * size - target_givens
        for r, c in positions:
            if removed >= target:
                break
            puzzle[r][c] = 0
            removed += 1
        return puzzle

    box_size = int(size ** 0.5)
    puzzles = []

    for i in range(num_puzzles):
        grid = [[0] * size for _ in range(size)]
        fill(grid, box_size)

        if size == 4:
            givens = random.randint(6, 10)
        elif size == 16:
            givens = random.randint(80, 120)
        else:
            givens = random.randint(200, 280)

        puzzle = create_puzzle(grid, givens)
        puzzle_str = ''.join(str(v) if v < 10 else chr(ord('A') + v - 10) for row in puzzle for v in row)
        puzzles.append(puzzle_str)

        if (i + 1) % 100 == 0:
            print(f"  Generated: {i+1}/{num_puzzles}")

    return puzzles


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape/generate generalized sudoku puzzles")
    parser.add_argument("--output", type=str, default="dataset/generalized_sudoku")
    parser.add_argument("--num-4x4", type=int, default=1000)
    parser.add_argument("--num-16x16", type=int, default=1000)
    parser.add_argument("--use-fallback", action="store_true", help="Use generator instead of scraper")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    if args.num_4x4 > 0:
        puzzles_4x4 = []
        if not args.use_fallback:
            puzzles_4x4 = scrape_size(session, "4x4", SIZES["4x4"], args.num_4x4)

        if len(puzzles_4x4) < args.num_4x4:
            print(f"  Scraped {len(puzzles_4x4)}, generating rest...")
            more = generate_puzzles_fallback(4, args.num_4x4 - len(puzzles_4x4))
            puzzles_4x4.extend(more)

        with open(output_dir / "sudoku_4x4_puzzles.txt", 'w') as f:
            f.write('\n'.join(puzzles_4x4))
        print(f"Saved {len(puzzles_4x4)} 4x4 puzzles")

    if args.num_16x16 > 0:
        puzzles_16x16 = []
        if not args.use_fallback:
            puzzles_16x16 = scrape_size(session, "16x16", SIZES["16x16"], args.num_16x16)

        if len(puzzles_16x16) < args.num_16x16:
            print(f"  Scraped {len(puzzles_16x16)}, generating rest...")
            more = generate_puzzles_fallback(16, args.num_16x16 - len(puzzles_16x16))
            puzzles_16x16.extend(more)

        with open(output_dir / "sudoku_16x16_puzzles.txt", 'w') as f:
            f.write('\n'.join(puzzles_16x16))
        print(f"Saved {len(puzzles_16x16)} 16x16 puzzles")

    print("\nDone!")


if __name__ == "__main__":
    main()
