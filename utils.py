import argparse
import json

from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(prog='mma_parser', description='MMA translations records parser')
    parser.add_argument('-v', '--path_video', type=str, help='path to video file in mp4 format')
    parser.add_argument('-r', '--path_result', type=str, help='path to output json file')
    return parser.parse_args()


def save_json(json_path: str, json_data: Dict):
    with open(json_path, 'w') as output:
        json.dump(json_data, output, indent=2)
