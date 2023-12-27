import logging
import time

from typing import Dict, List

import cv2
import easyocr
import numpy as np

from tqdm import tqdm

reader = easyocr.Reader(['en'], gpu=False)

logger = logging.getLogger(__name__)

CONFIDENCE_LEVEL = 0.4

INFO_TYPE = {-1: "information_not_found",
             0: "primary_fighter_info",
             1: "extensive_fighter_info",
             2: "fight_info",
             3: "adv",
             4: "winner"}

# patterns for finding white lines
PATTERNS_COORDINATES = (((slice(606, 683), slice(169, 172)), (slice(606, 683), slice(169 + 3, 172 + 3))),
                        ((slice(208, 212), slice(503, 590)), (slice(208 - 4, 212 - 4), slice(503, 590))),
                        ((slice(640, 680), slice(557, 560)), (slice(640, 680), slice(557 + 3, 560 + 3))),
                        ((slice(606, 656), slice(145, 148)), (slice(606, 656), slice(145 - 3, 148 - 3))))

# patterns for min/max of white lines
PATTERNS_MIN_MAX = np.array([[120, 250],
                             [135, 160],
                             [115, 150],
                             [115, 200]])

# coordinates of information on frames
INFO_COORDINATES = [
    [
        (slice(606, 683), slice(173, 377)), (slice(606, 643), slice(380, 654)), (slice(646, 683), slice(380, 654))
    ],
    [
        (slice(209, 250), slice(503, 590)), (slice(209, 250), slice(685, 767)),
        (slice(279, 329), slice(503, 590)), (slice(279, 329), slice(685, 767)),
        (slice(350, 400), slice(503, 590)), (slice(350, 400), slice(685, 767)),
        (slice(421, 470), slice(503, 590)), (slice(421, 470), slice(685, 767)),
        (slice(470, 587), slice(3, 496)), (slice(470, 587), slice(775, 1276)),
        (slice(590, 616), slice(10, 496)), (slice(590, 616), slice(775, 1276))
    ],
    [
        [(slice(641, 682), slice(562, 709))],
        [(slice(656, 667), slice(530, 541)), (slice(656, 667), slice(724, 735))],
        [(slice(628, 633), slice(617, 622)), (slice(628, 633), slice(630, 635)), (slice(628, 633), slice(643, 648))],
        [(slice(641, 882), slice(254, 525)), (slice(641, 882), slice(739, 1013))]
    ],
    [
        (slice(606, 655), slice(149, 435))
    ]
]


def parse_video(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    minutes = int(duration / 60)
    seconds = duration % 60
    logger.info(
        f"video stats: fps={fps}, frame_count={frame_count}, duration(S)={duration}, duration(M:S)={minutes}:{seconds}'")


    start_time = time.time()
    all_frames = {}
    counter_shot = 0
    frame_info: Dict = None

    cap = cv2.VideoCapture(video_path)

    current_second = 0
    for frame_idx in tqdm(range(frame_count)):
        success, frame = cap.read()
        if not success:
            logger.error(f"Unsuccessful reading frame {frame_idx}")
            break

        if frame_idx % fps == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_type = get_frame_type(image)
            if counter_shot > 0 and frame_type == -1:
                if frame_info:
                    frame_info['duration'] = counter_shot
                    new_sec = current_second - counter_shot
                    time_on_video = convert_time_m_s(new_sec)
                    all_frames[time_on_video] = frame_info
                counter_shot = 0

            if frame_type == 0 or frame_type == 1 or frame_type == 3:
                counter_shot += 1
                if counter_shot == 3:
                    frame_info = find_info_in_frame(image, frame_type)

            if frame_type == 2:
                frame_info = find_info_in_frame(image, frame_type)
                if frame_info:
                    time_on_video = convert_time_m_s(current_second)
                    all_frames[time_on_video] = frame_info
            current_second += 1

    end_time = time.time()
    logger.info(f"execution time: {end_time - start_time}s")
    return all_frames


def calc_difference_between_areas(slices, image):
    mean_area_white = image[slices[0][0], slices[0][1]].mean()
    mean_area_black = image[slices[1][0], slices[1][1]].mean()
    return mean_area_white - mean_area_black


# check if a frame presented
def get_frame_type(image) -> int:
    frame_type = -1
    for num in range(4):
        slices = PATTERNS_COORDINATES[num]
        min_for_frame = PATTERNS_MIN_MAX[num][0]
        max_for_frame = PATTERNS_MIN_MAX[num][1]
        if min_for_frame <= calc_difference_between_areas(slices, image) <= max_for_frame:
            frame_type = num
            break
    return frame_type


def find_info_in_frame(image, frame_type: int) -> Dict:
    frame_info = {}
    if frame_type == 0:
        frame_info = get_primary_fighter_info(image)
    elif frame_type == 1:
        frame_info = get_extensive_fighter_info(image)
    elif frame_type == 2:
        frame_info = get_fight_info(image)
    elif frame_type == 3:
        frame_info = get_winner_or_adv(image)

    return frame_info


def get_primary_fighter_info(image):
    frame_info = {}

    pfi_coordinates = INFO_COORDINATES[0]

    texts_array = reader.readtext(image[pfi_coordinates[0][0], pfi_coordinates[0][1]])
    name = from_texts_array_to_text(texts_array)

    if name:
        texts_array = reader.readtext(image[pfi_coordinates[1][0], pfi_coordinates[1][1]])
        record = from_texts_array_to_text(texts_array)
        if len(record) >= 1:
            record = record.split()[1]

        texts_array = reader.readtext(image[pfi_coordinates[2][0], pfi_coordinates[2][1]])
        place = from_texts_array_to_text(texts_array)

        frame_info['type'] = INFO_TYPE[0]
        frame_info['name'] = name
        frame_info['record'] = record
        frame_info['place'] = place

    return frame_info


def from_texts_array_to_text(texts_array: List) -> str:
    text = ''
    for i, t in enumerate(texts_array):
        if t[2] > CONFIDENCE_LEVEL:
            text += t[1]
            if i != len(texts_array) - 1:
                text += ' '
    return text


def get_extensive_fighter_info(image) -> Dict[str, str]:
    efi_coordinates = INFO_COORDINATES[1]

    frame_info = {'type': INFO_TYPE[1]}

    fighter_1 = {'color': 'red'}
    fighter_2 = {'color': 'blue'}

    texts_array = reader.readtext(image[efi_coordinates[8][0], efi_coordinates[8][1]])
    text = from_texts_array_to_text(texts_array)
    if text:
        fighter_1['name'] = text

        texts_array = reader.readtext(image[efi_coordinates[9][0], efi_coordinates[9][1]])
        fighter_2['name'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[10][0], efi_coordinates[10][1]])
        fighter_1['country'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[11][0], efi_coordinates[11][1]])
        fighter_2['country'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[0][0], efi_coordinates[0][1]], allowlist ='0123456789-')
        fighter_1['record'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[1][0], efi_coordinates[1][1]], allowlist ='0123456789-')
        fighter_2['record'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[2][0], efi_coordinates[2][1]], allowlist ='0123456789')
        fighter_1['age'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[3][0], efi_coordinates[3][1]], allowlist ='0123456789')
        fighter_2['age'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[4][0], efi_coordinates[4][1]], allowlist ='0123456789')
        fighter_1['height'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[5][0], efi_coordinates[5][1]], allowlist ='0123456789')
        fighter_2['height'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[6][0], efi_coordinates[6][1]], allowlist ='0123456789')
        fighter_1['weight'] = from_texts_array_to_text(texts_array)

        texts_array = reader.readtext(image[efi_coordinates[7][0], efi_coordinates[7][1]], allowlist ='0123456789')
        fighter_2['weight'] = from_texts_array_to_text(texts_array)

        frame_info['fighter_1'] = fighter_1
        frame_info['fighter_2'] = fighter_2
    else:
        frame_info = {}
    return frame_info


def get_time_from_text(text: str) -> str:
    time_str = ''
    for symbol in text:
        if symbol.isdigit():
            time_str += symbol
        else:
            time_str += ":"
    return time_str


def get_color_of_fighter(image, number_of_fighter: int) -> str:
    red_level_color = 40  # constant
    mean_color = image[
        INFO_COORDINATES[2][1][number_of_fighter][0], INFO_COORDINATES[2][1][number_of_fighter][1]].mean()
    if mean_color < red_level_color:
        return 'blue'
    return 'red'


def get_round_from_image(image, info_frame: List) -> int:
    pixel_color_level = 150  # constant for round number color
    round_num = 1  # default first
    second_round_idx = 1
    third_round_idx = 2
    if image[info_frame[2][third_round_idx][0], info_frame[2][third_round_idx][1]].mean() > pixel_color_level:
        round_num = 3
    elif image[info_frame[2][second_round_idx][0], info_frame[2][second_round_idx][1]].mean() > pixel_color_level:
        round_num = 2
    return round_num


def get_fight_info(image) -> Dict[str, str]:
    fi_coordinates = INFO_COORDINATES[2]

    frame_info = {'type': INFO_TYPE[2]}

    texts_array = reader.readtext(image[fi_coordinates[0][0][0], fi_coordinates[0][0][1]], allowlist ='0123456789:')
    text = from_texts_array_to_text(texts_array)
    if text:
        frame_info['time'] = get_time_from_text(text)
        frame_info['round'] = get_round_from_image(image, fi_coordinates)

        texts_array = reader.readtext(image[fi_coordinates[3][0][0], fi_coordinates[3][0][1]])
        text = from_texts_array_to_text(texts_array)
        if text:
            fighter_1 = {}
            fighter_2 = {}
            fighter_1['name'] = text
            texts_array = reader.readtext(image[fi_coordinates[3][1][0], fi_coordinates[3][1][1]])
            fighter_2['name'] = from_texts_array_to_text(texts_array)
            fighter_1['color'] = get_color_of_fighter(image, 0)
            fighter_2['color'] = get_color_of_fighter(image, 1)

            frame_info['fighter_1'] = fighter_1
            frame_info['fighter_2'] = fighter_2
    else:
        frame_info = {}

    return frame_info


def get_winner_or_adv(image):
    frame_info = {}
    info_frame = INFO_COORDINATES[3]
    texts_array = reader.readtext(image[info_frame[0][0], info_frame[0][1]])
    text = from_texts_array_to_text(texts_array)
    if text:
        if text.split()[0] == 'SUBSCRIBE':
            frame_info['type'] = 'advertising'
            frame_info['text'] = text
        else:
            frame_info['type'] = 'winner'
            frame_info['name'] = text
    return frame_info


def convert_time_m_s(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    time_on_video = f'{m:02d}:{s:02d}'
    return time_on_video
