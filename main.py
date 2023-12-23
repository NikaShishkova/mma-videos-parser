import logging

from video_parser import parse_video
from utils import parse_args, save_json

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    settings = parse_args()
    if not settings.path_video:
        logger.error('No path to video provided')
        exit(-1)
    if not settings.path_result:
        logger.info('No output path provided. Using default ./output.json')
        settings.path_result = './output.json'
    logger.info(f'Video parsing is starting')
    result = parse_video(settings.path_video)
    logger.info(f'Video parsing has completed')
    if result is None:
        result = {}
        logger.warning('No popup windows were found in the video file!')
    logger.info(f'Saving results to {settings.path_result}')
    try:
        save_json(json_data=result, json_path=settings.path_result)
    except Exception as e:
        logger.error(f'Saving failed: {str(e)}')
