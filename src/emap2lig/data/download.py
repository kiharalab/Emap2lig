import requests
from loguru import logger


def get_contour_level(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/api/entry/map/{emdb_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code.

        json_data = response.json()

        # Extract contour information
        contours = json_data.get("map", {}).get("contour_list", {}).get("contour", None)

        if contours:
            for contour in contours:
                level = contour.get("level", None)
                if level is not None:
                    logger.info(f"Contour level of EMD-{emdb_id}: {level}")
                    return level

        logger.error(f"Failed to get contour level for EMDB ID {emdb_id}")
        return None

    except requests.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return None
