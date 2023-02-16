import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from collections import defaultdict

PAIRS = defaultdict(list)

BASE_IMAGE_PATH = Path("/home/irisdc01/training/object-tracking/feature-training-data")
random_seed = np.random.randint(0, 100000)
rng = np.random.default_rng(random_seed)


def preprocess_video_breeds(image_metadata: pd.DataFrame) -> pd.DataFrame:
    data = []
    columns = ["h+h", "b+b", "j+j", "h+b", "h+j", "b+j"]
    breed_count = image_metadata.groupby(["video_name", "breed"]).size()
    video_names = image_metadata['video_name'].unique()
    for video_name in video_names:
        checks = [True, True, True, True, True, True]
        black_count = breed_count.get((video_name, "black"))
        holstein_count = breed_count.get((video_name, "holstein-like"))
        jersey_count = breed_count.get((video_name, "jersey-like"))

        if jersey_count is None:
            checks[2] = False
            checks[4] = False
            checks[5] = False
        elif jersey_count < 2:
            checks[2] = False

        if black_count is None:
            checks[1] = False
            checks[3] = False
            checks[5] = False
        elif black_count < 2:
            checks[1] = False

        if holstein_count is None:
            checks[0] = False
            checks[3] = False
            checks[4] = False
        elif holstein_count < 2:
            checks[0] = False

        data.append(checks)
    video_data = pd.DataFrame(data, columns=columns, index=video_names)
    return video_data


def get_farm_and_video(image_metadata: pd.DataFrame, video_data: pd.DataFrame, combo_key: str) -> tuple:
    valid_videos = video_data.index[video_data[combo_key] == True].tolist()

    farm_identifiers = image_metadata.farm_identifier[image_metadata.video_name.isin(valid_videos)].unique()
    selected_farm = farm_identifiers[int(rng.uniform(0, len(farm_identifiers)))]

    valid_videos_for_farm = image_metadata.video_name[
        (image_metadata.farm_identifier == selected_farm) & (image_metadata.video_name.isin(valid_videos))].unique()

    selected_video = valid_videos_for_farm[rng.integers(0, len(valid_videos_for_farm))]

    return selected_farm, selected_video


def get_track(image_metadata: pd.DataFrame, video_choice: str, breed: str, number_of_tracks: int):
    track_list = image_metadata[(image_metadata.video_name == video_choice) & (image_metadata.breed == breed)]
    track_id = track_list.track_id.sample(n=number_of_tracks, random_state=rng).values
    return track_id


def create_negative_pairs(image_metadata: pd.DataFrame, video_data: pd.DataFrame,
                          number_of_pairs: int, first_breed: str, second_breed: str
                          ):

    combo_key = first_breed[0] + "+" + second_breed[0]
    for i in range(0, number_of_pairs):
        farm_identifier, video_name = get_farm_and_video(image_metadata, video_data, combo_key)

        track_id_by_video_and_breed = partial(get_track, image_metadata, video_name)
        if first_breed == second_breed:
            first_track_id, second_track_id = tuple(track_id_by_video_and_breed(first_breed, 2))
        else:
            first_track_id = int(track_id_by_video_and_breed(first_breed, 1))
            second_track_id = int(track_id_by_video_and_breed(second_breed, 1))

        PAIRS["farm_identifier"].append(farm_identifier)
        PAIRS["video_name"].append(video_name)
        PAIRS["first_track_id"].append(first_track_id)
        PAIRS["second_track_id"].append(second_track_id)

    return


def create_positive_pairs(image_metadata: pd.DataFrame, video_data: pd.DataFrame, number_of_pairs: int, breed: str):
    combo_key = breed[0] + "+" + breed[0]
    for i in range(0, number_of_pairs):
        farm_identifier, video_name = get_farm_and_video(image_metadata, video_data, combo_key)

        track_id_by_video_and_breed = partial(get_track, image_metadata, video_name)
        track_id = int(track_id_by_video_and_breed(breed, 1))

        PAIRS["farm_identifier"].append(farm_identifier)
        PAIRS["video_name"].append(video_name)
        PAIRS["first_track_id"].append(track_id)
        PAIRS["second_track_id"].append(track_id)

    return


def main():
    pairs_per_combo = 100
    image_metadata = pd.read_csv("csv/feature-training-data-all.csv")

    video_data = preprocess_video_breeds(image_metadata)

    create_negative_pairs_for_combination = partial(create_negative_pairs, image_metadata, video_data, pairs_per_combo)
    create_negative_pairs_for_combination("holstein-like", "black")
    create_negative_pairs_for_combination("holstein-like", "jersey-like")
    create_negative_pairs_for_combination("black", "jersey-like")
    create_negative_pairs_for_combination("holstein-like", "holstein-like")
    create_negative_pairs_for_combination("black", "black")
    create_negative_pairs_for_combination("jersey-like", "jersey-like")

    create_positive_pairs_for_combination = partial(create_positive_pairs, image_metadata, video_data, pairs_per_combo * 2)
    create_positive_pairs_for_combination("holstein-like")
    create_positive_pairs_for_combination("black")
    create_positive_pairs_for_combination("jersey-like")

    pd.DataFrame(PAIRS).to_csv("csv/pairs.csv", index=False)


if __name__ == '__main__':
    main()
