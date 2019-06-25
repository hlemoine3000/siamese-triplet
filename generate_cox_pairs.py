# Implementation of pairs.txt from lfw dataset
# Section f: http://vis-www.cs.umass.edu/lfw/lfw.pdf
# More succint, less explicit: http://vis-www.cs.umass.edu/lfw/README.txt

import io
import os
import random
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Set, Tuple, cast
import re
import json

import utils

import numpy as np

Mismatch = Tuple[str, int, str, int]
Match = Tuple[str, int, int]
CommandLineArgs = Namespace


def write_pairs(fname: str,
                match_folds: List[List[Match]],
                mismatch_folds: List[List[Mismatch]],
                num_folds: int,
                num_matches_mismatches: int) -> None:
    metadata = '{}\t{}\n'.format(num_folds, num_matches_mismatches)
    with io.open(fname,
                 'w',
                 io.DEFAULT_BUFFER_SIZE,
                 encoding='utf-8') as fpairs:
        fpairs.write(metadata)
        for match_fold, mismatch_fold in zip(match_folds, mismatch_folds):
            for match in match_fold:
                line = '{}\t{}\t{}\n'.format(match[0], match[1], match[2])
                fpairs.write(line)
            for mismatch in mismatch_fold:
                line = '{}\t{}\t{}\t{}\n'.format(
                    mismatch[0], mismatch[1], mismatch[2], mismatch[3])
                fpairs.write(line)
        fpairs.flush()


def _split_people_into_folds(subject_list_path, num_folds: int) -> List[List[str]]:

    with open(subject_list_path) as f:
        names = f.read().splitlines()
    print('{} subjects divided in {} folds.'.format(len(names), num_folds))
    return [list(arr) for arr in np.array_split(names, num_folds)]


def _make_matches(image_dir: str,
                  people: List[str],
                  total_matches: int) -> List[Match]:
    matches = cast(Set[Match], set())
    curr_matches = 0
    while curr_matches < total_matches:
        person = random.choice(people)
        images = _clean_images(image_dir, person)
        if len(images) > 1:
            img1 = 0
            img2 = re.search(r'_...._(.*?).jpg', random.choice(images)).group(1)
            match = (person, img1, img2)
            if (img1 != img2) and (match not in matches):
                matches.add(match)
                curr_matches += 1
                if curr_matches % 1000 == 0:
                    print('matches: {}'.format(curr_matches))
    return sorted(list(matches), key=lambda x: x[0].lower())


def _make_mismatches(image_dir: str,
                     people: List[str],
                     total_matches: int) -> List[Mismatch]:
    mismatches = cast(Set[Mismatch], set())
    curr_matches = 0
    while curr_matches < total_matches:
        person1 = random.choice(people)
        person2 = random.choice(people)
        if person1 != person2:
            person1_images = _clean_images(image_dir, person1)
            person2_images = _clean_images(image_dir, person2)
            if person1_images and person2_images:
                img1 = 0
                img2 = re.search(r'_...._(.*?).jpg', random.choice(person2_images)).group(1)
                # if person1.lower() > person2.lower():
                #     person1, img1, person2, img2 = person2, img2, person1, img1
                mismatch = (person1, img1, person2, img2)
                if mismatch not in mismatches:
                    mismatches.add(mismatch)
                    curr_matches += 1
                    if curr_matches % 1000 == 0:
                        print('mismatches: {}'.format(curr_matches))
    return sorted(list(mismatches), key=lambda x: x[0].lower())


def _clean_images(base: str, folder: str):
    images = os.listdir(os.path.join(base, folder))
    images = [image for image in images if image.endswith(
        ".jpg") or image.endswith(".png")]
    return images


def _main(args: CommandLineArgs) -> None:

    print('COX pairs generator.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    people_folds = _split_people_into_folds(config.dataset.coxs2v.subject_list, config.dataset.coxs2v.num_fold)

    for video in config.dataset.coxs2v.video_list:

        print('Generating pairs for {}.'.format(video))
        image_dir = os.path.join(config.dataset.coxs2v.video_dir, video)
        pairs_file_name = os.path.join(config.dataset.coxs2v.video_pairs, video + '_pairs.txt')

        matches = []
        mismatches = []
        for i, fold in enumerate(people_folds):
            print('Fold {}/ {} subjects'.format(i, len(fold)))
            matches.append(_make_matches(image_dir,
                                         fold,
                                         config.num_matches_mismatches))
            mismatches.append(_make_mismatches(image_dir,
                                               fold,
                                               config.num_matches_mismatches))
        write_pairs(pairs_file_name,
                    matches,
                    mismatches,
                    config.dataset.coxs2v.num_fold,
                    config.num_matches_mismatches)


def _cli() -> None:
    args = parse_arguments()
    _main(args)


def parse_arguments() -> CommandLineArgs:
    parser = ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/coxpairs_config.json')

    return parser.parse_args()


if __name__ == '__main__':
    _cli()
