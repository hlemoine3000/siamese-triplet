
import os

class cox_data:
    def __init__(self,
                 cox_still_path,
                 cox_video_path,
                 cox_pairs_path):



        self.cox_still_path = cox_still_path
        self.cox_video_path = cox_video_path
        self.cox_pairs_path = cox_pairs_path

        self.subject_list, self.nb_folds = self._get_subject_list()
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds

    # Set up for training
    def get_dataset(self,
                    fold_list,
                    video_only=False):

        assert max(fold_list) < self.nb_folds, 'Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds)

        dataset = []

        fold_subject_list = self._extract_fold_list(fold_list)

        for i, subject in enumerate(fold_subject_list):
            subject_video_path = os.path.join(self.cox_video_path, subject)
            video_image_paths = facenet.get_image_paths(subject_video_path)

            if video_only:
                dataset.append(ImageClass(subject, video_image_paths))
            else:
                still_images_path = os.path.join(self.cox_still_path, subject + '_0000.JPG')
                assert os.path.isfile(still_images_path), 'Still image not found at {}'.format(still_images_path)

                dataset.append(COX_ImageClass(subject, video_image_paths, still_images_path))

            if not i % 100:
                print('Fetching subjects: {}/{}'.format(i, len(fold_subject_list)))

        return dataset



    def get_paths_from_file(self, subject_filename, max_subject=10, max_images_per_subject=10, tag=''):
        path_list = []
        label_list = []

        subjects_list = []
        with open(subject_filename, 'r') as f:
            for line in f.readlines()[1:]:
                subjects_list.append(line.strip())

        num_subject = 0
        for subject in subjects_list:

            # Get still image
            still_image_path = os.path.join(self.cox_still_path, subject + '_0000.JPG')
            path_list.append(still_image_path)
            label_list.append(subject + '_still' + tag)

            video_subject_dir = os.path.join(self.cox_video_path, subject)
            subject_images_list = os.listdir(video_subject_dir)

            images_per_subject = 0
            for subject_image in subject_images_list:
                path = os.path.join(video_subject_dir, subject_image)

                if os.path.exists(path):
                    path_list.append(path)
                    label_list.append(subject + tag)
                    images_per_subject += 1

                if images_per_subject >= max_images_per_subject:
                    break

            num_subject += 1
            if num_subject >= max_subject:
                break

        return path_list, label_list

    def _extract_fold_list(self, fold_list):

        list = []
        for fold in fold_list:
            upper_idx = fold * self.nb_subject_per_fold + self.nb_subject_per_fold
            lower_idx = fold * self.nb_subject_per_fold
            list += self.subject_list[lower_idx: upper_idx]

        return list

    def _get_subject_list(self):

        subject_list = []

        with open(self.cox_pairs_path, 'r') as f:

            nb_fold = f.readline().split('\t')[0]

            for line in f.readlines()[1:]:
                pair = line.strip().split()

                if len(pair) == 3:
                    if pair[0] not in subject_list:
                        subject_list.append(pair[0])

        return subject_list, int(nb_fold)