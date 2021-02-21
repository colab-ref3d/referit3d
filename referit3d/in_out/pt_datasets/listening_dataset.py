import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from ...data_generation.nr3d import decode_stimulus_string


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False):

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, tokens, is_nr3d

    def prepare_objects(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        objs = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        clutter = [o for o in scan.three_d_objects if
                   o.instance_label != target_label]

        np.random.shuffle(clutter)  # randomly select part of clutter objects
        objs.extend(clutter)
        objs = objs[:self.max_distractors]
        objs.append(target)
        np.random.shuffle(objs)

        target_pos = np.random.randint(len(objs) + 1)
        objs.insert(target_pos, target)

        target_cls_msk = map(lambda o: o.instance_label == target_label, objs)
        target_cls_msk = np.array(target_cls_msk, dtype=np.bool)

        return objs, target_pos, target_cls_msk

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d = self.get_reference_data(index)

        # Make a context of distractors
        objs, target_pos, target_cls_msk = self.prepare_objects(scan, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in objs])

        # mark their classes
        res['class_labels'] = instance_labels_of_context(objs, self.max_context_size, self.class_to_idx)

        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(objs)] = target_cls_msk

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(objs):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(objs):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        return res


def make_data_loaders(dist_mgr, args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate')

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, dist_mgr, seed=seed)

    return data_loaders
