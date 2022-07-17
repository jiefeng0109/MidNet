from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import cv2
import numpy as np
import torch.utils.data as data

from utils.image import color_aug
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import get_affine_transform, affine_transform


class CTDetDataset(data.Dataset):

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped_h = False
        flipped_v = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.8, 1.2, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip_h:
                flipped_h = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

            if np.random.random() < self.opt.flip_v:
                flipped_v = True
                img = img[::-1, :, :]
                c[1] = height - c[1] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        center_hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        left_hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        top_hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        right_hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        bottom_hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)

        center_wh = np.zeros((self.max_objs, 1), dtype=np.float32)
        left_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        top_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        right_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        bottom_wh = np.zeros((self.max_objs, 2), dtype=np.float32)

        center_ind = np.zeros((self.max_objs), dtype=np.int64)
        left_ind = np.zeros((self.max_objs), dtype=np.int64)
        top_ind = np.zeros((self.max_objs), dtype=np.int64)
        right_ind = np.zeros((self.max_objs), dtype=np.int64)
        bottom_ind = np.zeros((self.max_objs), dtype=np.int64)

        center_mask = np.zeros((self.max_objs), dtype=np.uint8)
        left_mask = np.zeros((self.max_objs), dtype=np.uint8)
        top_mask = np.zeros((self.max_objs), dtype=np.uint8)
        right_mask = np.zeros((self.max_objs), dtype=np.uint8)
        bottom_mask = np.zeros((self.max_objs), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            # bbox = self._coco_box_to_bbox(ann['bbox'])
            center = ann['center'][0:2]
            left, top, right, bottom = ann['left'][0:2], ann['top'][0:2], ann['right'][0:2], ann['bottom'][0:2]
            cls_id = int(self.cat_ids[ann['category_id']])

            if len(left) == 0 or len(top) == 0 or len(right) == 0 or len(bottom) == 0:
                continue

            if flipped_h:
                left, top, right, bottom = bottom, right, top, left
                center[0] = width - center[0] - 1
                left[0] = width - left[0] - 1
                top[0] = width - top[0] - 1
                right[0] = width - right[0] - 1
                bottom[0] = width - bottom[0] - 1

            if flipped_v:
                left, top, right, bottom = top, left, bottom, right
                center[1] = height - center[1] - 1
                left[1] = height - left[1] - 1
                top[1] = height - top[1] - 1
                right[1] = height - right[1] - 1
                bottom[1] = height - bottom[1] - 1

            center = affine_transform(center, trans_output)
            left = affine_transform(left, trans_output)
            top = affine_transform(top, trans_output)
            right = affine_transform(right, trans_output)
            bottom = affine_transform(bottom, trans_output)

            if not (0 < center[0] < output_w - 1 and 0 < center[1] < output_h - 1):
                continue

            w, h = min(ann['center'][2], ann['center'][3]), min(ann['center'][2], ann['center'][3])
            w, h = w / self.opt.down_ratio, h / self.opt.down_ratio

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            radius = self.opt.hm_gauss if self.opt.mse_loss else radius

            center = np.array(center, dtype=np.float32)
            center_int = center.astype(np.int32)
            draw_gaussian(center_hm[cls_id], center_int, radius)

            dis = []
            if 0 < left[0] < output_w - 1 and 0 < left[1] < output_h - 1:
                left = np.array(left, dtype=np.float32)
                left_int = left.astype(np.int32)
                draw_gaussian(left_hm[cls_id], left_int, radius)
                left_wh[k] = 1. * abs(left[0] - center[0]), 1. * abs(left[1] - center[1])
                left_ind[k] = left_int[1] * output_w + left_int[0]
                left_mask[k] = 1
                dis.append((left_wh[k][0] ** 2 + left_wh[k][1] ** 2) ** 0.5)

            if 0 < top[0] < output_w - 1 and 0 < top[1] < output_h - 1:
                top = np.array(top, dtype=np.float32)
                top_int = top.astype(np.int32)
                draw_gaussian(top_hm[cls_id], top_int, radius)
                top_wh[k] = 1. * abs(top[0] - center[0]), 1. * abs(top[1] - center[1])
                top_ind[k] = top_int[1] * output_w + top_int[0]
                top_mask[k] = 1
                dis.append((top_wh[k][0] ** 2 + top_wh[k][1] ** 2) ** 0.5)

            if 0 < right[0] < output_w - 1 and 0 < right[1] < output_h - 1:
                right = np.array(right, dtype=np.float32)
                right_int = right.astype(np.int32)
                draw_gaussian(right_hm[cls_id], right_int, radius)
                right_wh[k] = 1. * abs(right[0] - center[0]), 1. * abs(right[1] - center[1])
                right_ind[k] = right_int[1] * output_w + right_int[0]
                right_mask[k] = 1
                dis.append((right_wh[k][0] ** 2 + right_wh[k][1] ** 2) ** 0.5)

            if 0 < bottom[0] < output_w - 1 and 0 < bottom[1] < output_h - 1:
                bottom = np.array(bottom, dtype=np.float32)
                bottom_int = bottom.astype(np.int32)
                draw_gaussian(bottom_hm[cls_id], bottom_int, radius)
                bottom_wh[k] = 1. * abs(bottom[0] - center[0]), 1. * abs(bottom[1] - center[1])
                bottom_ind[k] = bottom_int[1] * output_w + bottom_int[0]
                bottom_mask[k] = 1
                dis.append((bottom_wh[k][0] ** 2 + bottom_wh[k][1] ** 2) ** 0.5)

            if dis == []:
                continue

            center_wh[k] = min(dis)
            center_ind[k] = center_int[1] * output_w + center_int[0]
            center_mask[k] = 1

            gt_det.append([right[0] - right_wh[k][0], right[1] - right_wh[k][1],
                           right[0] + right_wh[k][0], right[1] + right_wh[k][1], 1, cls_id])

        ret = {'input': inp,
               'center_hm': center_hm, 'center_wh': center_wh, 'center_ind': center_ind, 'center_mask': center_mask,
               'left_hm': left_hm, 'top_hm': top_hm, 'right_hm': right_hm, 'bottom_hm': bottom_hm,
               'left_wh': left_wh, 'top_wh': top_wh, 'right_wh': right_wh, 'bottom_wh': bottom_wh,
               'left_ind': left_ind, 'top_ind': top_ind, 'right_ind': right_ind, 'bottom_ind': bottom_ind,
               'left_mask': left_mask, 'top_mask': top_mask, 'right_mask': right_mask, 'bottom_mask': bottom_mask}

        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
