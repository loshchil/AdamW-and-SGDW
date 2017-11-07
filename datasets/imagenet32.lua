--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--

local t = require 'datasets/transforms'

local M = {}
local ImageNet32Dataset = torch.class('resnet.ImageNet32Dataset', M)

function ImageNet32Dataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function ImageNet32Dataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function ImageNet32Dataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire IMAGENET-10 training set
local meanstd = {
   mean = {122.7, 116.65, 104.0},  -- Mean over channels for imagenet dataset
   std  = {255.0,  255.0,  255.0}, -- This is not computed, it will force pixels to take values from 0 to 1 (before substracting mean)
}

function ImageNet32Dataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ImageNet32Dataset
