--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file)
      print(m['data']:size())
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)

   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_1.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_2.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_3.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_4.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_5.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_6.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_7.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_8.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_9.t7',
      '/data/aad/imagenet/data/imagenet_resized_torch/box/train_data_batch_10.t7',
   })
   local testData = convertToTensor({
      '/data/aad/imagenet/data/imagenet_resized_torch/box/val_data.t7',
   })

   print(" | saving IMAGENET32 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
