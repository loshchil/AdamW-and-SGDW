--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found here
--  https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Code modified for Shake-Shake by Xavier Gastaldi
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels
   local k = opt.widenFactor
   local forwardShake = opt.forwardShake
   local backwardShake = opt.backwardShake
   local shakeImage = opt.shakeImage
   local batchSize = opt.batchSize

   local function layer(block, nInputPlane, nOutputPlane, count, stride, forwardShake, backwardShake, shakeImage, batchSize)
      local s = nn.Sequential()

      if count < 1 then
        return s
      end

      s:add(block(nInputPlane, nOutputPlane, stride, forwardShake, backwardShake, shakeImage, batchSize))

      for i=2,count do
        s:add(block(nOutputPlane, nOutputPlane, 1, forwardShake, backwardShake, shakeImage, batchSize))
      end

      return s
   end

   -- Typically shareGradInput uses the same gradInput storage for all modules
   -- of the same type. This is incorrect for some SpatialBatchNormalization
   -- modules in this network b/c of the in-place CAddTable. This marks the
   -- module so that it's shared only with other modules with the same key
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      error('ImageNet is not yet implemented with Shake-Shake')
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(ShareGradInput(SBatchNorm(16), 'first'))
      model:add(layer(nn.ShakeShakeBlock, 16, 16*k, n, 1, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layer(nn.ShakeShakeBlock, 16*k, 32*k, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layer(nn.ShakeShakeBlock, 32*k, 64*k, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64*k):setNumInputDims(3))
      model:add(nn.Linear(64*k, 10))
   elseif opt.dataset == 'imagenet32' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' IMAGENET-32')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(ShareGradInput(SBatchNorm(16), 'first'))
      model:add(layer(nn.ShakeShakeBlock, 16, 16*k, n, 1, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layer(nn.ShakeShakeBlock, 16*k, 32*k, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layer(nn.ShakeShakeBlock, 32*k, 64*k, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64*k):setNumInputDims(3))
      model:add(nn.Linear(64*k, 1000))
   elseif opt.dataset == 'cifar100' then
      error('Cifar100 is not yet implemented with Shake-Shake')
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
