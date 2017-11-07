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

require 'torch'
require 'paths'
require 'optim'
require 'nn'
------Shake-Shake------
local json = require 'cjson'
------Shake-Shake------
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

------Shake-Shake------
local logfile = io.open(paths.concat(opt.save, 'log.txt'), 'w')

local function log(t)
   logfile:write('json_stats: '..json.encode(tablex.merge(t,opt,true))..'\n')
   logfile:flush()
end
------Shake-Shake------

local filename = "stat/"
filename = filename .. tostring(opt.irun) .. "_" .. opt.algorithmType .. "_Wide" .. tostring(opt.widenFactor) ..  "_Te" .. tostring(opt.Te) ..
	 	"_LR" .. tostring(opt.LR) .. "_LRdec" .. opt.LRdec .. "_WD" .. tostring(opt.weightDecay)  .. ".txt"

file = io.open(filename, "w")

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end

   file:write(string.format('%10.10f\t%10.10f\t%10.10f\t%10.10f\t%10.10f\n', epoch, trainLoss, trainTop1, testTop1, testTop5))
   file:flush()
   ------Shake-Shake------
   log{
      epoch = epoch,
      trainTop1 = trainTop1,
      trainTop5 = trainTop5,
      testTop1  = testTop1,
      testTop5 = testTop5,
      trainLoss = trainLoss,
   }
   ------Shake-Shake------

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end
file:close()
print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
