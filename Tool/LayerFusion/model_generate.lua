require('nn')
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options: model LeNet(LN) , ImageNet(IN)')
cmd:option('--model', '', 'input model to generate')
cmd:text()

opt = cmd:parse(arg or {})
print(opt)

if(opt.model == 'LN' or opt.model == 'LeNet') then
    channels = 1
    width = 32
    height = 32
    nstates = {8, 16, 10}
    filtsize = 5
    poolsize = 2
    stride = 2
    reshape = 400
    
    model = nn.Sequential()
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize,filtsize))
    --ReLU
    model:add(nn.Threshold(0,0))
    --stride and pooling size both are 2
    model:add(nn.SpatialMaxPooling(stride,stride,poolsize,poolsize))

    model:add(nn.SpatialConvolution(nstates[1],nstates[2],filtsize,filtsize))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(stride,stride,poolsize,poolsize))

    model:add(nn.Reshape(reshape))

    model:add(nn.Linear(reshape,nstates[3]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.LogSoftMax())
    model:type('torch.FloatTensor')

    torch.save('model_LN.net', model, 'ascii')

--[[else if(opt.model == 'ImageNet' or opt.model == 'IN') then
    
    channels = 3
    width = 224
    height = 224
    nstates = {96, 256, 384, 384, 256, 4096, 4096, 1000}
    filtsize = {11, 5, 3}
    poolsize = 2
    stride = 4
    reshape = 7*7*256
    
    model = nn.Sequential()
    pad = torch.floor(filtsize[1]/2)
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize[1],filtsize[1],stride, stride))
    model:add(nn.Threshold(0,0))
    
    pad = torch.floor(filtsize[2]/2)
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[1],nstates[2],filtsize[2],filtsize[2]))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
    
    pad = torch.floor(filtsize[3]/2)
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[2],nstates[3],filtsize[3],filtsize[3]))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
    
    pad = torch.floor(filtsize[3]/2)
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[3],nstates[4],filtsize[3],filtsize[3]))
    model:add(nn.Threshold(0,0))
    
    pad = torch.floor(filtsize[3]/2)
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[4],nstates[5],filtsize[3],filtsize[3]))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
    
   model:add(nn.Reshape(reshape))

    --[[model:add(nn.Linear(reshape,nstates[6]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[6], nstates[7]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[7], nstates[8]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.LogSoftMax())]]--
    --[[model:type('torch.FloatTensor')

    torch.save('model_IN2.net', model, 'ascii')


end]]--    

else if(opt.model == 'ImageNet_Paper' or opt.model == 'INP') then
    
    --bvlc Alexnet
    channels = 3
    width = 224
    height = 224
    nstates = {48, 128}
    filtsize = {11, 5, 3}
    poolsize = 2
    poolkernel = 3
    stride = 4
    reshape = 128*13*13
    
    model = nn.Sequential()
    pad = 2
    --conv1, relu, pool1
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize[1],filtsize[1],stride, stride))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv2, relu, pool2
    pad = 2
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[1],nstates[2],filtsize[2],filtsize[2]))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
   
    model:add(nn.Reshape(reshape))

    --[[model:add(nn.Linear(reshape,nstates[6]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[6], nstates[7]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[7], nstates[8]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.LogSoftMax())]]--
    model:type('torch.FloatTensor')
    
    Im = torch.FloatTensor(3,224,224)
    out = model:forward(Im)
    --print(out)
    torch.save('model_INP.net', model, 'ascii')


else if(opt.model == 'ImageNet_Paper2' or opt.model == 'INP2') then
    
    --bvlc Alexnet
    channels = 3
    width = 224
    height = 224
    nstates = {48}
    filtsize = {11}
    poolsize = 2
    poolkernel = 3
    stride = 4
    reshape = 48*27*27
    
    model = nn.Sequential()
    pad = 2
    --conv1, relu, pool1
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize[1],filtsize[1],stride, stride))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
   
    model:add(nn.Reshape(reshape))

    model:type('torch.FloatTensor')
    
    Im = torch.FloatTensor(3,224,224)
    out = model:forward(Im)
    --print(out)
    torch.save('model_INP2.net', model, 'ascii')


else if(opt.model == 'ImageNet' or opt.model == 'IN') then
    
    --bvlc Alexnet
    channels = 3
    width = 224
    height = 224
    nstates = {96, 256, 384, 384, 256, 4096, 4096, 1000}
    filtsize = {11, 5, 3}
    poolsize = 2
    poolkernel = 3
    stride = 4
    reshape = 6*6*256
    
    model = nn.Sequential()
    pad = 2
    --conv1, relu, pool1
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize[1],filtsize[1],stride, stride))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv2, relu, pool2
    pad = 2
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[1],nstates[2],filtsize[2],filtsize[2]))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv3, relu
    pad = 1
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[2],nstates[3],filtsize[3],filtsize[3]))
    model:add(nn.Threshold(0,0))
    
    --conv4, relu
    pad = 1
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[3],nstates[4],filtsize[3],filtsize[3]))
    model:add(nn.Threshold(0,0))
    
    --conv5, relu, pool3
    pad = torch.floor(filtsize[3]/2)
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[4],nstates[5],filtsize[3],filtsize[3]))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
   
    model:add(nn.Reshape(reshape))

    --[[model:add(nn.Linear(reshape,nstates[6]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[6], nstates[7]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[7], nstates[8]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.LogSoftMax())]]--
    model:type('torch.FloatTensor')
    
    Im = torch.FloatTensor(3,224,224)
    out = model:forward(Im)
    --print(out)
    torch.save('model_IN2.net', model, 'ascii')


else if(opt.model == 'VGG' or opt.model == 'vgg') then
    
    --VGG 
    channels = 3
    width = 224
    height = 224
    nstates = {64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 4096, 4096, 1000}
    filtsize = {3}
    poolsize = 2
    poolkernel = 2
    stride = 1
    reshape = 7*7*512
    
    model = nn.Sequential()
    pad = 1
    --conv1, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize[1],filtsize[1],stride, stride))
    model:add(nn.Threshold(0,0))
    
    --conv2, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[1],nstates[2],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))

    --First Pooling 
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv3, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[2],nstates[3],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))

    --conv4, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[3],nstates[4],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
   --Second Pooling      
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv5, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[4],nstates[5],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv6, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[5],nstates[6],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv7, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[6],nstates[7],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv8, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[7],nstates[8],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --Third Pooling      
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
   
    --conv9, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[8],nstates[9],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv10, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[9],nstates[10],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv11, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[10],nstates[11],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv12, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[11],nstates[12],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --Third Pooling      
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv13, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[12],nstates[13],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv14, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[13],nstates[14],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv15, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[14],nstates[15],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv16, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[15],nstates[16],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --Third Pooling      
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))

	
    model:add(nn.Reshape(reshape))
    model:evaluate()
    --[[model:add(nn.Linear(reshape,nstates[6]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[6], nstates[7]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[7], nstates[8]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.LogSoftMax())]]--
    model:type('torch.FloatTensor')
    
    --Im = torch.FloatTensor(3,224,224)
    --out = model:forward(Im)
    torch.save('model_VGG.net', model)


else if(opt.model == 'VGG2' or opt.model == 'vgg2') then
    
    --VGG 
    channels = 3
    width = 224
    height = 224
    nstates = {64, 64, 128, 128, 256, 256, 256, 256}
    filtsize = {3}
    poolsize = 2
    poolkernel = 2
    stride = 1
    reshape = 28*28*256
    
    model = nn.Sequential()
    pad = 1
    --conv1, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize[1],filtsize[1],stride, stride))
    model:add(nn.Threshold(0,0))
    
    --conv2, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[1],nstates[2],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))

    --First Pooling 
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv3, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[2],nstates[3],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))

    --conv4, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[3],nstates[4],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
   --Second Pooling      
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
    
    --conv5, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[4],nstates[5],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv6, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[5],nstates[6],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv7, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[6],nstates[7],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --conv8, relu
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[7],nstates[8],filtsize[1],filtsize[1]))
    model:add(nn.Threshold(0,0))
    
    --Third Pooling      
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))

    model:add(nn.Reshape(reshape))
    model:evaluate()
    --[[model:add(nn.Linear(reshape,nstates[6]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[6], nstates[7]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.Linear(nstates[7], nstates[8]))
    model:add(nn.Threshold(0,0))
    
    model:add(nn.LogSoftMax())]]--
    model:type('torch.FloatTensor')
    
    --Im = torch.FloatTensor(3,224,224)
    --out = model:forward(Im)
    torch.save('model_VGG2.net', model)


else if(opt.model == 'Test' or opt.model == 'T') then
    
    --bvlc Alexnet
    channels = 3
    width = 227
    height = 227
    nstates = {48, 128, 384, 384, 256, 4096, 4096, 1000}
    filtsize = {11, 5, 3}
    poolsize = 2
    poolkernel = 3
    stride = 4
    
	reshape = 128*13*13
    
    model = nn.Sequential()
    pad = 2
    --conv1, relu, pool1
    model:add(nn.SpatialConvolution(channels,nstates[1],filtsize[1],filtsize[1], stride, stride))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))

	--conv2, relu, pool2
    pad = 2
    model:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
    model:add(nn.SpatialConvolution(nstates[1],nstates[2],filtsize[2],filtsize[2]))
    model:add(nn.Threshold(0,0))
    model:add(nn.SpatialMaxPooling(poolkernel,poolkernel,poolsize,poolsize))
         
    model:add(nn.Reshape(reshape))
    model:type('torch.FloatTensor')
    
    Im = torch.FloatTensor(3,227,227)
    out = model:forward(Im)
    print(out:size())
    torch.save('model_test.net', model, 'ascii')


end

end

end

end

end

end

end
