require('nn')
require 'multiple_cone.lua'
require 'TimeMultiplex.lua'
require 'Parameters.lua'
require 'FPGAParameters.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--modelFile', '', 'input model file')
cmd:option('--W', '', 'input image Width')
cmd:option('--H', '', 'input image height')
cmd:option('--channels', '', 'input image channels')
cmd:option('--model', '', 'Recomp or Reuse')
cmd:option('--outfile', '', 'outfile')
cmd:option('--ascii', false, 'format')
cmd:option('--multicone', false, 'multicone evaluation')
cmd:option('--singlePyramid', false, 'single pyramid calculation')
cmd:option('--optimize', false, 'Optimization')
cmd:option('--plot', false, 'Plots when we get same executions cycle at each layer')

cmd:text()
opt = cmd:parse(arg or {})

if(opt.model == '' and opt.multicone == false) then
   print('Please give model ReComputation (C) or Reuse (U) or multicone status')
   return  
end

if opt.modelFile=='' then 
   print "Network unspecified (type it after the program's name) --modelFile" return
else
   --print('Loading: ' .. opt.modelFile)
   modelFile = opt.modelFile
   if(opt.ascii) then
     inputNet = torch.load(modelFile, 'ascii')
   else
     inputNet = torch.load(modelFile)
   end
   --print(inputNet)
end

--This is flag for results of multicone or story
local multicone_stats = 0
if(opt.multicone) then
  multicone_stats = 1
end
	
optimize = 0
if(opt.optimize) then
  optimize = 1
end
plot = false
if(opt.plot) then
 plot = true
end

local file, fileCycle
computation = 0
if(opt.model == 'Recomp' or opt.model == 'recomp' or opt.model == 'comp' or opt.model == 'C') then
   file = tostring(opt.outfile).."_Recomputation"
   computation = 1
else
   file = tostring(opt.outfile).. '_Reuse'
   fileCycle = tostring(opt.outfile).. '_ReuseCycle'
   if(opt.singlePyramid) then		
     fileSinglePyramid = tostring(opt.outfile).. '_SinglePyramid'
     outSinglePyramid = io.open(fileSinglePyramid, "w")
   end  
     
end

outfile = io.open(file, "w")
outfileCycle = io.open(fileCycle, "w")

function reshape_layer(inputNet) 
    for i = 1, #inputNet.modules do
       if(inputNet.modules[i].__typename == 'nn.Reshape') then 
         count = i
         break
       end
    end
    
    return count-1
end

--Finds out region for bottom pixels at each layer from bottom to top
function calculate_roi(inputNet, count, pixel1, shift, top)
   
    local start = 1
    if(top~=nil and top ~=0) then
      start = top 
    end

    local r, c, tab
    local region = {}
    region[count+1] = pixel1[count+1]
    for i = count, start, -1 do
       if(inputNet.modules[i].__typename == 'nn.SpatialMaxPooling') then
          tab = {}
          r = pixel1[i+1]['tl'][1] * inputNet.modules[i].dH
          c = pixel1[i+1]['tl'][2] * inputNet.modules[i].dW
          tab['tl'] = {r,c}
          r = pixel1[i+1]['br'][1] * inputNet.modules[i].dH + inputNet.modules[i].kH -1 
          c = pixel1[i+1]['br'][2] * inputNet.modules[i].dW + inputNet.modules[i].kW -1
          tab['br'] = {r,c}
          pixel1[i] = tab
          region[i] = tab
          
       else if (inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
          tab = {}
          r = pixel1[i+1]['tl'][1] * inputNet.modules[i].dH - torch.floor(inputNet.modules[i].kH/2)
          c = pixel1[i+1]['tl'][2] * inputNet.modules[i].dW - torch.floor(inputNet.modules[i].kW/2)
          tab['tl'] = {r,c}
          r = pixel1[i+1]['br'][1] * inputNet.modules[i].dH + torch.floor(inputNet.modules[i].kH/2) 
          c = pixel1[i+1]['br'][2] * inputNet.modules[i].dW + torch.floor(inputNet.modules[i].kW/2) 
          tab['br'] = {r,c}
          pixel1[i] = tab
          region[i] = tab
        else if (inputNet.modules[i].__typename == 'nn.SpatialZeroPadding') then
          tab = {}
          r = pixel1[i+1]['tl'][1] 
          c = pixel1[i+1]['tl'][2] 
          
          --[[if(r < inputNet.modules[i].pad_t) then
            r = r + inputNet.modules[i].pad_t
          end
          
	  if (c < inputNet.modules[i].pad_l) then
            c = c + inputNet.modules[i].pad_l
          end]]--

          if(r < 0) then
            r = r + inputNet.modules[i].pad_t
          end
          
	  if (c < 0) then
            c = c + inputNet.modules[i].pad_l
          end
          

          tab['tl'] = {r,c}
          
          r = pixel1[i+1]['br'][1]     
          c = pixel1[i+1]['br'][2]  
       
         if (i == 1) then
		if(r >= H-1) then
		    r = H-1
		 end
		 if(c >= W-1) then
		    c = W-1	
		 end
         else

		 if(r >= inputNet.modules[i-1].output:size()[3]-1) then
		    r = inputNet.modules[i-1].output:size()[3]-1
		 end
		 if(c >= inputNet.modules[i-1].output:size()[2]-1) then
		    c = inputNet.modules[i-1].output:size()[2]-1
		 end
         end
 
          tab['br'] = {r,c}
          pixel1[i] = tab      
          region[i] = tab
       else
          tab = {}
          local r = pixel1[i+1]['tl'][1] 
          local c = pixel1[i+1]['tl'][2]
          tab['tl'] = {r,c}
          local r = pixel1[i+1]['br'][1] 
          local c = pixel1[i+1]['br'][2]
          tab['br'] = {r,c}
          pixel1[i] = tab
	  region[i] = tab         
       end   
       end
      end
    end

   if(shift) then
     for i = start, count do
        pixel1[i]['tl'][1] = pixel1[i]['tl'][1] + shift[i]['y']
        pixel1[i]['tl'][2] = pixel1[i]['tl'][2] + shift[i]['x']
      
        pixel1[i]['br'][1] = pixel1[i]['br'][1] + shift[i]['y']
        pixel1[i]['br'][2] = pixel1[i]['br'][2] + shift[i]['x'] 
     end
  end
	    
  return region  
end    

-- Caluculate shifts in boundary region
function calculate_shift(inputNet, count, pixel1)
    local r,c, tab
    pixel1 = calculate_roi(inputNet, count, pixel1, shift)

    local shift = {}
    for i = 1, count do      
      shift_x = 0 - pixel1[i]['tl'][1]
      shift_y = 0 - pixel1[i]['tl'][2]
      dir={}
      dir['x'] = shift_x
      dir['y'] = shift_y            
      shift[i] = dir            
    end
       
   return shift    
end

function ReUse_calculate_left(inputNet, pixel, pixel_left, count)

  local rows, cols, lrows, lcols, NRows, NCols, NewRows, NewCols
  NewRows = pixel_left[1]['br'][1] - pixel_left[1]['tl'][1] + 1
  NewCols = pixel[1]['br'][2] - pixel_left[1]['br'][2]
  	
  --outfile:write(string.format('comparision between pixel (%d, %d) and (%d, %d) Sliding Window %d %s direction\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2],pixel2[count+1]['tl'][1], pixel2[count+1]['tl'][2], SW, dir))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2], pixel1[1]['tl'][1], pixel1[1]['tl'][2], pixel1[1]['br'][1], pixel1[1]['br'][2]))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], pixel[1]['tl'][1], pixel[1]['tl'][2], pixel[1]['br'][1], pixel[1]['br'][2]))  

  local total_reuse_mem = 0; local_reuse_mem = 0;
  if(pixel_left[1]['br'][2] > pixel[1]['tl'][2]) then
    --print('Input image overlapping region:'..'('..pixel2[1]['tl'][1]..','..pixel2[1]['tl'][2]..')'..','..'('..pixel1[1]['br'][1]..','..pixel1[1]['br'][2]..')\n\n')
  else
    print('No region to reuse or recompute')
    return
  end

  --outfile:write(string.format('Reuse Region for each layer\n\n')) 
  local multiplications, additions, comparison, orows, ocols
  i = 1
  while(i <= count) do
      rows = math.max(0, pixel_left[i]['br'][1] - pixel_left[i]['tl'][1] + 1)
      cols = math.max(0, pixel_left[i]['br'][2] - pixel[i]['tl'][2] + 1)
      NRows = pixel_left[i+1]['br'][1] - pixel_left[i+1]['tl'][1] + 1
      NCols = pixel[i+1]['br'][2] - pixel_left[i+1]['br'][2]
      orows = pixel_left[i]['br'][1] - pixel_left[i]['tl'][1] + 1
      
      if(rows == 0 or cols == 0) then
	 outfile:write(string.format("No Overlapping below this layer\n\n"))
         break
      end
      outfile:write(string.format('\nLayer %d: %s\n\n', i, tostring(inputNet.modules[i])))         
      
      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
	ocols = inputNet.modules[i].kW - inputNet.modules[i].dW
	--outfile:write(string.format('Layer %d: %s\n\n', i, tostring(inputNet.modules[i]))) 
        
	if(i == 1) then 	  
          outfile:write(string.format('Load input image data: %d X %d X %d, or %d bytes\n\n', opt.channels, NewRows, NewCols, opt.channels*NewRows*NewCols*4)) 
          outfile:write(string.format('Reuse:\nReuse Input image region stored by left pixel: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ocols, opt.channels*orows*ocols*4))
 
          outfile:write(string.format("Store:\nStore Input Image Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", opt.channels, NewRows, inputNet.modules[i].kW - 1, opt.channels * NewRows  * (inputNet.modules[i].kW - 1) * 4, opt.channels, inputNet.modules[i].kH - 1, NewCols, opt.channels * (inputNet.modules[i].kH - 1) * NewCols * 4))
          
          local_reuse_mem = opt.channels*orows*ocols*4     	  	     
          total_reuse_mem = total_reuse_mem + local_reuse_mem 
        	
	else
          -- i-1 layer is combined and used for convolution	  	  
	  outfile:write(string.format('Reuse:\nReUse data stored by left pixel at layer %d: %s %d X %d X %d or %d bytes\n\n', i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], orows, ocols, inputNet.modules[i-1].output:size()[1] * orows* ocols* 4))       
	  
           
          outfile:write(string.format("Store:\nStore layer %d: %s Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], rows, inputNet.modules[i].kW - 1, inputNet.modules[i-1].output:size()[1] * rows * (inputNet.modules[i].kW - 1) * 4, inputNet.modules[i-1].output:size()[1], inputNet.modules[i].kH - 1, NCols, inputNet.modules[i-1].output:size()[1] * (inputNet.modules[i].kH - 1) * NCols * 4))
	  
	  
	  local_reuse_mem = inputNet.modules[i-1].output:size()[1] * orows* ocols* 4
          total_reuse_mem = total_reuse_mem + local_reuse_mem
	end
           outfile:write(string.format('Total Memory Reused at this layer: %d bytes\n\n', local_reuse_mem)) 
	   multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * NRows * NCols * inputNet.modules[i].kW * inputNet.modules[i].kH
           additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * NRows * NCols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1)
           --outfile:write(string.format("\nComputation: \ninputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, InputRows = %d, InputCols = %d\n\n", inputNet.modules[i].nInputPlane, inputNet.modules[i].nOutputPlane, inputNet.modules[i].kW, inputNet.modules[i].kH, inputNet.modules[i].dH, inputNet.modules[i].dW, NRows, NCols))
           --outfile:write(string.format('Computation for Pixel2: NewRows = %d, NewCols = %d, mul = %d add = %d\n\n', NRows, NCols, multiplications, additions))          

        i = i + 1
      else if(inputNet.modules[i].__typename == 'nn.SpatialZeroPadding' and inputNet.modules[i+1].__typename == 'nn.SpatialConvolution') then
	  
          ocols = math.floor((inputNet.modules[i+1].kW/2)) * 2 --+ inputNet.modules[i].pad_l 
          NRows = pixel_left[i+2]['br'][1] - pixel_left[i+2]['tl'][1] + 1
    	  NCols = pixel[i+2]['br'][2] - pixel_left[i+2]['br'][2]
	  	
	  if(i == 1) then 
	     outfile:write(string.format('Load input image data: %d X %d X %d, or %d bytes\n\n', opt.channels, NewRows, NewCols, opt.channels*NewRows*NewCols*4)) 
             outfile:write(string.format('Reuse:\nReuse Input image region stored by left pixel: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ocols, opt.channels*orows*ocols*4)) 
             
	     outfile:write(string.format("Store:\nStore Input Image Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", opt.channels, NewRows, inputNet.modules[i+1].kW - 1, opt.channels * NewRows  * (inputNet.modules[i+1].kW - 1) * 4, opt.channels, inputNet.modules[i+1].kH - 1, NewCols, opt.channels * (inputNet.modules[i+1].kH - 1) * NewCols * 4))
            
             local_reuse_mem = opt.channels*orows*ocols*4
             total_reuse_mem = total_reuse_mem + local_reuse_mem  	       
       	
	  else	  	  
	     outfile:write(string.format('Reuse:\nReUse data stored by left pixel at layer %d: %s %d X %d X %d or %d bytes\n\n',  i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], orows, ocols, inputNet.modules[i-1].output:size()[1] * orows* ocols* 4))  
 
             outfile:write(string.format("Store:\nStore layer %d: %s Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], rows, inputNet.modules[i+1].kW - 1, inputNet.modules[i-1].output:size()[1] * rows * (inputNet.modules[i+1].kW - 1) * 4, inputNet.modules[i-1].output:size()[1], inputNet.modules[i+1].kH - 1, NCols, inputNet.modules[i-1].output:size()[1] * (inputNet.modules[i+1].kH - 1) * NCols * 4))
	     
             local_reuse_mem = inputNet.modules[i-1].output:size()[1] * orows* ocols* 4
  	     total_reuse_mem = total_reuse_mem + local_reuse_mem        
	  end      
             outfile:write(string.format('Layer %d: %s\n\n', i+1, tostring(inputNet.modules[i+1])))  

	    outfile:write(string.format('Total Memory Reused at this layer: %d bytes\n\n', local_reuse_mem)) 
       	   
 	   multiplications = inputNet.modules[i+1].nOutputPlane * inputNet.modules[i+1].nInputPlane * NRows * NCols * inputNet.modules[i+1].kW * inputNet.modules[i+1].kH
           additions = inputNet.modules[i+1].nOutputPlane * inputNet.modules[i+1].nInputPlane * NRows * NCols * (inputNet.modules[i+1].kW* inputNet.modules[i+1].kH - 1)
           --outfile:write(string.format("Computation:\ninputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, InputRows = %d, InputCols = %d\n\n",inputNet.modules[i+1].nInputPlane, inputNet.modules[i+1].nOutputPlane, inputNet.modules[i+1].kW, inputNet.modules[i+1].kH, inputNet.modules[i+1].dH, inputNet.modules[i+1].dW, NRows, NCols))
           --outfile:write(string.format('New Computation for Pixel2: NewRows = %d, NewCols = %d, mul = %d add = %d\n\n', NRows, NCols, multiplications, additions))          
	 
         i = i + 2 
     else
         i = i + 1
     end 
     
    end         
  end   
  outfile:write(string.format('Total Memory Reused for this pixel at all layer: %d bytes\n\n', total_reuse_mem)) 
  
end


function ReUse_calculate_up(inputNet, pixel, pixel_up, count)

  local rows, cols, lrows, lcols, NRows, NCols, NewRows, NewCols
  local right = 0  -- Right or Down 
  local SW
  NewRows = pixel[1]['br'][1] - pixel_up[1]['br'][1]
  NewCols = pixel_up[1]['br'][2] - pixel_up[1]['tl'][2] + 1
  SW = pixel_up[1]['br'][1] - pixel_up[1]['tl'][1] + 1
  local total_reuse_mem = 0; local_reuse_mem = 0;	
  --outfile:write(string.format('comparision between pixel (%d, %d) and (%d, %d) Sliding Window %d %s direction\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2],pixel2[count+1]['tl'][1], pixel2[count+1]['tl'][2], SW, dir))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2], pixel1[1]['tl'][1], pixel1[1]['tl'][2], pixel1[1]['br'][1], pixel1[1]['br'][2]))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], pixel[1]['tl'][1], pixel[1]['tl'][2], pixel[1]['br'][1], pixel[1]['br'][2]))  


  if(pixel_up[1]['br'][2] > pixel[1]['tl'][2]) then
    --print('Input image overlapping region:'..'('..pixel2[1]['tl'][1]..','..pixel2[1]['tl'][2]..')'..','..'('..pixel1[1]['br'][1]..','..pixel1[1]['br'][2]..')\n\n')
  else
    print('No region to reuse or recompute')
    return
  end

  --outfile:write(string.format('Reuse Region for each layer\n\n')) 
  local multiplications, additions, comparison, orows, ocols
  i = 1
  while(i <= count) do
      rows = math.max(0, pixel_up[i]['br'][1] - pixel[i]['tl'][1])
      cols = math.max(0, pixel[i]['br'][2] - pixel_up[i]['tl'][2] + 1)
      NRows = pixel[i+1]['br'][1] - pixel_up[i+1]['br'][1]
      NCols = pixel_up[i+1]['br'][2] - pixel_up[i+1]['tl'][2] + 1
      ocols = pixel_up[i]['br'][2] - pixel_up[i]['tl'][2] + 1 
      
      if(rows == 0 or cols == 0) then
	 outfile:write(string.format("No Overlapping below this layer\n\n"))
         break
      end
      outfile:write(string.format('Layer %d: %s\n\n', i, tostring(inputNet.modules[i])))         
      
      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
	orows = inputNet.modules[i].kH - inputNet.modules[i].dH 
      	--outfile:write(string.format('Layer %d: %s\n\n', i, tostring(inputNet.modules[i]))) 
        
	if(i == 1) then 	  
          local_reuse_mem = opt.channels*orows*ocols*4
          outfile:write(string.format('Load input image data: %d X %d X %d, or %d bytes\n\n', opt.channels, NewRows, NewCols, opt.channels*NewRows*NewCols*4))
          outfile:write(string.format('Reuse:\nReuse Input image region stored by above pixel: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ocols, opt.channels*orows*ocols*4))

          
	 outfile:write(string.format("Store:\nStore Input Image Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", opt.channels, NRows, inputNet.modules[i].kW - 1, opt.channels * NRows  * (inputNet.modules[i].kW - 1) * 4, opt.channels, inputNet.modules[i].kH - 1, cols, opt.channels * (inputNet.modules[i].kH - 1) * cols * 4))
 
          total_reuse_mem = total_reuse_mem + local_reuse_mem           	  	              	
	else
          -- i-1 layer is combined and used for convolution	  	  
	 outfile:write(string.format('Reuse:\nReUse data stored by above pixel at layer %d: %s %d X %d X %d or %d bytes\n\n', i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], orows, ocols, inputNet.modules[i-1].output:size()[1] * orows* ocols* 4))   
         
         outfile:write(string.format("Store:\nStore layer %d: %s Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], NRows, inputNet.modules[i].kW - 1, inputNet.modules[i-1].output:size()[1] * NRows * (inputNet.modules[i].kW - 1) * 4, inputNet.modules[i-1].output:size()[1], inputNet.modules[i].kH - 1, cols, inputNet.modules[i-1].output:size()[1] * (inputNet.modules[i].kH - 1) * cols * 4))          

         local_reuse_mem = inputNet.modules[i-1].output:size()[1] * orows* ocols* 4               
         total_reuse_mem = total_reuse_mem + local_reuse_mem           	  	              	
	end
         
           outfile:write(string.format('Total Memory Reused at this layer: %d bytes\n\n', local_reuse_mem)) 
               
	   multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * NRows * NCols * inputNet.modules[i].kW * inputNet.modules[i].kH
           additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * NRows * NCols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1)
           --outfile:write(string.format("\nComputation: \ninputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, InputRows = %d, InputCols = %d\n\n", inputNet.modules[i].nInputPlane, inputNet.modules[i].nOutputPlane, inputNet.modules[i].kW, inputNet.modules[i].kH, inputNet.modules[i].dH, inputNet.modules[i].dW, NRows, NCols))
           --outfile:write(string.format('Computation for Pixel: NewRows = %d, NewCols = %d, mul = %d add = %d\n\n', NRows, NCols, multiplications, additions))          

        i = i + 1
      else if(inputNet.modules[i].__typename == 'nn.SpatialZeroPadding' and inputNet.modules[i+1].__typename == 'nn.SpatialConvolution') then
	
            
             orows = math.floor((inputNet.modules[i+1].kH/2)) * 2 --+ inputNet.modules[i].pad_t
             NRows = pixel[i+2]['br'][1] - pixel_up[i+2]['br'][1]
	     NCols = pixel_up[i+2]['br'][2] - pixel_up[i+2]['tl'][2] + 1
	     local crows, ccols
             
	    if(i == 1) then 

             outfile:write(string.format('Load input image data: %d X %d X %d, or %d bytes\n\n', opt.channels, NewRows, NewCols, opt.channels*NewRows*NewCols*4)) 
	
             outfile:write(string.format('Reuse:\nReuse Input image region stored by above pixel: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ocols, opt.channels*orows*ocols*4)) 
             

	     outfile:write(string.format("Store:\nStore Input Image Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", opt.channels, NewRows, inputNet.modules[i+1].kW - 1, opt.channels * NewRows  * (inputNet.modules[i+1].kW - 1) * 4, opt.channels, inputNet.modules[i+1].kH - 1, NewCols, opt.channels * (inputNet.modules[i+1].kH - 1) * NewCols * 4))


             local_reuse_mem = opt.channels*orows*ocols*4               
             total_reuse_mem = total_reuse_mem + local_reuse_mem           	  		  	              	
	   else
             -- i has a result of i-1
             crows = pixel[i]['br'][1] - pixel_up[i]['br'][1]
             ccols = pixel[i]['br'][2] - pixel[i]['tl'][2] + 1
	  	  
	     outfile:write(string.format('Reuse:\nReUse data stored by above pixel at layer %d: %s %d X %d X %d or %d bytes\n\n',  i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], orows, ocols, inputNet.modules[i-1].output:size()[1] * orows* ocols* 4))
             
	     outfile:write(string.format("Store:\nStore layer %d: %s Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], crows, inputNet.modules[i+1].kW - 1, inputNet.modules[i-1].output:size()[1] * crows * (inputNet.modules[i+1].kW - 1) * 4, inputNet.modules[i-1].output:size()[1], inputNet.modules[i+1].kH - 1, ccols, inputNet.modules[i-1].output:size()[1] * (inputNet.modules[i+1].kH - 1) * ccols * 4))          
	     
	     local_reuse_mem = inputNet.modules[i-1].output:size()[1] * orows* ocols* 4               
             total_reuse_mem = total_reuse_mem + local_reuse_mem           	  	
	   end      
	     outfile:write(string.format('Total Memory Reused at this layer: %d bytes\n\n', local_reuse_mem)) 
             outfile:write(string.format('Layer %d: %s\n\n', i+1, tostring(inputNet.modules[i+1]))) 
        
       	    multiplications = inputNet.modules[i+1].nOutputPlane * inputNet.modules[i+1].nInputPlane * NRows * NCols * inputNet.modules[i+1].kW * inputNet.modules[i+1].kH
            additions = inputNet.modules[i+1].nOutputPlane * inputNet.modules[i+1].nInputPlane * NRows * NCols * (inputNet.modules[i+1].kW* inputNet.modules[i+1].kH - 1)
            --outfile:write(string.format("Computation:\ninputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, InputRows = %d, InputCols = %d\n\n",inputNet.modules[i+1].nInputPlane, inputNet.modules[i+1].nOutputPlane, inputNet.modules[i+1].kW, inputNet.modules[i+1].kH, inputNet.modules[i+1].dH, inputNet.modules[i+1].dW, NRows, NCols))
           --outfile:write(string.format('New Computation for Pixel: NewRows = %d, NewCols = %d, mul = %d add = %d\n\n', NRows, NCols, multiplications, additions))          
	 
         i = i + 2 
     else
         i = i + 1
     end 
     
    end         
  end   
  outfile:write(string.format('Total Memory Reused for this pixel at all layer: %d bytes\n\n', total_reuse_mem)) 
        
end

function ReUse_calculate_left_up(inputNet, pixel, pixel_left, pixel_up, count)

  local lrows, lcols
  local NRows_L, NCols_L, NRowsU, NCols_U, NRows, NCols  
  local NewRows, NewCols
  local NewRows_L, NewCols_L, NewRows_U, NewCols_U
  local rows_L, cols_L, rows_U, cols_U   

  NewRows_L = pixel_left[1]['br'][1] - pixel_left[1]['tl'][1] + 1
  NewCols_L = pixel[1]['br'][2] - pixel_left[1]['br'][2]
  NewRows_U = pixel[1]['br'][1] - pixel_up[1]['br'][1]
  NewCols_U = pixel_up[1]['br'][2] - pixel_up[1]['tl'][2] + 1
  NewRows = NewRows_U 
  NewCols = NewCols_L   	
 
  local total_reuse_mem = 0; local_reuse_mem = 0;
  --outfile:write(string.format('comparision between pixel (%d, %d) and (%d, %d) Sliding Window %d %s direction\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2],pixel2[count+1]['tl'][1], pixel2[count+1]['tl'][2], SW, dir))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2], pixel1[1]['tl'][1], pixel1[1]['tl'][2], pixel1[1]['br'][1], pixel1[1]['br'][2]))
  outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], pixel[1]['tl'][1], pixel[1]['tl'][2], pixel[1]['br'][1], pixel[1]['br'][2]))  


  if(pixel_left[1]['br'][2] > pixel[1]['tl'][2]) then
    --print('Input image overlapping region:'..'('..pixel2[1]['tl'][1]..','..pixel2[1]['tl'][2]..')'..','..'('..pixel1[1]['br'][1]..','..pixel1[1]['br'][2]..')\n\n')
  else
    print('No region to reuse or recompute')
    return
  end

  --outfile:write(string.format('Reuse Region for each layer\n\n')) 
  local multiplications, additions, comparison, orows, ocols
  i = 1
  while(i <= count) do
     rows_L = math.max(0, pixel_left[i]['br'][1] - pixel_left[i]['tl'][1] + 1)
     cols_L = math.max(0, pixel_left[i]['br'][2] - pixel[i]['tl'][2] + 1)
     NRows_L = pixel_left[i+1]['br'][1] - pixel_left[i+1]['tl'][1] + 1
     NCols_L = pixel[i+1]['br'][2] - pixel_left[i+1]['br'][2]
     
     rows_U = math.max(0, pixel_up[i]['br'][1] - pixel[i]['tl'][1])
     cols_U = math.max(0, pixel[i]['br'][2] - pixel_up[i]['tl'][2] + 1)
     NRows_U = pixel[i+1]['br'][1] - pixel_up[i+1]['br'][1]
     NCols_U = pixel_up[i+1]['br'][2] - pixel_up[i+1]['tl'][2] + 1
     
     NRows = NRows_U
     NCols = NCols_L
     local crows, ccols
      if((rows_L == 0 or cols_L == 0) and (rows_U == 0 or cols_U == 0)) then
	 outfile:write(string.format("No Overlapping below this layer\n\n"))
         break
      end
      outfile:write(string.format('Layer %d: %s\n\n', i, tostring(inputNet.modules[i])))         
      
      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
        ocols = inputNet.modules[i].kW - inputNet.modules[i].dW
        orows = inputNet.modules[i].kH - inputNet.modules[i].dH
        	
	--Below line is important 
        rows_L = rows_L - orows
        crows = NRows          
        ccols = NCols + ocols
	--outfile:write(string.format('Layer %d: %s\n\n', i, tostring(inputNet.modules[i]))) 
        
	if(i == 1) then 
	  
          outfile:write(string.format('Load input image data : %d X %d X %d, or %d bytes\n\n', opt.channels, NewRows, NewCols, opt.channels*NewRows*NewCols*4)) 
          outfile:write(string.format('Reuse:\n Input image Reuse region Left: %d X %d X %d or %d bytes \n\n', opt.channels, crows, ocols, opt.channels*crows*ocols*4)) 
          outfile:write(string.format('Reuse:\n Input image Reuse region UP: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ccols, opt.channels*orows* ccols *4)) 

          local_reuse_mem = opt.channels * crows * ocols * 4 +  opt.channels * orows * ccols * 4 
          total_reuse_mem = total_reuse_mem + local_reuse_mem

          memory_used[i][1] = math.max(memory_used[i][1], opt.channels * crows * ocols * 4) 
	  memory_used[i][2] = math.max(memory_used[i][2], opt.channels * opt.W * ocols * 4) 
	  
         outfile:write(string.format("Store:\nStore Input Image Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", opt.channels, NewRows, inputNet.modules[i].kW - 1, opt.channels * NewRows  * (inputNet.modules[i].kW - 1) * 4, opt.channels, inputNet.modules[i].kH - 1, NewCols, opt.channels * (inputNet.modules[i].kH - 1) * NewCols * 4))

	  	  	              	
	else
          -- i-1 layer is combined and used for convolution	  	  
	outfile:write(string.format('Reuse Left:\n ReUse data from layer %d: %s %d X %d X %d or %d bytes\n\n', i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], crows, ocols, inputNet.modules[i-1].output:size()[1] * crows * ocols* 4))                  
	outfile:write(string.format('Reuse UP:\n ReUse data from layer %d: %s %d X %d X %d or %d bytes\n\n', i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], orows, ccols, inputNet.modules[i-1].output:size()[1] * orows* ccols* 4))

         outfile:write(string.format("Store:\nStore layer %d: %s Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], crows, inputNet.modules[i].kW - 1, inputNet.modules[i-1].output:size()[1] * crows * (inputNet.modules[i].kW - 1) * 4, inputNet.modules[i-1].output:size()[1], inputNet.modules[i].kH - 1, ccols, inputNet.modules[i-1].output:size()[1] * (inputNet.modules[i].kH - 1) * ccols * 4))          

        memory_used[i][1] = math.max(memory_used[i][1], inputNet.modules[i-1].output:size()[1] * crows * ocols* 4) 
	memory_used[i][2] = math.max(memory_used[i][2], inputNet.modules[i-1].output:size()[1] * orows * inputNet.modules[i-1].output:size()[2] * 4) 
	  
        local_reuse_mem = inputNet.modules[i-1].output:size()[1] * crows* ocols* 4 +  inputNet.modules[i-1].output:size()[1] * orows* ccols * 4
        total_reuse_mem = total_reuse_mem + local_reuse_mem
                             	
        end
        outfile:write(string.format('Total Memory Reused at this layer: %d bytes\n\n', local_reuse_mem)) 
       
	   multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * NRows * NCols * inputNet.modules[i].kW * inputNet.modules[i].kH
           additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * NRows * NCols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1)
           --outfile:write(string.format("\nComputation: \ninputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, InputRows = %d, InputCols = %d\n\n", inputNet.modules[i].nInputPlane, inputNet.modules[i].nOutputPlane, inputNet.modules[i].kW, inputNet.modules[i].kH, inputNet.modules[i].dH, inputNet.modules[i].dW, NRows, NCols))
           --outfile:write(string.format('Computation for Pixel: NewRows = %d, NewCols = %d, mul = %d add = %d\n\n', NRows, NCols, multiplications, additions))          

        i = i + 1
      else if(inputNet.modules[i].__typename == 'nn.SpatialZeroPadding' and inputNet.modules[i+1].__typename == 'nn.SpatialConvolution') then
	
          --outfile:write(string.format('Layer %d: %s\n\n', i+1, tostring(inputNet.modules[i+1]))) 
        
          ocols = math.floor((inputNet.modules[i+1].kW/2)) * 2 --+ inputNet.modules[i].pad_l 
	  orows = math.floor((inputNet.modules[i+1].kH/2)) * 2 -- +inputNet.modules[i].pad_t
          
	  NRows_L = pixel_left[i+1]['br'][1] - pixel_left[i+1]['tl'][1] + 1
    	  NCols_L = pixel[i+1]['br'][2] - pixel_left[i+1]['br'][2]
	  
        
          NRows_U = pixel[i+1]['br'][1] - pixel_up[i+1]['br'][1]
	  NCols_U = pixel_up[i+1]['br'][2] - pixel_up[i+1]['tl'][2] + 1


          NRows = NRows_U
          NCols = NCols_L

          local crows, ccols
          crows = NRows
          ccols = NCols + ocols 
	     
	  if(i == 1) then
             outfile:write(string.format('New data to load: %d X %d X %d, or %d bytes\n\n', opt.channels, NewRows, NewCols, opt.channels*NewRows*NewCols*4))	
             outfile:write(string.format('Reuse:\n Input image Reuse region Left: %d X %d X %d or %d bytes \n\n', opt.channels, crows, ocols, opt.channels*crows*ocols*4)) 
             outfile:write(string.format('Reuse:\n Input image Reuse region UP: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ccols, opt.channels*orows*ccols*4)) 

             local_reuse_mem = opt.channels*crows*ocols*4 + opt.channels*orows*ccols*4             
             total_reuse_mem = total_reuse_mem + local_reuse_mem           
             memory_used[i+1][1] = math.max(memory_used[i+1][1], opt.channels * crows * ocols * 4) 
	     memory_used[i+1][2] = math.max(memory_used[i+1][2], opt.channels * opt.W * ocols * 4) 
	 
             outfile:write(string.format("Store:\nStore Input Image Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", opt.channels, NewRows, inputNet.modules[i+1].kW - 1, opt.channels * NewRows  * (inputNet.modules[i+1].kW - 1) * 4, opt.channels, inputNet.modules[i+1].kH - 1, NewCols, opt.channels * (inputNet.modules[i+1].kH - 1) * NewCols * 4))


             	  	              	
	   else	  	  
             
	      outfile:write(string.format('Reuse Left:\n ReUse data from layer %d: %s %d X %d X %d or %d bytes\n\n', i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], crows, ocols, inputNet.modules[i-1].output:size()[1] * crows* ocols* 4))                  
	      outfile:write(string.format('Reuse UP:\n ReUse data from layer %d: %s %d X %d X %d or %d bytes\n\n', i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], orows, ccols, inputNet.modules[i-1].output:size()[1] * orows* ccols* 4))          
             memory_used[i+1][1] = math.max(memory_used[i+1][1], inputNet.modules[i-1].output:size()[1] * crows * ocols* 4) 
	     memory_used[i+1][2] = math.max(memory_used[i+1][2], inputNet.modules[i-1].output:size()[1] * orows * inputNet.modules[i-1].output:size()[2] * 4) 
             local_reuse_mem = inputNet.modules[i-1].output:size()[1] * crows* ocols* 4 + inputNet.modules[i-1].output:size()[1] * orows* ccols* 4             
             total_reuse_mem = total_reuse_mem + local_reuse_mem           

	      outfile:write(string.format("Store:\nStore layer %d: %s Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], crows, inputNet.modules[i+1].kW - 1, inputNet.modules[i-1].output:size()[1] * crows * (inputNet.modules[i+1].kW - 1) * 4, inputNet.modules[i-1].output:size()[1], inputNet.modules[i+1].kH - 1, ccols, inputNet.modules[i-1].output:size()[1] * (inputNet.modules[i+1].kH - 1) * ccols * 4))          

        	
	   end      
	    outfile:write(string.format('Total Memory Reused at this layer: %d bytes\n\n', local_reuse_mem)) 
        
       	    multiplications = inputNet.modules[i+1].nOutputPlane * inputNet.modules[i+1].nInputPlane * NRows * NCols * inputNet.modules[i+1].kW * inputNet.modules[i+1].kH
            additions = inputNet.modules[i+1].nOutputPlane * inputNet.modules[i+1].nInputPlane * NRows * NCols * (inputNet.modules[i+1].kW* inputNet.modules[i+1].kH - 1)
            --outfile:write(string.format("Computation:\ninputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, InputRows = %d, InputCols = %d\n\n",inputNet.modules[i+1].nInputPlane, inputNet.modules[i+1].nOutputPlane, inputNet.modules[i+1].kW, inputNet.modules[i+1].kH, inputNet.modules[i+1].dH, inputNet.modules[i+1].dW, NRows, NCols))
            --outfile:write(string.format('New Computation for Pixel2: NewRows = %d, NewCols = %d, mul = %d add = %d\n\n', NRows, NCols, multiplications, additions))          
	 
         i = i + 2 
     else
         i = i + 1
     end 
     
    end         
  end 
  
  outfile:write(string.format('Total Memory Reused for this pixel at all layer: %d bytes\n\n', total_reuse_mem)) 
end

function ReUse_calculate(inputNet, pixel, pixel_left, pixel_up, count)

   if(pixel_left == nil and pixel_up == nil) then
      outfile:write(string.format('\nFirst Pixel No Reuse or Recompute for this\n\n'))
      Recomputation_calculate_orig(inputNet, pixel, count)
   else if(pixel_left ~= nil and pixel_up ~= nil) then
      ReUse_calculate_left_up(inputNet, pixel, pixel_left, pixel_up, count) 
   else if(pixel_left ~= nil) then
      ReUse_calculate_left(inputNet, pixel, pixel_left, count)
   else if(pixel_up ~= nil) then
      ReUse_calculate_up(inputNet, pixel, pixel_up, count)
   end 
   end
   end
   end
end

--Finds out Recomputation happening between two pixels in single row
function Recomputation_calculate_left_up(inputNet, pixel, pixel_left, pixel_up, count)

  local rows_L, cols_L    -- overlapping rows and cols at each layer 
  local rows_U, cols_U    -- overlapping rows and cols at each layer 

  local lrows, lcols  -- local rows and cols at that layer
  local NRows_L, NCols_L  -- New rows and cols
  local NRows_U, NCols_U  -- New rows and cols  
  local orows_L, ocols_L
  local orows_U, ocols_U
  local Nrows, Ncols
  local right = 0  -- Right or Down 
  local Icols = pixel[1]['br'][2] - pixel[1]['tl'][2] + 1
  local Irows = pixel[1]['br'][1] - pixel[1]['tl'][1] + 1
  local SW
  
  --outfile:write(string.format('\nInput Region for one pixel in ouput layer (Reshape Layer) is %dX%dX%d\n\n', opt.channels, Irows, Icols))
  	
  orows_L = pixel_left[1]['br'][1] - pixel_left[1]['tl'][1] + 1 
  ocols_L = pixel_left[1]['br'][2] - pixel[1]['tl'][2] + 1

  NRows_L = pixel_left[1]['br'][1] - pixel_left[1]['tl'][1] + 1
  NCols_L = pixel[1]['br'][2] - pixel_left[1]['br'][2]

  orows_U = pixel_up[1]['br'][1] - pixel[1]['tl'][1] + 1
  ocols_U = pixel_up[1]['br'][2] - pixel_up[1]['tl'][2] + 1
  -- Important to do below step else it will be in both the cases    
  ocols_U = ocols_U - ocols_L     

  NRows_U = pixel[1]['br'][1] - pixel_up[1]['br'][1]
  NCols_U = pixel_up[1]['br'][2] - pixel_up[1]['tl'][2] + 1
  
  NRows = NRows_U
  NCols = NCols_L 

  --outfile:write(string.format('comparision between pixel (%d, %d) and (%d, %d) Sliding Window %d %s direction\n\n', pixel_left[count+1]['tl'][1], pixel_left[count+1]['tl'][2],pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], SW, dir))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2], pixel1[1]['tl'][1], pixel1[1]['tl'][2], pixel1[1]['br'][1], pixel1[1]['br'][2]))
  outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], pixel[1]['tl'][1], pixel[1]['tl'][2], pixel[1]['br'][1], pixel[1]['br'][2]))
   
  local multiplications, additions, comparison 
  local multiplications_L, additions_L, comparison_L 
  local multiplications_U, additions_U, comparison_U 
  
  local multiplications_t = 0; additions_t = 0; comparison_relu_t = 0; comparison_pool_t = 0  
  local multiplications_r = 0; additions_r = 0; comparison_relu_r = 0; comparison_pool_r = 0  
   
  for i = 1, count do
          -- check for overlapping computation
      rows_L = math.max(0, pixel_left[i+1]['br'][1] - pixel_left[i+1]['tl'][1] + 1) 
      cols_L = math.max(0, pixel_left[i+1]['br'][2] - pixel[i+1]['tl'][2] + 1)
      -- check for total computation

      rows_U = math.max(0, pixel_up[i+1]['br'][1] - pixel[i+1]['tl'][1] + 1)
      cols_U = math.max(0, pixel_up[i+1]['br'][2] - pixel_up[i+1]['tl'][2] + 1)

      if((rows_L == 0 or cols_L == 0) and (rows_U == 0 or cols_U == 0)) then
	 outfile:write(string.format("No Overlapping below this layer\n\n"))
         break
      end

      -- below statement is important
      cols_U = cols_U - cols_L 

      lrows = pixel[i+1]['br'][1] - pixel[i+1]['tl'][1] + 1 
      lcols = pixel[i+1]['br'][2] - pixel[i+1]['tl'][2] + 1

      if(i == 1) then 
        outfile:write(string.format('Input image reuse region UP: %d X %d X %d or %d bytes \n\n', opt.channels, orows_U, ocols_U, opt.channels*orows_U*ocols_U*4))
        outfile:write(string.format('Input image reuse region Left: %d X %d X %d or %d bytes \n\n', opt.channels, orows_L, ocols_L, opt.channels*orows_L*ocols_L*4))  
        outfile:write(string.format('New data to load: %d X %d X %d, or %d bytes\n\n', opt.channels, NRows, NCols, opt.channels*NRows*NCols*4))
        outfile:write(string.format('Operations at each layer\n\n'))
      end	  	

      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
        
  	  outfile:write(string.format("Layer %d : %s, inputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, OutputRows = %d, OutputCols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].nInputPlane, inputNet.modules[i].nOutputPlane, inputNet.modules[i].kW, inputNet.modules[i].kH, inputNet.modules[i].dH, inputNet.modules[i].dW, lrows, lcols))
          --Total Operations
	  multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * inputNet.modules[i].kW * inputNet.modules[i].kH
          additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1) 
 	  outfile:write(string.format('Total Multiplications and addition, mul = %d add = %d\n\n', multiplications, additions))
	  multiplications_t = multiplications_t + multiplications
	  additions_t = additions_t + additions
	  
	  --Recomputation
	   multiplications_L = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows_L * cols_L * inputNet.modules[i].kW * inputNet.modules[i].kH
           additions_L = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows_L * cols_L * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1)

           multiplications_r = multiplications_r + multiplications_L
           additions_r = additions_r + additions_L
 
           outfile:write(string.format('Recomputation Left: OverlappingRows = %d, OverlappingCols = %d, mul = %d add = %d\n\n', rows_L, cols_L, multiplications_L, additions_L))          
	 
           multiplications_U = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows_U * cols_U * inputNet.modules[i].kW * inputNet.modules[i].kH
           additions_U = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows_U * cols_U * (inputNet.modules[i].kW * inputNet.modules[i].kH - 1)

	   multiplications_r = multiplications_r + multiplications_U
           additions_r = additions_r + additions_U
            

	   outfile:write(string.format('Recomputation UP: OverlappingRows = %d, OverlappingCols = %d, mul = %d add = %d\n\n', rows_U, cols_U, multiplications_U, additions_U))           
           
	   outfile:write(string.format('Recomputation Total: multiplication = %d addition = %d\n\n', multiplications_U + multiplications_L, additions_U + additions_L))           
          	 
     end
     
     if(inputNet.modules[i].__typename == 'nn.Threshold' or inputNet.modules[i].__typename == 'nn.ReLU') then
         outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, rows = %d, cols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows, lcols))
	  comparison = inputNet.modules[i].output:size()[1] * lrows * lcols
          outfile:write(string.format('Total Number of comparison,  comp = %d \n\n', comparison))

          comparison_relu_t = comparison_relu_t + comparison

	  comparison_L = inputNet.modules[i].output:size()[1] * rows_L * cols_L
          outfile:write(string.format('Recomputation Left: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows_L, cols_L, comparison_L))

          comparison_relu_r = comparison_relu_r + comparison_L

	  comparison_U = inputNet.modules[i].output:size()[1] * rows_U * cols_U
          comparison_relu_r = comparison_relu_r + comparison_U
 
	 outfile:write(string.format('Recomputation UP: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows_U, cols_U, comparison_U))
          outfile:write(string.format('Recomputation Total: comparison = %d \n\n', comparison_U + comparison_L))
            
     end
    
     if(inputNet.modules[i].__typename == 'nn.SpatialMaxPooling') then
           outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, Inputrows = %d, Inputcols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows, lcols))
	  comparison = inputNet.modules[i].output:size()[1] * (lrows) * (lcols) * inputNet.modules[i].kW * inputNet.modules[i].kH 
	  outfile:write(string.format('Total Max Comparison: comp = %d \n\n', comparison))	
          comparison_pool_t = comparison_pool_t + comparison

	  comparison_L = inputNet.modules[i].output:size()[1] * (rows_L) * (cols_L) * inputNet.modules[i].kW * inputNet.modules[i].kH 
          outfile:write(string.format('Recomputation Left: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows_L, cols_L, comparison_L))
	  comparison_pool_r = comparison_pool_r + comparison_L          
          
	  comparison_U = inputNet.modules[i].output:size()[1] * (rows_U) * (cols_U) * inputNet.modules[i].kW * inputNet.modules[i].kH 
	  comparison_pool_r = comparison_pool_r + comparison_U          
                    
          outfile:write(string.format('Recomputation UP: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows_U, cols_U, comparison_U))

          outfile:write(string.format('Recomputation Total: Comparison = %d \n\n', comparison_U + comparison_L))
          
     end
  end
  
   total_multiplication = total_multiplication + multiplications_t
   total_addition = total_addition + additions_t
   total_comparison_Relu = total_comparison_Relu + comparison_relu_t
   total_comparison_pool = total_comparison_pool + comparison_pool_t
   
   re_multiplication = re_multiplication + multiplications_r
   re_addition = re_addition + additions_r
   re_comparison_Relu = re_comparison_Relu + comparison_relu_r
   re_comparison_pool = re_comparison_pool + comparison_pool_r
   
   outfile:write(string.format('Recomputation:\n\n'))
   outfile:write(string.format('Multiplications: %0.2f\n\n', (multiplications_r/multiplications_t)*100))
   outfile:write(string.format('Additions: %0.2f\n\n', (additions_r/additions_t)*100))
   outfile:write(string.format('Relu Comparison: %0.2f\n\n', (comparison_relu_r/comparison_relu_t)*100))
   outfile:write(string.format('Max pool Comparison: %0.2f\n\n', (comparison_pool_r/comparison_pool_t)*100))      
end


--Finds out Recomputation happening between two pixels in single row
function Recomputation_calculate_left(inputNet, pixel, pixel_left, count)

  local rows, cols    -- overlapping rows and cols at each layer 
  local lrows, lcols  -- local rows and cols at that layer
  local NRows, NCols  -- New rows and cols
  local orows, ocols
  local right = 0  -- Right or Down 
  local Icols = pixel_left[1]['br'][2] - pixel_left[1]['tl'][2] + 1
  local Irows = pixel_left[1]['br'][1] - pixel_left[1]['tl'][1] + 1
  
  --outfile:write(string.format('\nInput Region for one pixel in ouput layer (Reshape Layer) is %dX%dX%d\n\n', opt.channels, Irows, Icols))
  --outfile:write(string.format('At first iteration input data to load %d X %d X %d or %d bytes\n\n', opt.channels, Irows, Icols, opt.channels* Irows*Icols *4))
  	
   orows = pixel_left[1]['br'][1] - pixel_left[1]['tl'][1] + 1 
   ocols = pixel_left[1]['br'][2] - pixel[1]['tl'][2] + 1

   NRows = pixel_left[1]['br'][1] - pixel_left[1]['tl'][1] + 1
   NCols = pixel[1]['br'][2] - pixel_left[1]['br'][2]
   SW = pixel[1]['tl'][2] - pixel_left[1]['tl'][2]
 

  --outfile:write(string.format('comparision between pixel (%d, %d) and (%d, %d) Sliding Window %d %s direction\n\n', pixel_left[count+1]['tl'][1], pixel_left[count+1]['tl'][2],pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], SW, dir))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2], pixel1[1]['tl'][1], pixel1[1]['tl'][2], pixel1[1]['br'][1], pixel1[1]['br'][2]))
  outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], pixel[1]['tl'][1], pixel[1]['tl'][2], pixel[1]['br'][1], pixel[1]['br'][2]))

 
  if(pixel_left[1]['br'][2] > pixel[1]['tl'][2]) then
    --print('Input image reuse region:'..'('..pixel2[1]['tl'][1]..','..pixel2[1]['tl'][2]..')'..','..'('..pixel1[1]['br'][1]..','..pixel1[1]['br'][2]..')\n\n')
  else
    print('No region to reuse or recompute')
    return
  end

   
  local multiplications, additions, comparison 
  local multiplications_t = 0; additions_t = 0; comparison_relu_t = 0; comparison_pool_t = 0  
  local multiplications_r = 0; additions_r = 0; comparison_relu_r = 0; comparison_pool_r = 0  
  
  for i = 1, count do
       -- check for overlapping computation
      rows = math.max(0, pixel_left[i+1]['br'][1] - pixel_left[i+1]['tl'][1] + 1) 
      cols = math.max(0, pixel_left[i+1]['br'][2] - pixel[i+1]['tl'][2] + 1)
      -- check for total computation
      lrows = pixel[i+1]['br'][1] - pixel[i+1]['tl'][1] + 1 
      lcols = pixel[i+1]['br'][2] - pixel[i+1]['tl'][2] + 1
   
      if(i == 1) then 
        outfile:write(string.format('Input image reuse region: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ocols, opt.channels*orows*ocols*4))  
        outfile:write(string.format('New data to load: %d X %d X %d, or %d bytes\n\n', opt.channels, NRows, NCols, opt.channels*NRows*NCols*4))
        outfile:write(string.format('Operations at each layer\n\n'))
      end	  	

      if(rows == 0 or cols == 0) then
	 outfile:write(string.format("No Overlapping below this layer\n\n"))
         break
      end

      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
        
  	  outfile:write(string.format("Layer %d : %s, inputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, OutputRows = %d, OutputCols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].nInputPlane, inputNet.modules[i].nOutputPlane, inputNet.modules[i].kW, inputNet.modules[i].kH, inputNet.modules[i].dH, inputNet.modules[i].dW, lrows, lcols))
          --Total Operations
	  multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * inputNet.modules[i].kW * inputNet.modules[i].kH
          additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1) 
 	  outfile:write(string.format('Total Multiplications and addition, mul = %d add = %d\n\n', multiplications, additions))
	
	  multiplications_t = multiplications_t + multiplications
	  additions_t = additions_t + additions	  

	  --Recomputation
	   multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows * cols * inputNet.modules[i].kW * inputNet.modules[i].kH
           additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows * cols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1)
           
           outfile:write(string.format('Recomputation: OverlappingRows = %d, OverlappingCols = %d, mul = %d add = %d\n\n', rows, cols, multiplications, additions))          

           multiplications_r = multiplications_r + multiplications
	   additions_r = additions_r + additions

	  	 
     end
     
     if(inputNet.modules[i].__typename == 'nn.Threshold' or inputNet.modules[i].__typename == 'nn.ReLU') then
         outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, rows = %d, cols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows, lcols))
	  comparison = inputNet.modules[i].output:size()[1] * lrows * lcols
          outfile:write(string.format('Total Number of comparison,  comp = %d \n\n', comparison))
	  comparison_relu_t = comparison_relu_t + comparison

	  comparison = inputNet.modules[i].output:size()[1] * rows * cols
          outfile:write(string.format('Recomputation: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows, cols, comparison))
          comparison_relu_r = comparison_relu_r + comparison
     end
    
     if(inputNet.modules[i].__typename == 'nn.SpatialMaxPooling') then
           outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, Inputrows = %d, Inputcols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows, lcols))
	  comparison = inputNet.modules[i].output:size()[1] * (lrows) * (lcols) * inputNet.modules[i].kW * inputNet.modules[i].kH 
	  outfile:write(string.format('Total Max Comparison: comp = %d \n\n', comparison))	          
          comparison_pool_t = comparison_pool_t + comparison
	  comparison = inputNet.modules[i].output:size()[1] * (rows) * (cols) * inputNet.modules[i].kW * inputNet.modules[i].kH 
          outfile:write(string.format('Recomputation: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows, cols, comparison))
          comparison_pool_r = comparison_pool_r + comparison
     end
  end      
  
   total_multiplication = total_multiplication + multiplications_t
   total_addition = total_addition + additions_t
   total_comparison_Relu = total_comparison_Relu + comparison_relu_t
   total_comparison_pool = total_comparison_pool + comparison_pool_t
   re_multiplication = re_multiplication + multiplications_r
   re_addition = re_addition + additions_r
   re_comparison_Relu = re_comparison_Relu + comparison_relu_r
   re_comparison_pool = re_comparison_pool + comparison_pool_r
 
   outfile:write(string.format('Recomputation:\n\n'))
   outfile:write(string.format('Multiplications: %0.2f\n\n', (multiplications_r/multiplications_t)*100))
   outfile:write(string.format('Additions: %0.2f\n\n', (additions_r/additions_t)*100))
   outfile:write(string.format('Relu Comparison: %0.2f\n\n', (comparison_relu_r/comparison_relu_t)*100))
   outfile:write(string.format('Max pool Comparison: %0.2f\n\n', (comparison_pool_r/comparison_pool_t)*100))
end


--Finds out Recomputation happening between two pixels in single col
function Recomputation_calculate_up(inputNet, pixel, pixel_up, count)

  local rows, cols    -- overlapping rows and cols at each layer 
  local lrows, lcols  -- local rows and cols at that layer
  local NRows, NCols  -- New rows and cols
  local orows, ocols
  local Icols = pixel_up[1]['br'][2] - pixel_up[1]['tl'][2] + 1
  local Irows = pixel_up[1]['br'][1] - pixel_up[1]['tl'][1] + 1
  local SW
  
  --outfile:write(string.format('\nInput Region for one pixel in ouput layer (Reshape Layer) is %dX%dX%d\n\n', opt.channels, Irows, Icols))
  --outfile:write(string.format('At first iteration input data to load %d X %d X %d or %d bytes\n\n', opt.channels, Irows, Icols, opt.channels* Irows*Icols *4))
  	
  orows = pixel_up[1]['br'][1] - pixel[1]['tl'][1] + 1
  ocols = pixel_up[1]['br'][2] - pixel_up[1]['tl'][2] + 1    

  NRows = pixel[1]['br'][1] - pixel_up[1]['br'][1]
  NCols = pixel_up[1]['br'][2] - pixel_up[1]['tl'][2] + 1
  SW = pixel[1]['tl'][1] - pixel_up[1]['tl'][1]

  --outfile:write(string.format('comparision between pixel (%d, %d) and (%d, %d) Sliding Window %d %s direction\n\n', pixel_left[count+1]['tl'][1], pixel_left[count+1]['tl'][2],pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], SW, dir))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2], pixel1[1]['tl'][1], pixel1[1]['tl'][2], pixel1[1]['br'][1], pixel1[1]['br'][2]))
  outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], pixel[1]['tl'][1], pixel[1]['tl'][2], pixel[1]['br'][1], pixel[1]['br'][2]))

 
  if(pixel_up[1]['br'][1] > pixel[1]['tl'][1]) then
    --print('Input image reuse region:'..'('..pixel2[1]['tl'][1]..','..pixel2[1]['tl'][2]..')'..','..'('..pixel1[1]['br'][1]..','..pixel1[1]['br'][2]..')\n\n')
  else
    print('No region to reuse or recompute')
    return
  end

   
  local multiplications, additions, comparison 
  local multiplications_t = 0; additions_t = 0; comparison_relu_t = 0; comparison_pool_t = 0  
  local multiplications_r = 0; additions_r = 0; comparison_relu_r = 0; comparison_pool_r = 0  
  
  for i = 1, count do
      rows = math.max(0, pixel_up[i+1]['br'][1] - pixel[i+1]['tl'][1] + 1)
      cols = math.max(0, pixel_up[i+1]['br'][2] - pixel_up[i+1]['tl'][2] + 1)
      lrows = pixel[i+1]['br'][1] - pixel[i+1]['tl'][1] + 1 
      lcols = pixel[i+1]['br'][2] - pixel[i+1]['tl'][2] + 1
      

      if(i == 1) then 
        outfile:write(string.format('Input image reuse region: %d X %d X %d or %d bytes \n\n', opt.channels, orows, ocols, opt.channels*orows*ocols*4))  
        outfile:write(string.format('New data to load: %d X %d X %d, or %d bytes\n\n', opt.channels, NRows, NCols, opt.channels*NRows*NCols*4))
        outfile:write(string.format('Operations at each layer\n\n'))
      end	  	

      if(rows == 0 or cols == 0) then
	 outfile:write(string.format("No Overlapping below this layer\n\n"))
         break
      end

      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
        
  	  outfile:write(string.format("Layer %d : %s, inputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, OutputRows = %d, OutputCols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].nInputPlane, inputNet.modules[i].nOutputPlane, inputNet.modules[i].kW, inputNet.modules[i].kH, inputNet.modules[i].dH, inputNet.modules[i].dW, lrows, lcols))
          --Total Operations
	  multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * inputNet.modules[i].kW * inputNet.modules[i].kH
          additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1) 
 	  outfile:write(string.format('Total Multiplications and addition, mul = %d add = %d\n\n', multiplications, additions))
	  multiplications_t = multiplications_t + multiplications
	  additions_t = additions_t + additions
	  

	  --Recomputation
           multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows * cols * inputNet.modules[i].kW * inputNet.modules[i].kH
           additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * rows * cols * (inputNet.modules[i].kW * inputNet.modules[i].kH - 1)
	   outfile:write(string.format('Recomputation: OverlappingRows = %d, OverlappingCols = %d, mul = %d add = %d\n\n', rows, cols, multiplications, additions))           
          multiplications_r = multiplications_r + multiplications
          additions_r = additions_r + additions
	 
     end
     
     if(inputNet.modules[i].__typename == 'nn.Threshold' or inputNet.modules[i].__typename == 'nn.ReLU') then
         outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, rows = %d, cols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows, lcols))
	  comparison = inputNet.modules[i].output:size()[1] * lrows * lcols        
          outfile:write(string.format('Total Number of comparison,  comp = %d \n\n', comparison))
	  comparison_relu_t = comparison_relu_t + comparison          

	  comparison = inputNet.modules[i].output:size()[1] * rows * cols
          outfile:write(string.format('Recomputation: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows, cols, comparison))
          comparison_relu_r = comparison_relu_r + comparison
     end
    
     if(inputNet.modules[i].__typename == 'nn.SpatialMaxPooling') then
           outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, Inputrows = %d, Inputcols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows, lcols))
	  comparison = inputNet.modules[i].output:size()[1] * (lrows) * (lcols) * inputNet.modules[i].kW * inputNet.modules[i].kH 
	  outfile:write(string.format('Total Max Comparison: comp = %d \n\n', comparison))
          comparison_pool_t = comparison_pool_t + comparison
	          
	  comparison = inputNet.modules[i].output:size()[1] * (rows) * (cols) * inputNet.modules[i].kW * inputNet.modules[i].kH 
          outfile:write(string.format('Recomputation: OverlappingRows = %d, OverlappingCols = %d, comp = %d \n\n', rows, cols, comparison))
          comparison_pool_r = comparison_pool_r + comparison
     end
  end 
  
   total_multiplication = total_multiplication + multiplications_t
   total_addition = total_addition + additions_t
   total_comparison_Relu = total_comparison_Relu + comparison_relu_t
   total_comparison_pool = total_comparison_pool + comparison_pool_t
   re_multiplication = re_multiplication + multiplications_r
   re_addition = re_addition + additions_r
   re_comparison_Relu = re_comparison_Relu + comparison_relu_r
   re_comparison_pool = re_comparison_pool + comparison_pool_r

   outfile:write(string.format('Recomputation:\n\n'))
   outfile:write(string.format('Multiplications: %0.2f\n\n', (multiplications_r/multiplications_t)*100))
   outfile:write(string.format('Additions: %0.2f\n\n', (additions_r/additions_t)*100))
   outfile:write(string.format('Relu Comparison: %0.2f\n\n', (comparison_relu_r/comparison_relu_t)*100))
   outfile:write(string.format('Max pool Comparison: %0.2f\n\n', (comparison_pool_r/comparison_pool_t)*100))     
end

--Finds out Recomputation happening between two pixels in single col
function Recomputation_calculate_orig(inputNet, pixel, count)

  local rows, cols    -- overlapping rows and cols at each layer 
  local lrows, lcols  -- local rows and cols at that layer
  local NRows, NCols  -- New rows and cols
  local crows, ccols
  local Icols = pixel[1]['br'][2] - pixel[1]['tl'][2] + 1
  local Irows = pixel[1]['br'][1] - pixel[1]['tl'][1] + 1
  local SW
  
  --outfile:write(string.format('\nFirst Pixel in output layer\n\n'))  
  --outfile:write(string.format('\nInput Region for one pixel in ouput layer (Reshape Layer) is %dX%dX%d\n\n', opt.channels, Irows, Icols))
  --outfile:write(string.format('At first iteration input data to load %d X %d X %d or %d bytes\n\n', opt.channels, Irows, Icols, opt.channels* Irows*Icols *4))

  NRows = pixel[1]['br'][1] - pixel[1]['tl'][1] + 1
  NCols = pixel[1]['br'][2] - pixel[1]['tl'][2] + 1

  --outfile:write(string.format('comparision between pixel (%d, %d) and (%d, %d) Sliding Window %d %s direction\n\n', pixel_left[count+1]['tl'][1], pixel_left[count+1]['tl'][2],pixel[count+1]['tl'][1], pixel[count+1]['tl'][2], SW, dir))
  --outfile:write(string.format('Input Image Region for pixel (%d, %d) is (%d, %d) to (%d, %d)\n\n', pixel1[count+1]['tl'][1], pixel1[count+1]['tl'][2], pixel1[1]['tl'][1], pixel1[1]['tl'][2], pixel1[1]['br'][1], pixel1[1]['br'][2]))
  outfile:write(string.format('Load input image data (%d, %d) to (%d, %d)\n\n', pixel[1]['tl'][1], pixel[1]['tl'][2], pixel[1]['br'][1], pixel[1]['br'][2]))

   
  local multiplications, additions, comparison 
  local multiplications_t = 0; additions_t = 0; comparison_relu_t = 0; comparison_pool_t = 0  
  local multiplications_r = 0; additions_r = 0; comparison_relu_r = 0; comparison_pool_r = 0  
  
  for i = 1, count do
      lrows = pixel[i+1]['br'][1] - pixel[i+1]['tl'][1] + 1 
      lcols = pixel[i+1]['br'][2] - pixel[i+1]['tl'][2] + 1
      -- current rows and cols
      crows = pixel[i]['br'][1] - pixel[i]['tl'][1] + 1 
      ccols = pixel[i]['br'][2] - pixel[i]['tl'][2] + 1
      
      if(i == 1) then 
        outfile:write(string.format('New data to load: %d X %d X %d, or %d bytes\n\n', opt.channels, NRows, NCols, opt.channels*NRows*NCols*4))
        --outfile:write(string.format('Operations at each layer\n\n'))
      end	  	

      --outfile:write(string.format('Layer %d: %s\n\n', i, tostring(inputNet.modules[i])))
               
      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
        
  	  outfile:write(string.format("Layer %d : %s, inputplanes = %d, outputplanes = %d, Kernel Width = %d, Kernel Height = %d, StrideRows = %d, StrideCols = %d, OutputRows = %d, OutputCols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].nInputPlane, inputNet.modules[i].nOutputPlane, inputNet.modules[i].kW, inputNet.modules[i].kH, inputNet.modules[i].dH, inputNet.modules[i].dW, lrows, lcols))
          --Total Operations
	  multiplications = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * inputNet.modules[i].kW * inputNet.modules[i].kH
          additions = inputNet.modules[i].nOutputPlane * inputNet.modules[i].nInputPlane * lrows * lcols * (inputNet.modules[i].kW* inputNet.modules[i].kH - 1)

          multiplications_t = multiplications_t + multiplications
          additions_t = additions_t + additions
          if(computation ~= 1) then
            if(i == 1) then
               outfile:write(string.format("Store:\nStore Input Image Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", opt.channels, NRows, inputNet.modules[i].kW - inputNet.modules[i].dW, opt.channels * NRows  * (inputNet.modules[i].kW - inputNet.modules[i].dW) * 4, opt.channels, inputNet.modules[i].kH - inputNet.modules[i].dH, NCols, opt.channels * (inputNet.modules[i].kH - inputNet.modules[i].dH) * NCols * 4))
            else
               outfile:write(string.format("Store:\nStore layer %d: %s Region for next Pixels\nStore Right %dX%dX%d or %d bytes\nStore Bottom %dX%dX%d or %d bytes\n\n", i-1, tostring(inputNet.modules[i-1]), inputNet.modules[i-1].output:size()[1], crows, inputNet.modules[i].kW - inputNet.modules[i].dW, inputNet.modules[i-1].output:size()[1] * crows * (inputNet.modules[i].kW - inputNet.modules[i].dW) * 4, inputNet.modules[i-1].output:size()[1], inputNet.modules[i].kH - inputNet.modules[i].dH, ccols, inputNet.modules[i-1].output:size()[1] * (inputNet.modules[i].kH - inputNet.modules[i].dW) * ccols * 4))
          
            end
          end
	  	 
     end
     
     if(inputNet.modules[i].__typename == 'nn.Threshold' or inputNet.modules[i].__typename == 'nn.ReLU') then
          outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, rows = %d, cols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows, lcols))
	  comparison = inputNet.modules[i].output:size()[1] * lrows * lcols
          comparison_relu_t = comparison_relu_t + comparison
          outfile:write(string.format('Total Number of comparison,  comp = %d \n\n', comparison))
          
     end
    
     if(inputNet.modules[i].__typename == 'nn.SpatialMaxPooling') then
           outfile:write(string.format("Layer %d : %s, inputplanes = outputplanes = %d, Inputrows = %d, Inputcols = %d\n\n", i, tostring(inputNet.modules[i]), inputNet.modules[i].output:size()[1], lrows * inputNet.modules[i].kH, lcols * inputNet.modules[i].kW))
	  comparison = inputNet.modules[i].output:size()[1] * (lrows) * (lcols) * inputNet.modules[i].kW * inputNet.modules[i].kH 
	  comparison_pool_t = comparison_pool_t + comparison
	  outfile:write(string.format('Total Max Comparison: comp = %d \n\n', comparison))	          
     end
  end      

 if(computation == 1) then  
   total_multiplication = total_multiplication + multiplications_t
   total_addition = total_addition + additions_t
   total_comparison_Relu = total_comparison_Relu + comparison_relu_t
   total_comparison_pool = total_comparison_pool + comparison_pool_t

   re_multiplication = re_multiplication + multiplications_r
   re_addition = re_addition + additions_r
   re_comparison_Relu = re_comparison_Relu + comparison_relu_r
   re_comparison_pool = re_comparison_pool + comparison_pool_r
   outfile:write(string.format('Recomputation:\n\n'))
   outfile:write(string.format('Multiplications: %0.2f\n\n', (multiplications_r/multiplications_t)*100))
   outfile:write(string.format('Additions: %0.2f\n\n', (additions_r/additions_t)*100))
   outfile:write(string.format('Relu Comparison: %0.2f\n\n', (comparison_relu_r/comparison_relu_t)*100))
   outfile:write(string.format('Max pool Comparison: %0.2f\n\n', (comparison_pool_r/comparison_pool_t)*100))
 end
  
end

function Recomputation_calculate(inputNet, pixel, pixel_left, pixel_up, count)
 
  if(pixel_up == nil and pixel_left == nil) then
    Recomputation_calculate_orig(inputNet, pixel, count)
  else if(pixel_up ~= nil and pixel_left ~= nil) then
     Recomputation_calculate_left_up(inputNet, pixel, pixel_left, pixel_up, count)
  else if(pixel_up ~= nil) then
    Recomputation_calculate_up(inputNet, pixel, pixel_up, count)
  else if(pixel_left ~= nil) then
    Recomputation_calculate_left(inputNet, pixel, pixel_left, count)
  end
  end
  end
  end

end

--print region for pixel in each layer
function PrintRegion(pixel, pixel_left, pixel_up)
   outfile:write(string.format("\n\n----------------------Printing Input Region for each Layer --------------------\n"))
   outfile:write(string.format("Layer\t\t\t\tPixel_left\t\t\tPixel\t\t\t  pixel_up\n"))
   for i = 1, count do
      
      outfile:write(string.format('Layer %d: \t\t', i ))

      if(pixel_left ~= nil) then
         outfile:write(string.format('(%d, %d) to (%d, %d) \t\t ' , pixel_left[i]['tl'][1], pixel_left[i]['tl'][2], pixel_left[i]['br'][1], pixel_left[i]['br'][2]))
      else
         outfile:write(string.format('                     \t\t '))
      end		

      outfile:write(string.format('(%d, %d) to (%d, %d) \t\t ', pixel[i]['tl'][1], pixel[i]['tl'][2], pixel[i]['br'][1], pixel[i]['br'][2]))

      
      if(pixel_up ~= nil) then
         outfile:write(string.format('(%d, %d) to (%d, %d) \t\t ' , pixel_up[i]['tl'][1], pixel_up[i]['tl'][2], pixel_up[i]['br'][1], pixel_up[i]['br'][2]))
      end

      outfile:write(string.format('\n'))

   end
end


--Finds out Recomputation happening between two pixels in single row
function layer_count(inputNet, pixel, pixel_left, count)
  local rows, cols    -- overlapping rows and cols at each layer   
  for i = 1, count do
       -- check for overlapping computation
      rows = math.max(0, pixel_left[i+1]['br'][1] - pixel_left[i+1]['tl'][1] + 1) 
      cols = math.max(0, pixel_left[i+1]['br'][2] - pixel[i+1]['tl'][2] + 1)
   
      if(rows == 0 or cols == 0) then
            return i-2
         else
         if(inputNet.modules[i].__typename == 'nn.SpatialZeroPadding') then
            return i-1
         end
      end
  end      
end

function sliding_window(pixel, pixel_r, pixel_d)
    pixel = calculate_roi(inputNet, count, pixel, shift)
    pixel_r = calculate_roi(inputNet, count, pixel_r, shift)
    pixel_d = calculate_roi(inputNet, count, pixel_d, shift)
    if(computation == 1) then
       SW_R = pixel_r[1]['tl'][2] - pixel[1]['tl'][2]
       SW_D = pixel_d[1]['tl'][1] - pixel[1]['tl'][1]

       SW_R1 = pixel_r[1]['tl'][2] - pixel[1]['tl'][2]
       SW_D1 = pixel_d[1]['tl'][1] - pixel[1]['tl'][1]
        
    else
       SW_R = pixel[1]['br'][2] - pixel[1]['tl'][2] + 1
       SW_D = pixel[1]['br'][1] - pixel[1]['tl'][1] + 1

       SW_R1 = pixel_r[1]['br'][2] - pixel[1]['br'][2]
       SW_D1 = pixel_d[1]['br'][1] - pixel[1]['br'][1]
       
    end

    --print(SW_R, SW_D, SW_R1, SW_D1)
end


--init function which looks upto reshape layer and calculate shift
function init()
    count = reshape_layer(inputNet)    
    channels = tonumber(opt.channels)
    W = tonumber(opt.W)
    H = tonumber(opt.H)
    input = torch.FloatTensor(channels, W, H)
    out = inputNet:forward(input)
    print(inputNet.modules)
    local multi_cones = 0
    if(multi_cones == 1) then

	    --shift set
	    local pixel1 = {}
	    local pixel2 = {}
	    -- top left and bottom right
	    local tab = {}
	    tab['tl'] = {0,0}
	    tab['br'] = {0,0}
	    pixel1[count+1] = tab
	    
	    local local_shift = calculate_shift(inputNet, count, pixel1, 0)
	    pixel1 = {}
	    tab['tl'] = {0,0}
	    tab['br'] = {0,0}
	    pixel1[count+1] = tab
	    tab = {}
	    tab['tl'] = {0,1}
	    tab['br'] = {0,1}
	    pixel2[count+1] = tab
	    pixel1 = calculate_roi(inputNet, count, pixel1, local_shift)
	    pixel2 = calculate_roi(inputNet, count, pixel2, local_shift)

	    count = layer_count(inputNet, pixel2, pixel1, count)
    end
    local pixel = {}
    local tab = {}	  
    tab['tl'] = {0,0}
    tab['br'] = {0,0}
    pixel[count+1] = tab    
    shift = calculate_shift(inputNet, count, pixel, 0) 
    --SW calculation
    pixel = {}
    tab = {}	  
    tab['tl'] = {0,0}
    tab['br'] = {0,0}
    pixel[count+1] = tab

    local pixel_r = {}
    tab = {}	  
    tab['tl'] = {0,1}
    tab['br'] = {0,1}
    pixel_r[count+1] = tab

    local pixel_d = {}
    tab = {}	  
    tab['tl'] = {1,0}
    tab['br'] = {1,0} 
    pixel_d[count+1] = tab
    sliding_window(pixel, pixel_r, pixel_d)
    --print(shift)
end

--print(inputNet.modules)
--print(inputNet)


init()   
if(multicone_stats == 1) then
	local map = cone_position(inputNet, count)
	print(map)
	map = nil
	map = {20, 0}
	local res = {}
	local cases = {}
	local total_solutions = back_track(inputNet, map, res, 1, cases, count)
    print('total_solutions', total_solutions)
   	
	outfile:write(string.format("Input Network\n\n\n"))
	outfile:write(string.format("%s\n", tostring(inputNet)))

	outfileCycle:write(string.format("Input Network\n\n\n"))		
	outfileCycle:write(string.format("%s\n", tostring(inputNet)))
	outfileCycle:write(string.format("\nAvailable DSP Resouces to distribute between different layers %d\n", DSP_LIMIT))
	outfileCycle:write(string.format("\nMaximum difference between any two layers execution cycle %d\n", Cycle_diff))
		
	--[[--selection_sort(res, 'cost')
    insertion_sort(res, 'cost')
    processed_res = res

    --print('pre_total_solutions', #processed_res)
	--res_cost is reuse cost
	outfile:write(string.format("\nReuse Cost (Onchip buffer size to store intermediate results)\n"))
    outfile:write(string.format("\n\t\t\t\t\t\t\t\t\t\t\tCone\t\t\tReuse Cost\n"))
	for i = 1, #processed_res do
	 outfile:write(string.format("%48s\t\t%10.2f\n", processed_res[i]['cone'], processed_res[i]['cost']/1024))
	end
    outfile:write(string.format("\n"))
	outfile:write(string.format("\n"))

    insertion_sort(processed_res, 'storage_cost')
	--res_storage is storage cost
	outfile:write(string.format("\nBandwidth Cost (Need to transfer this memory from onchip to offchip)\n"))
    outfile:write(string.format("\n\t\t\t\t\t\t\t\t\t\t\tCone\t\t\tBandwidth Cost\n"))	

	for i = 1, #processed_res do
	 outfile:write(string.format("%48s\t\t%10.2f\n", processed_res[i]['cone'], processed_res[i]['storage_cost']/1024))
	end

    outfile:write(string.format("\n"))
	outfile:write(string.format("\n"))
    insertion_sort(processed_res, 'total_cost')
	
	--res_cost_storage is combined cost
	outfile:write(string.format("\nCombined ReUse and Bandwidth cost\n"))
    outfile:write(string.format("\n\t\t\t\t\t\t\t\t\t\t\tCone\t\t\tReuse + Bandwidth Cost\n"))

	for i = 1, #processed_res do
	 outfile:write(string.format("%48s\t\t%10.2f\n", processed_res[i]['cone'], processed_res[i]['total_cost']/1024))
	end
    ]]--
    
    processed_res = res
    insertion_sort(processed_res, 'cost')

	outfile:write(string.format("\nAll Costs together\n"))
	outfile:write(string.format("\n\t\t\t\t\t\t\t\t\t\t\tCone\t\t\tReUse Cost\t\t\tIntermediate Storage Cost\t\t\tBandwidth\t\t\tWeight Cost\n"))
	
	for i = #processed_res, #processed_res do
	   outfile:write(string.format("%48s\t\t\t%10.2f\t\t\t%10.2f\t\t\t\t\t\t%10.2f\t\t\t%10.2f\n", processed_res[i]['cone'], processed_res[i]['cost']/1024, processed_res[i]['IScost']/1024, (processed_res[i]['storage_cost'])/1024, processed_res[i]['weight_cost']/1024))
		  
		local layerT = {}
		local countT = 1    
		outfileCycle:write("--------------------------------------------------------------")
	    outfileCycle:write(string.format("\n\nPyramid %20s \n\n", processed_res[i]['cone']))	    

		for c1 = 1, count do
			if(processed_res[i]['layer'][c1]) then
				layerT[countT] = processed_res[i]['layer'][c1]	
		  		countT = countT + 1
		  	end
		end

		TimeMultiplex(processed_res[i]['cone'], layerT, outfileCycle)
		collectgarbage()
    end
    
    outfile:write(string.format("\nParameters for each cone\n\n"))
	
    for i = 1, #processed_res do
        PrintParameters(inputNet, processed_res[i]['Mpyramid'], outfile)
        outfile:write(string.format("\n-----------------------------\n"))
    end
    
    if(opt.singlePyramid) then		    
		
       local single_cone = {20, 0}
       outSinglePyramid:write(string.format("Input Network\n\n\n"))
	   outSinglePyramid:write(string.format("%s\n", tostring(inputNet)))
	   FPGAPrintParameters(inputNet, single_cone, outSinglePyramid)
    end
	--print(res_cost)
else

	local S = inputNet.modules[count].output:size()
	outfile:write(string.format("----------------Input Network upto Reshape Layer----------------\n %s \n", tostring(inputNet)))
	outfile:write(string.format("\nOutput Size before Fully connected Layer %dX%dX%d \n", S[1], S[2], S[3]))
	local sh = shift
        --local sh
	if(computation == 1) then
	  outfile:write(string.format("\n----------------------ReComputation Model --------------------\n"))
	  outfile:write(string.format("Input Image Size %dX%dX%d \n", opt.channels, opt.H, opt.W))
	  total_multiplication = 0
	  total_addition = 0
	  total_comparison_Relu = 0
	  total_comparison_pool = 0

	  re_multiplication = 0
	  re_addition = 0
	  re_comparison_Relu = 0
	  re_comparison_pool = 0
	else
	  outfile:write(string.format("\n----------------------ReUse Model --------------------\n"))
	  outfile:write(string.format('No extra computation in this model but we use stored computation from previous pixel in left or above \n\n')) 
	  outfile:write(string.format("Input Image Size %dX%dX%d \n\n", opt.channels, opt.H, opt.W))
	  memory_used = {}
	  for i = 1, count do
	      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
		  memory_used[i] = {0, 0}  --left, up
	      end
	  end
	end  


	for r = 0, S[3]-1 do   
	  --outfile:write(string.format("\n\nRow %d of Fully Connected Layer\n\n", r+1))
	  if(computation ~= 1) then 
	    if(r == 1) then
	      outfile:write(string.format("\n\nRow %d Sliding Window %d\n\n", r+1, SW_D))
	    else if(r > 1) then
	      outfile:write(string.format("\n\nRow %d Sliding Window %d\n\n", r+1, SW_D1))
	    end
	  end

	  end   
	  for c = 0, S[2]-1 do
	   
	  if(computation ~= 1) then 

	   if(c == 1) then
	     outfile:write(string.format("\n\nSliding Window Right %d\n\n", SW_R))
	   else if(c > 1) then
	     outfile:write(string.format("\n\nSliding Window Right %d\n\n", SW_R1))
	   end
	   end
	  end

	   local pixel = {}    
	   tab = {}
	   tab['tl'] = {r,c}
	   tab['br'] = {r,c}
	   pixel[count+1] = tab
	   pixel = calculate_roi(inputNet, count, pixel, sh)

	   local pixel_left
	   local pixel_up
	   
	   if (c > 0) then
	     pixel_left = {} 
	     tab = {}
	     tab['tl'] = {r,c-1}
	     tab['br'] = {r,c-1}
	     pixel_left[count+1] = tab
	     pixel_left = calculate_roi(inputNet, count, pixel_left, sh)
	   end

	   if(r > 0) then
	     pixel_up = {}
	     tab = {}
	     tab['tl'] = {r-1,c}
	     tab['br'] = {r-1,c}
	     pixel_up[count+1] = tab
	     pixel_up = calculate_roi(inputNet, count, pixel_up, sh)
	   end
	  
	   outfile:write(string.format("\nPixel:  (%d,%d)\n", r, c))
	   PrintRegion(pixel, pixel_left, pixel_up)
	   if(computation == 1) then
	      Recomputation_calculate(inputNet, pixel, pixel_left, pixel_up, count)
	   else
	      ReUse_calculate(inputNet, pixel, pixel_left, pixel_up, count)
	   end   
	 end
	end

	if(computation == 1) then
	   outfile:write(string.format("\n-----------------------------------------------------------------\n"))
	   outfile:write(string.format('Recomputation Total:\n\n'))
	   outfile:write(string.format('Multiplications: %0.2f\n\n', (re_multiplication/total_multiplication)*100))
	   outfile:write(string.format('Additions: %0.2f\n\n', (re_addition/total_addition)*100))
	   outfile:write(string.format('Relu Comparison: %0.2f\n\n', (re_comparison_Relu/total_comparison_Relu)*100))
	   outfile:write(string.format('Max pool Comparison: %0.2f\n\n', (re_comparison_pool/total_comparison_pool)*100))


	   outfile:write(string.format('Recomputed Multiplications: %d\n\n', re_multiplication))
	   outfile:write(string.format('Recomputed Additions: %d\n\n', re_addition))


	   outfile:write(string.format('intermediate data to store: %d bytes\n\n', inputNet.modules[count].output:size()[1] * inputNet.modules[count].output:size()[2] * inputNet.modules[count].output:size()[3] * 4))   
	else 
	   outfile:write(string.format("\n-----------------------------------------------------------------\n"))
	   outfile:write(string.format('Memory Used Total:\n\n'))
	   model_memory = 0
	   
	   for key, value in pairs(memory_used) do
	      outfile:write(string.format('Layer %d: %s\n\n', key, tostring(inputNet.modules[key])))
	      outfile:write(string.format('Memory Used Left %d bytes\n', value[1]))   
	      outfile:write(string.format('Memory Used Up %d bytes\n\n', value[2]))
	      model_memory = model_memory + value[1] + value[2]   
	   end 
	   outfile:write(string.format('Total Memory Used by model %d bytes\n', model_memory))
	   outfile:write(string.format('intermediate data to store: %d bytes\n\n', inputNet.modules[count].output:size()[1] * inputNet.modules[count].output:size()[2] * inputNet.modules[count].output:size()[3] * 4))
	end

end	
