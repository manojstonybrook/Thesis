Debug_flag = 0 

function cone_position(model, count)
  local positions = {}
  positions[1] = 0
  local pos = 2
  local i = 1
  while((inputNet.modules[i].__typename ~= 'nn.SpatialConvolution')) do
        i = i + 1
  end
  --print(i)
  i = 1
  local j
  while(i < count) do
     --print('i'..tostring(i))
     j = i
     --if((inputNet.modules[j].__typename == 'nn.SpatialConvolution')) then
        --j = j + 1
        while((inputNet.modules[j].__typename ~= 'nn.SpatialConvolution') and (inputNet.modules[j].__typename ~= 'nn.SpatialMaxPooling')) do
             --print('j'..tostring(j))
            j = j + 1
        end
        positions[pos] = j
        pos = pos + 1
        j = j + 1
        i = j
     --else
       -- i = i + 1
     --end
        
  end
  local pos_rev = {}
  --print(positions)
  pos = 1
  for i = #positions, 1, -1 do
     pos_rev[pos] = positions[i]
     pos = pos + 1
  end
  --print(pos_rev)
  return pos_rev
end

function find_candidates(candidates, pos, k, map)
    local counter = 1
    for i = k+1, #map do
        candidates[counter] = map[i]
        pos[counter] = i
        counter = counter + 1 
    end
    --print(candidates)
end


function calcualte_Intermediate_storage(inputNet, pixel, bottom, top)
  local i = top
  local Istorage = 0
  --print(top..'-'..bottom)
  while(i <= bottom) do
    if(i == 1) then
        Istorage = Istorage + opt.channels *  (pixel[1]['br'][1] - pixel[1]['tl'][1] + 1) * (pixel[1]['br'][2] - pixel[1]['tl'][2] + 1) * 4
        --print(opt.channels..'X'..(pixel[1]['br'][1] - pixel[1]['tl'][1] + 1)..'X'..(pixel[1]['br'][2] - pixel[1]['tl'][2] + 1))
    else
        if(inputNet.modules[i].__typename ~= 'nn.Threshold') then
           Istorage = Istorage + inputNet.modules[i-1].output:size()[1] *  (pixel[i]['br'][1] - pixel[i]['tl'][1] + 1) * (pixel[i]['br'][2] - pixel[i]['tl'][2] + 1) * 4
         end
        --print(inputNet.modules[i-1].output:size()[1]..'X'..(pixel[i]['br'][1] - pixel[i]['tl'][1] + 1)..'X'..(pixel[i]['br'][2] - pixel[i]['tl'][2] + 1))
    end
    i = i + 1 
  end

  if(bottom ~= count) then
    --print(inputNet.modules[bottom+1].output:size()[1]..'X'..(pixel[bottom+1]['br'][1] - pixel[bottom+1]['tl'][1] + 1)..'X'..(pixel[bottom+1]['br'][2] - pixel[bottom+1]['tl'][2] + 1))
    Istorage = Istorage + inputNet.modules[bottom+1].output:size()[1] * (pixel[bottom+1]['br'][1] - pixel[bottom+1]['tl'][1] + 1) * (pixel[bottom+1]['br'][2] - pixel[bottom+1]['tl'][2] + 1) * 4
  else
    --print(inputNet.modules[bottom].output:size()[1]..'X'..(pixel[bottom+1]['br'][1] - pixel[bottom+1]['tl'][1] + 1)..'X'..(pixel[bottom+1]['br'][2] - pixel[bottom+1]['tl'][2] + 1))
    Istorage = Istorage + inputNet.modules[bottom].output:size()[1] * (pixel[bottom+1]['br'][1] - pixel[bottom+1]['tl'][1] + 1) * (pixel[bottom+1]['br'][2] - pixel[bottom+1]['tl'][2] + 1) * 4
  end 
  --print("\n")
  return Istorage
end

function calculate_storage(inputNet, pixel, pixel_up, count, top)
  local start = top
  local rows, cols, lrows, lcols, NRows, NCols, NewRows, NewCols
  --NewRows = pixel_left[start]['br'][1] - pixel_left[start]['tl'][1] + 1
  --NewCols = pixel[start]['br'][2] - pixel_left[start]['br'][2]
  	
  NewRows = pixel[start]['br'][1] - pixel_up[start]['br'][1]
  NewCols = pixel_up[start]['br'][2] - pixel_up[start]['br'][2] +1
  local total_reuse_mem = 0; local_reuse_mem_left = 0; local_reuse_mem_up = 0;
  local weight_storage = 0
  if(pixel_up[start]['br'][2] > pixel[start]['tl'][2]) then
    --print('Input image overlapping region:'..'('..pixel2[1]['tl'][1]..','..pixel2[1]['tl'][2]..')'..','..'('..pixel1[1]['br'][1]..','..pixel1[1]['br'][2]..')\n\n')
  else
    print('No region to reuse or recompute')
    return
  end

  --outfile:write(string.format('Reuse Region for each layer\n\n')) 
  local multiplications, additions, comparison, orows, ocols, crows
  i = start
  --print('memory',start, count)
  while(i <= count) do
      --[[rows = math.max(0, pixel_left[i]['br'][1] - pixel_left[i]['tl'][1] + 1)
      cols = math.max(0, pixel_left[i]['br'][2] - pixel[i]['tl'][2] + 1)
      NRows = pixel_left[i+1]['br'][1] - pixel_left[i+1]['tl'][1] + 1
      NCols = pixel[i+1]['br'][2] - pixel_left[i+1]['br'][2]
      orows = pixel_left[i]['br'][1] - pixel_left[i]['tl'][1] + 1
      ]]--	

      rows = math.max(0, pixel_up[i]['br'][1] - pixel[i]['tl'][1] + 1)   -- overlapping rows
      cols = math.max(0, pixel_up[i]['br'][2] - pixel_up[i]['tl'][2] + 1)
      NRows = pixel[i+1]['br'][1] - pixel_up[i+1]['br'][1]
      NCols = pixel_up[i+1]['br'][2] - pixel_up[i+1]['tl'][2] + 1

      orows = NRows
      
      if(rows == 0 or cols == 0) then
         break
      end
      
      if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
	    ocols = inputNet.modules[i].kW - inputNet.modules[i].dW 
        crows = inputNet.modules[i].kH - inputNet.modules[i].dH
        local size = inputNet.modules[i].weight:size()
        weight_storage = weight_storage + size[1] * size[2] * size[3] * size[4] * 4

	    if(i == 1) then
   	  
          local_reuse_mem_left = local_reuse_mem_left + opt.channels*orows*ocols*4
	      local_reuse_mem_up = local_reuse_mem_up + opt.channels * opt.W * crows * 4
          if(Debug_flag == 1) then
            print("Left", "channels", opt.channels, "rows=", orows, "cols", ocols, local_reuse_mem_left)
     	    print("UP", "channels", opt.channels, "rows=", crows, "cols", opt.W, local_reuse_mem_up)
     	  end	     
          total_reuse_mem = total_reuse_mem + local_reuse_mem_left + local_reuse_mem_up 
        	
    	else
          -- i-1 layer is combined and used for convolution	  	  
	  
	       local_reuse_mem_left = local_reuse_mem_left + inputNet.modules[i-1].output:size()[1] * orows* ocols* 4
           local_reuse_mem_up = local_reuse_mem_up + inputNet.modules[i-1].output:size()[1] * crows * inputNet.modules[i-1].output:size()[2]* 4

       if(Debug_flag == 1) then
            print("Left", "channels", inputNet.modules[i-1].output:size()[1], "rows=", orows, "cols", ocols, local_reuse_mem_left)
     	    print("UP", "channels", inputNet.modules[i-1].output:size()[1], "rows=", crows, "cols", inputNet.modules[i-1].output:size()[2],local_reuse_mem_left)
          end
          
     	  
          total_reuse_mem = total_reuse_mem + local_reuse_mem_left + local_reuse_mem_up
	end
        i = i + 1
      else if(inputNet.modules[i].__typename == 'nn.SpatialZeroPadding' and inputNet.modules[i+1].__typename == 'nn.SpatialConvolution') then
	  
		  ocols = inputNet.modules[i+1].kW - inputNet.modules[i+1].dW 
          crows = inputNet.modules[i+1].kH - inputNet.modules[i+1].dH
          local size = inputNet.modules[i+1].weight:size()
          weight_storage = weight_storage + size[1] * size[2] * size[3] * size[4] * 4
	
	  	  NRows = pixel[i]['br'][1] - pixel_up[i]['br'][1] 
    	  NCols = pixel_up[i]['br'][2] - pixel_up[i]['tl'][2] + 1
	      orows = NRows
	
	  if(i == 1) then 
         local_reuse_mem_left = local_reuse_mem_left + opt.channels*orows*ocols*4
         local_reuse_mem_up =  local_reuse_mem_up + opt.channels* opt.W * crows *4
         if(Debug_flag == 1) then
           print("Image Left", "channels", opt.channels, "rows=", orows, "cols", ocols, local_reuse_mem_left)
   	       print("Image UP", "channels", opt.channels, "rows=", crows, "cols", opt.W, local_reuse_mem_up)
         end
	     total_reuse_mem = total_reuse_mem + local_reuse_mem_left + local_reuse_mem_up 	       
       	
	  else	  	  
	     
         local_reuse_mem_left = local_reuse_mem_left + inputNet.modules[i-1].output:size()[1] * orows* ocols* 4 
         local_reuse_mem_up =  local_reuse_mem_up + inputNet.modules[i-1].output:size()[1] * crows* inputNet.modules[i-1].output:size()[2]* 4
         if(Debug_flag == 1) then
           print("Left", "channels", inputNet.modules[i-1].output:size()[1], "rows=", orows, "cols", ocols, local_reuse_mem_left)
   	       print("UP", "channels", inputNet.modules[i-1].output:size()[1], "rows=", crows, "cols", inputNet.modules[i-1].output:size()[2], local_reuse_mem_up)
	    end
     	  
  	     total_reuse_mem = total_reuse_mem + local_reuse_mem_left + local_reuse_mem_up        
	  end      
         i = i + 2 
     else
         i = i + 1
     end 
     
    end         
  end   
  --print(start, count, ((weight_storage)/1024)/1024)
  return local_reuse_mem_left, local_reuse_mem_up, weight_storage
end


function calculate_layer(inputNet, pixel_up, pixel, count, top)
  local start = top
  local NRows
  --NewRows = pixel_left[start]['br'][1] - pixel_left[start]['tl'][1] + 1
  --NewCols = pixel[start]['br'][2] - pixel_left[start]['br'][2]
  local layer = {}

  local i = start
  while(i <= count) do

--Assumption is that region to calculate is same
	if(inputNet.modules[i].__typename == 'nn.SpatialConvolution') then
	  NRows = pixel[i+1]['br'][1] - pixel_up[i+1]['br'][1]
	  local size = inputNet.modules[i].weight:size()
      local layerL = {}
      layerL['R'] = NRows
   	  layerL['C'] = NRows
   	  layerL['M'] = size[1]
   	  layerL['N'] = size[2]
   	  layerL['K'] = inputNet.modules[i].kH
   	  layerL['S'] = inputNet.modules[i].dH
	  layer[i] = layerL   	     
	end    
	i = i+1
  end
	
  return layer
end


function cost_calculate_step(model, bottom, top)
    local pixel1 = {}
    local pixel2 = {}
    -- top left and bottom right
    local midx = math.floor(inputNet.modules[bottom].output:size()[2]/2)
    local midy = math.floor(inputNet.modules[bottom].output:size()[3]/2)
    
    local tab = {}
    tab['tl'] = {midy, midx}
    tab['br'] = {midy, midx}
    pixel1[bottom+1] = tab
    
    tab = {}
    tab['tl'] = {midy+1, midx}
    tab['br'] = {midy+1, midx}
    pixel2[bottom+1] = tab
    pixel1 = calculate_roi(inputNet, bottom, pixel1, nil, top)
    pixel2 = calculate_roi(inputNet, bottom, pixel2, nil, top)

    local cost_left, cost_up, weight_storage = calculate_storage(inputNet, pixel1, pixel2, bottom, top)
    local layers = calculate_layer(inputNet, pixel1, pixel2, bottom, top) 
	
    tab = {}
    tab['tl'] = {0, 0}
    tab['br'] = {0, 0}
    pixel1 = {}
    pixel1[bottom+1] = tab
    pixel1 = calculate_roi(inputNet, bottom, pixel1, nil, top)
	local Istorage = calcualte_Intermediate_storage(inputNet, pixel1, bottom, top)	   
    return cost_left, cost_up, Istorage, weight_storage, layers
end

function cost_calculate(model, cases)
    local cost = 0
    local storage_cost = 0
    local weight_cost
    local step_cost_left, step_cost_up, step_cost, step_storage_cost
    local Istorage = 0
    local step_cost_left_max = 0; step_cost_up_max = 0; weight_cost_max = 0; Istorage_max = 0;
    step_storage_cost = 0

    if(Debug_flag == 1) then
      print('New cones')
    end
    
	layerG = {}
	for b = 1, #cases - 1 do
	  local bottom = cases[b]
	  local top =  cases[b+1]
	  local test =  cases[b+1]
      local layerL    
      local local_str = bottom..'-'..top
      if(dp_table[local_str]~=nil) then

         step_cost_left_max = math.max(step_cost_left_max, dp_table[local_str]['step_cost_left'])
	     step_cost_up_max = math.max(step_cost_up_max, dp_table[local_str]['step_cost_up'])
         weight_cost_max = math.max(weight_cost_max, dp_table[local_str]['weight_cost'])
         Istorage_max = math.max(Istorage_max, dp_table[local_str]['IScost'])
         storage_cost = storage_cost + dp_table[local_str]['step_storage_cost']
		 layerL = dp_table[local_str]['layer']  
      else
		   if(top == 0) then
		   	top = 1
			step_storage_cost = opt.channels * opt.W * opt.H * 4
		  else 
			step_storage_cost = inputNet.modules[top].output:size()[1] * inputNet.modules[top].output:size()[2] * inputNet.modules[top].output:size()[3] * 4 
            -- multiply by 2 because of load and store
            step_storage_cost = step_storage_cost * 2
			top = top + 1
		  end

		  step_cost_left, step_cost_up, Istorage, weight_cost, layerL = cost_calculate_step(model, bottom, top)
		  --print(layerL)
	      step_cost_left_max = math.max(step_cost_left_max, step_cost_left)
	      step_cost_up_max = math.max(step_cost_up_max, step_cost_up)
          weight_cost_max = math.max(weight_cost_max, weight_cost)
          Istorage_max   =   math.max(Istorage_max, Istorage)
	      if(Debug_flag == 1) then
			print(bottom, test, top, step_cost_left, step_cost_up)
	      end
		  --cost = cost + step_cost
		  storage_cost = storage_cost + step_storage_cost
          local tab = {}
          tab['step_cost_left'] = step_cost_left 
          tab['step_cost_up'] = step_cost_up
          tab['IScost'] = Istorage
          tab['step_storage_cost'] = step_storage_cost
          tab['weight_cost'] = weight_cost
		  tab['layer'] = layerL
          dp_table[local_str] = tab
		  --print(bottom, top, step_cost, cost, step_storage_cost, storage_cost)
      end

	 for l = 1, #model.modules do
	   if(layerL[l]~=nil) then 
      	 layerG[l] = layerL[l]
       end
     end

	end

    cost = step_cost_left_max + step_cost_up_max
    if(Debug_flag == 1) then
	     print(cases, step_cost_left_max, step_cost_up_max)
    end
    
	--print(layerG)   
    return cost, storage_cost, Istorage_max, weight_cost_max, layerG
    
end

total_solutions = 0
dp_table = {}
unique_solutions = {} --cost, solution_index, storage_cost
--optimize = 1
function back_track(model, map, res, k, cases, count)
   cases[#cases + 1] = map[k]
   if(k == #map) then
      --if(total_solutions == 0) then
      local cost, storage_cost, Istorage, weight_cost, layerG = cost_calculate(model, cases)
      local case_str = ''
      local Mpyramid = {}
      for i = 1, #cases-1 do
        case_str = case_str..tostring(cases[i])..'-'
        Mpyramid[i] = cases[i]
      end
      Mpyramid[#Mpyramid+1] = 0
      case_str = case_str..'0'
      --[[local reuse_cost = cost + weight_cost
      local bw_cost = storage_cost + weight_cost]]--
      
	  local reuse_cost = cost
      local bw_cost = storage_cost + inputNet.modules[count].output:size()[1] * inputNet.modules[count].output:size()[2] * inputNet.modules[count].output:size()[3] * 4
 
     if(optimize == 0) then
         tab = {}      
      	 tab['cost'] = reuse_cost
      	 tab['storage_cost'] = bw_cost 
		 total_solutions = total_solutions + 1
		 tab['cone'] = case_str
		 tab['IScost'] = Istorage
		 tab['total_cost'] = tab['cost'] + tab['storage_cost'] 
		 tab['weight_cost'] = weight_cost
		 tab['layer'] = layerG
		 tab['Mpyramid'] = Mpyramid
		 res[total_solutions] = tab
         
	     tab = {}
         tab['total_cost'] = reuse_cost + bw_cost 
		 tab['index'] = total_solutions
		 unique_solutions[reuse_cost] = tab

	else
      if(unique_solutions[reuse_cost] == nil) then
         tab = {}      
      	 tab['cost'] = reuse_cost
      	 tab['storage_cost'] = bw_cost
      	 tab['IScost'] = Istorage 
		 total_solutions = total_solutions + 1
		 tab['cone'] = case_str
		 tab['total_cost'] = tab['cost'] + tab['storage_cost'] 
		 tab['weight_cost'] = weight_cost
		 tab['layer'] = layerG
         tab['Mpyramid'] = Mpyramid
		 res[total_solutions] = tab
         
	     tab = {}
         tab['total_cost'] = reuse_cost + bw_cost 
		 tab['index'] = total_solutions
		 unique_solutions[reuse_cost] = tab 
		 --end
      else
         if(unique_solutions[reuse_cost]['total_cost'] > (reuse_cost+bw_cost)) then
		     tab = {}      
      	     tab['cost'] = reuse_cost
      	 	 tab['storage_cost'] = bw_cost 
			 tab['cone'] = case_str
			 tab['IScost'] = Istorage
			 tab['total_cost'] = tab['cost'] + tab['storage_cost'] 
			 tab['weight_cost'] = weight_cost
			 tab['Mpyramid'] = Mpyramid
			 tab['layer'] = layerG 		
             local index = unique_solutions[reuse_cost]['index'] 
			 res[index] = tab
			 
			 tab = {}
			 tab['total_cost'] = reuse_cost + bw_cost  
			 tab['index'] = index
			 unique_solutions[reuse_cost] = tab 			   
         end

      end
end 
--]]--
      return
   end

   local candidates = {}
   local pos = {}
   find_candidates(candidates, pos, k, map)
   for i = 1, #candidates do
       back_track(model, map, res, pos[i], cases, count)
       cases[#cases] = nil 
   end

   return total_solutions   
end


function selection_sort(tab, case)
   local index, min, i
   local temp = {}
   for i = 1, #tab-1 do
      min = tab[i][case]
      index = i
      for j = i+1, #tab do
          if(tab[j][case] < min) then
             min = tab[j][case]
             index = j
          end
      end
      temp = tab[i]
      tab[i] = tab[index]
      tab[index] = temp
   end
end

--[[
/* Function to sort an array using insertion sort*/
void insertionSort(int arr[], int n)
{
   int i, key, j;
   for (i = 1; i < n; i++)
   {
       key = arr[i];
       j = i-1;
 
       /* Move elements of arr[0..i-1], that are
          greater than key, to one position ahead
          of their current position */
       while (j >= 0 && arr[j] > key)
       {
           arr[j+1] = arr[j];
           j = j-1;
       }
       arr[j+1] = key;
   }
}
]]--
function insertion_sort(tab, case)
   local key, index, temp, i
   for i = 2, #tab do
      temp = tab[i]
      key = tab[i][case]
      index = i-1
      while(index >= 1 and tab[index][case] > key) do
          tab[index + 1] = tab[index]
          index = index - 1
      end
      tab[index+1] = temp
   end
   return tab
end


function pre_process(tab, case, cmp)
  local res = {}
  local count = 1
  local i = 1
  while(i <= #tab) do
     j = i + 1
     --print('HERE'..i)
     local min_cmp = tab[i][cmp]
     local min_index = i
     while(1) do       
       if(tab[j]~=nil and tab[j][case] == tab[i][case]) then
       
     	  if(tab[j][cmp] < min_cmp) then
            min_cmp = tab[j][cmp]
            min_index = j
          end
          j = j + 1
       
       else
          i = j
          break
       end
     end
     res[count] = tab[min_index]
     count = count + 1
     --print(count) 
  end 
  return res
end

