require('nn')

function Parameters(Network, bottom, top, outfile)

    local pixel1 = {}
    local pixel2 = {}
    
    local tab = {}
    tab['tl'] = {0, 0}
    tab['br'] = {0, 0}
    pixel1[bottom+1] = tab
    
    tab = {}
    tab['tl'] = {0, 1}
    tab['br'] = {0, 1}
    pixel2[bottom+1] = tab
    
    outfile:write(string.format("%10s - %s Pyramid \n\n", top-1, bottom))
    
    --print(Network)
    if(top == 0) then
      top = 1
    end 
    
    pixel1 = calculate_roi(Network, bottom, pixel1, nil, top)
    pixel2 = calculate_roi(Network, bottom, pixel2, nil, top)
    
    local stride1 = pixel2[top]['br'][2] 
    local stride2 = pixel1[top]['br'][2]
    local stride = stride1 - stride2
    
    outfile:write(string.format("strideX = %d; strideY = %d\n", stride, stride))
      
    for i = top, bottom+1 do
       if(i == 1) then
         outfile:write(string.format("0. %5s \t\t\t\t\t\t\t\t%dX%dX%d\n", "Input", opt.channels, pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
       else
         if(Network.modules[i-1].__typename == 'nn.SpatialConvolution' or Network.modules[i-1].__typename == 'nn.SpatialMaxPooling' or Network.modules[i-1].__typename == 'nn.SpatialZeroPadding') then   
           outfile:write(string.format("%d. %5s \t\t\t\t%dX%dX%d\n", i-1, Network.modules[i-1].__typename, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
         end  
       end
    end
    
    outfile:write(string.format("\n\n"))
end

function PrintParameters(Network, str, outfile)

    outfile:write(string.format("Pyramid  "))
    for i = 1, #str-1 do
      outfile:write(string.format("%d-", str[i]))  
    end
    outfile:write(string.format("%d", str[#str]))
    outfile:write(string.format("\n\n"))
   
    local pyramid = {}
    for i = #str, 1, -1 do
       pyramid[#pyramid+1] = str[i]
    end
    
 
    for l = 1, #pyramid-1 do
       local top = pyramid[l]+1
       local bottom = pyramid[l+1]
       Parameters(Network, bottom, top, outfile)
    end
    
    
end

