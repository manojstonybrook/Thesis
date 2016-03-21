require('nn')

function FPGAParameters(Network, bottom, top, outfile)

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
    
    outfile:write(string.format("const int strideX = %d; \nconst int strideY = %d\n", stride, stride))
    
    local countConv = 1
    local countPool = 1
    local countPad = 1
    for i = top, bottom+1 do
       if(i == 1) then
         outfile:write(string.format("const int inputX = %d;\nconst int inputY = %d;\n", pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
       else if(Network.modules[i-1].__typename == 'nn.SpatialConvolution') then   
           outfile:write(string.format("const int f%dK =  %d;\nconst int f%dS =  %d;\n", countConv, Network.modules[i-1].kH, countConv, Network.modules[i-1].dH))
           outfile:write(string.format("const int TM%d =  %d;\nconst int TN%d =  %d;\n", countConv, Network.modules[i-1].nOutputPlane, countConv, Network.modules[i-1].nInputPlane)) 
           outfile:write(string.format("const int filter%d[%d][%d][%d][%d];\n\n", countConv, Network.modules[i-1].nOutputPlane, Network.modules[i-1].nInputPlane, Network.modules[i-1].kH, Network.modules[i-1].kW))                
           countConv = countConv + 1 
       else if(Network.modules[i-1].__typename == 'nn.SpatialMaxPooling') then   
           outfile:write(string.format("const float pool%dH = %d;\nconst float pool%dS = %d;\n\n", countPool, Network.modules[i-1].kH, countPool, Network.modules[i-1].dH))
           countPool = countPool + 1
       else if(Network.modules[i-1].__typename == 'nn.SpatialZeroPadding') then   
           outfile:write(string.format("const float pad%d = %d\n\n", countPad, Network.modules[i-1].pad_l)) 
           countPad = countPad + 1       
       else
       
       end
       end
       end
       end   
   end

    
    countConv = 1
    countPool = 1
    countPad = 1
    
    
    for i = top, bottom+1 do
       if(i == 1) then
         outfile:write(string.format("float input[%d][%d][%d]\n", opt.channels, pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
       else if(Network.modules[i-1].__typename == 'nn.SpatialConvolution') then   
           outfile:write(string.format("float conv%d[%d][%d][%d]\n", countConv, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
           countConv = countConv + 1 
       else if(Network.modules[i-1].__typename == 'nn.SpatialMaxPooling') then   
           outfile:write(string.format("float pool%d[%d][%d][%d]\n", countPool, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
           countPool = countPool + 1
       else if(Network.modules[i-1].__typename == 'nn.SpatialZeroPadding') then   
           outfile:write(string.format("float pad%d[%d][%d][%d]\n", countPad, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1)) 
           countPad = countPad + 1       
       else
       
       end
       end
       end
       end   
   end

    
      
    for i = top, bottom+1 do
       if(i == 1) then
         outfile:write(string.format("float input[%d][%d][%d]\n", opt.channels, pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
       else if(Network.modules[i-1].__typename == 'nn.SpatialConvolution') then   
           outfile:write(string.format("float conv%d[%d][%d][%d]\n", countConv, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
           countConv = countConv + 1 
       else if(Network.modules[i-1].__typename == 'nn.SpatialMaxPooling') then   
           outfile:write(string.format("float pool%d[%d][%d][%d]\n", countPool, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
           countPool = countPool + 1
       else if(Network.modules[i-1].__typename == 'nn.SpatialZeroPadding') then   
           outfile:write(string.format("float pad%d[%d][%d][%d]\n", countPad, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1)) 
           countPad = countPad + 1       
       else
       
       end
       end
       end
       end   
   end
    
    outfile:write(string.format("\n\n"))
end

function FPGAPrintParameters(Network, str, outfile)

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
       FPGAParameters(Network, bottom, top, outfile)
    end
    
end

