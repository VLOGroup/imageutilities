[mem_size] = mem_size_mex(size, channels)
% [stride] = mem_size_mex(size, color_flag)
%
% Mex wrapper to get the memory layout of allocated gpu memory.
%
% Input:
%    size      ... desired memory size; can be 1D, 2D and 3D sizes
%    channels  ... Color channels (interleaved formatting with vector types)
%
% Output:
%    mem_size  ... Really allocated memory layout (including padding)